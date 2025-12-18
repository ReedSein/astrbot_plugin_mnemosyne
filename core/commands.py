# Mnemosyne 插件的命令处理函数实现
# (注意：装饰器已移除，函数接收 self)

import json
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .constants import MAX_TOTAL_FETCH_RECORDS, PRIMARY_FIELD_NAME
from .security_utils import safe_build_milvus_expression, validate_session_id

if TYPE_CHECKING:
    from ..main import Mnemosyne


async def list_collections_cmd_impl(self: "Mnemosyne", event: AstrMessageEvent):
    """[实现] 列出当前 Milvus 实例中的所有集合"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或未连接。")
        return
    try:
        collections = self.milvus_manager.list_collections()
        if collections is None:
            yield event.plain_result("⚠️ 获取集合列表失败，请检查日志。")
            return
        if not collections:
            response = "当前 Milvus 实例中没有找到任何集合。"
        else:
            response = "当前 Milvus 实例中的集合列表：\n" + "\n".join(
                [f"📚 {col}" for col in collections]
            )
            if self.collection_name in collections:
                response += f"\n\n当前插件使用的集合: {self.collection_name}"
            else:
                response += (
                    f"\n\n⚠️ 当前插件配置的集合 '{self.collection_name}' 不在列表中！"
                )
        yield event.plain_result(response)
    except Exception as e:
        logger.error(f"执行 'memory list' 命令失败: {str(e)}", exc_info=True)
        yield event.plain_result(f"⚠️ 获取集合列表时出错: {str(e)}")


async def delete_collection_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    collection_name: str,
    confirm: str | None = None,
):
    """[实现] 删除指定的 Milvus 集合及其所有数据"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或未连接。")
        return

    is_current_collection = collection_name == self.collection_name
    warning_msg = ""
    if is_current_collection:
        warning_msg = f"\n\n🔥🔥🔥 警告：您正在尝试删除当前插件正在使用的集合 '{collection_name}'！这将导致插件功能异常，直到重新创建或更改配置！ 🔥🔥🔥"

    if confirm != "--confirm":
        yield event.plain_result(
            f"⚠️ 操作确认 ⚠️\n"
            f"此操作将永久删除 Milvus 集合 '{collection_name}' 及其包含的所有数据！此操作无法撤销！\n"
            f"{warning_msg}\n\n"
            f"如果您确定要继续，请再次执行命令并添加 `--confirm` 参数:\n"
            f"`/memory drop_collection {collection_name} --confirm`"
        )
        return

    try:
        sender_id = event.get_sender_id()
        logger.warning(f"管理员 {sender_id} 请求删除集合: {collection_name} (确认执行)")
        if is_current_collection:
            logger.critical(
                f"管理员 {sender_id} 正在删除当前插件使用的集合 '{collection_name}'！"
            )

        success = self.milvus_manager.drop_collection(collection_name)
        if success:
            msg = f"✅ 已成功删除 Milvus 集合 '{collection_name}'。"
            if is_current_collection:
                msg += "\n插件使用的集合已被删除，请尽快处理！"
            yield event.plain_result(msg)
            logger.warning(f"管理员 {sender_id} 成功删除了集合: {collection_name}")
            if is_current_collection:
                logger.error(
                    f"插件当前使用的集合 '{collection_name}' 已被删除，相关功能将不可用。"
                )
        else:
            yield event.plain_result(
                f"⚠️ 删除集合 '{collection_name}' 的请求已发送，但 Milvus 返回失败。请检查 Milvus 日志获取详细信息。"
            )

    except Exception as e:
        logger.error(
            f"执行 'memory drop_collection {collection_name}' 命令时发生严重错误: {str(e)}",
            exc_info=True,
        )
        yield event.plain_result(f"⚠️ 删除集合时发生严重错误: {str(e)}")


async def list_records_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    collection_name: str | None = None,
    limit: int = 5,
):
    """[实现] 查询指定集合的最新记忆记录 (按创建时间倒序，自动获取最新)"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或未连接。")
        return

    # 获取当前会话的 session_id (如果需要按会话过滤)
    # 直接使用 unified_msg_origin 作为 session_id，与存储时保持一致
    session_id = event.unified_msg_origin
    # session_id = "session_1" # 如果要测试特定会话或无会话过滤，可以在这里硬编码或设为 None

    target_collection = collection_name or self.collection_name

    # M16 修复: 增强 limit 参数类型和范围验证
    try:
        # 确保 limit 是整数类型
        limit = int(limit)
    except (ValueError, TypeError):
        yield event.plain_result(f"⚠️ limit 参数必须是有效的整数，当前值: {limit}")
        logger.warning(
            f"用户提供了无效的 limit 参数: {limit} (类型: {type(limit).__name__})"
        )
        return

    # 验证范围
    if limit <= 0 or limit > 50:
        # 限制用户请求的显示数量
        yield event.plain_result("⚠️ 显示数量 (limit) 必须在 1 到 50 之间。")
        return

    try:
        if not self.milvus_manager.has_collection(target_collection):
            yield event.plain_result(f"⚠️ 集合 '{target_collection}' 不存在。")
            return

        # 构建查询表达式 - 仅基于 session_id (如果需要)
        if session_id:
            # 安全检查：验证 session_id 格式
            if not validate_session_id(session_id):
                yield event.plain_result("⚠️ 会话 ID 格式无效，无法查询记录。")
                logger.warning(f"尝试使用无效的 session_id 查询记录: {session_id}")
                return

            # 如果有会话ID，则按会话ID过滤（使用安全的表达式构建）
            try:
                expr = safe_build_milvus_expression("session_id", session_id, "==")
            except ValueError as e:
                yield event.plain_result(f"⚠️ 构建查询表达式失败: {e}")
                logger.error(f"构建查询表达式时出错: {e}")
                return

            logger.info(
                f"将按会话 ID '{session_id}' 过滤并查询所有相关记录 (上限 {MAX_TOTAL_FETCH_RECORDS} 条)。"
            )
        else:
            # 如果没有会话ID上下文，查询所有记录
            expr = f"{PRIMARY_FIELD_NAME} >= 0"
            logger.info(
                "未指定会话 ID，将查询集合 '{target_collection}' 中的所有记录 (上限 {MAX_TOTAL_FETCH_RECORDS} 条)。"
            )
            # 或者，如果您的 milvus_manager 支持空表达式查询所有，则 expr = "" 或 None

        # logger.debug(f"查询集合 '{target_collection}' 记录: expr='{expr}'") # 上面已有更具体的日志
        output_fields = [
            "content",
            "create_time",
            "session_id",
            "personality_id",
            PRIMARY_FIELD_NAME,
        ]

        logger.debug(
            f"准备查询 Milvus: 集合='{target_collection}', 表达式='{expr}', 限制={limit},输出字段={output_fields}, 总数上限={MAX_TOTAL_FETCH_RECORDS}"
        )

        # 直接使用 Milvus 的 offset 和 limit 参数进行分页查询
        # records = self.milvus_manager.query(
        #     collection_name=target_collection,
        #     expression=expr,
        #     output_fields=output_fields,
        #     limit=limit,
        #     offset=offset,  # 直接使用函数参数 offset
        # )

        # 重要的修改：移除 Milvus query 的 offset 和 limit 参数，使用总数上限作为 Milvus 的 limit
        fetched_records = self.milvus_manager.query(
            collection_name=target_collection,
            expression=expr,
            output_fields=output_fields,
            limit=MAX_TOTAL_FETCH_RECORDS,  # 使用总数上限作为 Milvus 的 limit
        )

        # 检查查询结果
        if fetched_records is None:
            # 查询失败，milvus_manager.query 通常会返回 None 或抛出异常
            logger.error(
                f"查询集合 '{target_collection}' 失败，milvus_manager.query 返回 None。"
            )
            yield event.plain_result(
                f"⚠️ 查询集合 '{target_collection}' 记录失败，请检查日志。"
            )
            return

        if not fetched_records:
            # 查询成功，但没有返回任何记录
            session_filter_msg = f"在会话 '{session_id}' 中" if session_id else ""
            logger.info(
                f"集合 '{target_collection}' {session_filter_msg} 没有找到任何匹配的记忆记录。"
            )
            yield event.plain_result(
                f"集合 '{target_collection}' {session_filter_msg} 中没有找到任何匹配的记忆记录。"
            )
            return
        # 检查是否达到了总数上限
        if len(fetched_records) >= MAX_TOTAL_FETCH_RECORDS:
            logger.warning(
                f"查询到的记录数量达到总数上限 ({MAX_TOTAL_FETCH_RECORDS})，可能存在更多未获取的记录，导致无法找到更旧的记录，但最新记录应该在获取范围内。"
            )
            yield event.plain_result(
                f"ℹ️ 警告：查询到的记录数量已达到系统获取最新记录的上限 ({MAX_TOTAL_FETCH_RECORDS})。如果记录非常多，可能无法显示更旧的内容，但最新记录应该已包含在内。"
            )

        logger.debug(f"成功获取到 {len(fetched_records)} 条原始记录用于排序。")
        # --- 在获取全部结果后进行排序 (按创建时间倒序) ---
        # 这确保了排序是基于所有获取到的记录，找到真正的最新记录
        try:
            # 使用 lambda 表达式按 create_time 字段排序，如果字段不存在或为 None，默认为 0
            fetched_records.sort(
                key=lambda x: x.get("create_time", 0) or 0, reverse=True
            )
            logger.debug(
                f"已将获取到的 {len(fetched_records)} 条记录按 create_time 降序排序。"
            )
        except Exception as sort_e:
            logger.warning(
                f"对查询结果进行排序时出错: {sort_e}。显示顺序可能不按时间排序。"
            )
            # 如果排序失败，继续处理，但不保证按时间顺序

        # --- 在排序后获取最前的 limit 条记录 ---
        # 从排序后的 fetched_records 中取出最前的 limit 条记录
        display_records = fetched_records[:limit]

        # display_records 不会为空，除非 fetched_records 本身就为空，
        # 而 fetched_records 为空的情况已经在前面处理过了。

        # 准备响应消息
        total_fetched = len(fetched_records)
        display_count = len(display_records)
        # 消息提示用户这是最新的记录
        response_lines = [
            f"📜 集合 '{target_collection}' 的最新记忆记录 (共获取 {total_fetched} 条进行排序, 显示最新的 {display_count} 条):"
        ]

        # 格式化每条记录以供显示
        # 使用 enumerate 从 1 开始生成序号
        for i, record in enumerate(display_records, start=1):
            ts = record.get("create_time")
            try:
                # 根据 Milvus 文档，Query 结果中的 time 是 float 类型的 Unix 时间戳（秒）。
                time_str = (
                    datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    if ts is not None  # 检查 ts 是否存在且不是 None
                    else "未知时间"
                )
            except (TypeError, ValueError, OSError) as time_e:
                # 处理无效或无法解析的时间戳
                logger.warning(
                    f"记录 {record.get(PRIMARY_FIELD_NAME, '未知ID')} 的时间戳 '{ts}' 无效或解析错误: {time_e}"
                )
                time_str = f"无效时间戳({ts})" if ts is not None else "未知时间"

            content = record.get("content", "内容不可用")
            # 截断过长的内容以优化显示
            content_preview = content[:200] + ("..." if len(content) > 200 else "")
            record_session_id = record.get("session_id", "未知会话")
            persona_id = record.get("personality_id", "未知人格")
            pk = record.get(PRIMARY_FIELD_NAME, "未知ID")  # 获取主键

            response_lines.append(
                f"#{i} [ID: {pk}]\n"  # 使用从 1 开始的序号
                f"  时间: {time_str}\n"
                f"  人格: {persona_id}\n"
                f"  会话: {record_session_id}\n"
                f"  内容: {content_preview}"
            )

        # 发送格式化后的结果
        yield event.plain_result("\n\n".join(response_lines))

    except Exception as e:
        # 捕获所有其他潜在异常
        logger.error(
            f"执行 'memory list_records' 命令时发生意外错误 (集合: {target_collection}): {str(e)}",
            exc_info=True,  # 记录完整的错误堆栈
        )
        yield event.plain_result("⚠️ 查询记忆记录时发生内部错误，请联系管理员。")


async def delete_session_memory_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    session_id: str,
    confirm: str | None = None,
):
    """[实现] 删除指定会话 ID 相关的所有记忆信息"""
    if not self.milvus_manager or not self.milvus_manager.is_connected():
        yield event.plain_result("⚠️ Milvus 服务未初始化或未连接。")
        return

    if not session_id or not session_id.strip():
        yield event.plain_result("⚠️ 请提供要删除记忆的会话 ID (session_id)。")
        return

    session_id_to_delete = session_id.strip().strip('"`')

    # 安全检查：验证 session_id 格式，防止SQL注入
    if not validate_session_id(session_id_to_delete):
        yield event.plain_result("⚠️ 会话 ID 格式无效，无法执行删除操作。")
        logger.warning(f"尝试删除无效的 session_id: {session_id_to_delete}")
        return

    if confirm != "--confirm":
        yield event.plain_result(
            f"⚠️ 操作确认 ⚠️\n"
            f"此操作将永久删除会话 ID '{session_id_to_delete}' 在集合 '{self.collection_name}' 中的所有记忆信息！此操作无法撤销！\n\n"
            f"要确认删除，请再次执行命令并添加 `--confirm` 参数:\n"
            f'`/memory delete_session_memory "{session_id_to_delete}" --confirm`'
        )
        return

    try:
        collection_name = self.collection_name

        # 使用安全的表达式构建方法，防止注入
        try:
            expr = safe_build_milvus_expression(
                "session_id", session_id_to_delete, "=="
            )
        except ValueError as e:
            yield event.plain_result(f"⚠️ 构建删除表达式失败: {e}")
            logger.error(f"构建删除表达式时出错: {e}")
            return

        sender_id = event.get_sender_id()
        logger.warning(
            f"管理员 {sender_id} 请求删除会话 '{session_id_to_delete}' 的所有记忆 (集合: {collection_name}, 表达式: '{expr}') (确认执行)"
        )

        mutation_result = self.milvus_manager.delete(
            collection_name=collection_name, expression=expr
        )

        if mutation_result:
            delete_pk_count = (
                mutation_result.delete_count
                if hasattr(mutation_result, "delete_count")
                else "未知"
            )
            logger.info(
                f"已发送删除会话 '{session_id_to_delete}' 记忆的请求。返回的删除计数（可能不准确）: {delete_pk_count}"
            )
            try:
                logger.info(
                    f"正在刷新 (Flush) 集合 '{collection_name}' 以应用删除操作..."
                )
                self.milvus_manager.flush([collection_name])
                logger.info(f"集合 '{collection_name}' 刷新完成。删除操作已生效。")
                yield event.plain_result(
                    f"✅ 已成功删除会话 ID '{session_id_to_delete}' 的所有记忆信息。"
                )
            except Exception as flush_err:
                logger.error(
                    f"刷新集合 '{collection_name}' 以应用删除时出错: {flush_err}",
                    exc_info=True,
                )
                yield event.plain_result(
                    f"⚠️ 已发送删除请求，但在刷新集合使更改生效时出错: {flush_err}。删除可能未完全生效。"
                )
        else:
            yield event.plain_result(
                f"⚠️ 删除会话 ID '{session_id_to_delete}' 记忆的请求失败。请检查 Milvus 日志。"
            )

    except Exception as e:
        logger.error(
            f"执行 'memory delete_session_memory' 命令时发生严重错误 (Session ID: {session_id_to_delete}): {str(e)}",
            exc_info=True,
        )
        yield event.plain_result(f"⚠️ 删除会话记忆时发生严重错误: {str(e)}")


async def get_session_id_cmd_impl(self: "Mnemosyne", event: AstrMessageEvent):
    """[实现] 获取当前与您对话的会话 ID"""
    try:
        # 直接使用 unified_msg_origin 作为 session_id，与存储时保持一致
        session_id = event.unified_msg_origin
        if session_id:
            yield event.plain_result(f"当前会话 ID: {session_id}")
        else:
            yield event.plain_result(
                "🤔 无法获取当前会话 ID。可能还没有开始对话，或者会话已结束/失效。"
            )
            logger.warning(
                f"用户 {event.get_sender_id()} 在 {event.unified_msg_origin} 尝试获取 session_id 失败。"
            )
    except Exception as e:
        logger.error(f"执行 'memory get_session_id' 命令失败: {str(e)}", exc_info=True)
        yield event.plain_result(f"⚠️ 获取当前会话 ID 时发生错误: {str(e)}")


async def init_memory_system_cmd_impl(
    self: "Mnemosyne",
    event: AstrMessageEvent,
    force: str | None = None,
):
    """[实现] 初始化或重新初始化记忆系统"""
    if not self.milvus_manager:
        yield event.plain_result("⚠️ Milvus 服务未初始化。")
        return

    # 尝试确保连接 - MilvusManager 使用延迟连接，首次操作时才会真正连接
    try:
        # 通过调用一个轻量级操作来触发连接（如果尚未连接）
        if not self.milvus_manager.is_connected():
            # 尝试连接
            self.milvus_manager.list_collections()
    except Exception as e:
        logger.error(f"尝试连接 Milvus 失败: {e}")
        yield event.plain_result(
            f"⚠️ 无法连接到 Milvus 服务: {e}\n请检查 Milvus 配置和服务状态。"
        )
        return

    try:
        # 检查 embedding provider 是否就绪
        if not self.embedding_provider or not self._embedding_provider_ready:
            yield event.plain_result(
                "⚠️ Embedding Provider 尚未就绪。\n"
                "请确保已在 AstrBot 中配置并启用 Embedding Provider。\n"
                "配置完成后请重试此命令。"
            )
            return

        # 获取当前 embedding 维度
        current_dim = None
        try:
            current_dim = getattr(self.embedding_provider, "embedding_dim", None)
            if not current_dim and callable(
                getattr(self.embedding_provider, "get_dim", None)
            ):
                current_dim = self.embedding_provider.get_dim()
        except Exception as e:
            logger.error(f"获取 embedding 维度失败: {e}")
            yield event.plain_result(f"⚠️ 无法获取 Embedding Provider 的维度信息: {e}")
            return

        if not current_dim or not isinstance(current_dim, int) or current_dim <= 0:
            yield event.plain_result(
                f"⚠️ Embedding Provider 返回的维度无效: {current_dim}\n"
                "请检查 Embedding Provider 配置。"
            )
            return

        collection_name = self.collection_name
        needs_migration = False
        old_dim = None

        # 检查集合是否已存在
        if self.milvus_manager.has_collection(collection_name):
            # 检查现有集合的维度
            collection = self.milvus_manager.get_collection(collection_name)
            if collection:
                for field in collection.schema.fields:
                    if field.name == "embedding":  # 向量字段名
                        old_dim = field.params.get("dim")
                        if old_dim != current_dim:
                            needs_migration = True
                            logger.warning(
                                f"检测到维度不匹配: 集合维度={old_dim}, 模型维度={current_dim}"
                            )
                        break

            if needs_migration:
                if force != "--force":
                    yield event.plain_result(
                        f"⚠️ 维度不匹配警告 ⚠️\n\n"
                        f"现有集合 '{collection_name}' 的向量维度为 {old_dim}\n"
                        f"当前 Embedding Provider 的维度为 {current_dim}\n\n"
                        f"需要重新初始化集合以匹配新维度。\n"
                        f"旧数据的文本内容将被保留并使用新维度重新生成向量。\n\n"
                        f"⚠️ 此操作将：\n"
                        f"1. 备份当前集合的文本数据\n"
                        f"2. 删除旧集合\n"
                        f"3. 创建新集合（使用新维度）\n"
                        f"4. 重新生成向量并导入数据\n\n"
                        f"如果确认执行，请运行:\n"
                        f"`/memory init --force`"
                    )
                    return

                # 执行数据迁移
                yield event.plain_result(
                    f"🔄 开始迁移数据...\n从维度 {old_dim} 迁移到 {current_dim}"
                )

                # 检查插件数据目录
                if not self.plugin_data_dir:
                    yield event.plain_result("⚠️ 无法获取插件数据目录，迁移中止")
                    logger.error("plugin_data_dir 未初始化，无法进行备份")
                    return

                # 创建备份目录

                backup_dir = Path(self.plugin_data_dir) / "backups"
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    yield event.plain_result(f"⚠️ 无法创建备份目录: {e}，迁移中止")
                    logger.error(f"创建备份目录失败: {e}")
                    return

                timestamp = int(time_module.time())
                backup_file = (
                    backup_dir
                    / f"memory_backup_{collection_name}_{old_dim}to{current_dim}_{timestamp}.json"
                )

                # 分批导出旧数据
                logger.info(f"开始分批导出集合 '{collection_name}' 的所有数据...")
                yield event.plain_result("📦 正在分批导出所有记忆数据...")

                all_records = []
                batch_size = 16384  # Milvus 单次查询上限
                offset = 0

                try:
                    while True:
                        batch_records = self.milvus_manager.query(
                            collection_name=collection_name,
                            expression=f"{PRIMARY_FIELD_NAME} >= 0",
                            output_fields=[
                                "content",
                                "create_time",
                                "session_id",
                                "personality_id",
                            ],
                            limit=batch_size,
                            offset=offset,
                        )

                        if not batch_records:
                            break

                        all_records.extend(batch_records)
                        offset += len(batch_records)

                        logger.info(f"已导出 {len(all_records)} 条记录...")

                        # 如果本批次少于batch_size，说明已经到达末尾
                        if len(batch_records) < batch_size:
                            break

                    if not all_records:
                        logger.warning("旧集合中没有数据，将创建新集合。")

                except Exception as e:
                    logger.error(f"导出旧数据失败: {e}")
                    yield event.plain_result(f"⚠️ 导出旧数据失败: {e}，迁移中止")
                    return

                record_count = len(all_records)

                # 保存备份到文件 - 备份失败则终止整个操作
                try:
                    backup_data = {
                        "collection_name": collection_name,
                        "old_dimension": old_dim,
                        "new_dimension": current_dim,
                        "timestamp": timestamp,
                        "record_count": record_count,
                        "records": all_records,
                    }
                    with open(backup_file, "w", encoding="utf-8") as f:
                        json.dump(backup_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"已将 {record_count} 条记录备份到: {backup_file}")
                    yield event.plain_result(
                        f"✅ 已导出并备份 {record_count} 条记录\n"
                        f"备份文件: {backup_file.name}"
                    )
                except Exception as e:
                    logger.error(f"保存备份文件失败: {e}")
                    yield event.plain_result(
                        f"⚠️ 保存备份文件失败: {e}\n"
                        f"为保证数据安全，迁移操作已终止。\n"
                        f"请检查磁盘空间和文件权限后重试。"
                    )
                    return

                old_records = all_records

                # 删除旧集合
                logger.info(f"删除旧集合 '{collection_name}'...")
                if not self.milvus_manager.drop_collection(collection_name):
                    yield event.plain_result("⚠️ 删除旧集合失败")
                    return
                yield event.plain_result("✅ 已删除旧集合")

                # 更新 schema 并创建新集合
                logger.info("更新 schema 并创建新集合...")
                self.config["embedding_dim"] = current_dim

                # 重新初始化 schema
                from . import initialization

                initialization.initialize_config_and_schema(self)

                # 创建新集合
                initialization.setup_milvus_collection_and_index(
                    self, skip_if_not_ready=False
                )
                yield event.plain_result(f"✅ 已创建新集合（维度: {current_dim}）")

                # 重新生成向量并导入
                if old_records:
                    yield event.plain_result(
                        f"🔄 正在重新生成 {record_count} 条记录的向量..."
                    )
                    success_count = 0
                    fail_count = 0

                    for i, record in enumerate(old_records):
                        try:
                            content = record.get("content", "")
                            if not content:
                                continue

                            # 生成新向量
                            embedding = await self.embedding_provider.get_embedding(
                                content
                            )
                            if not embedding:
                                fail_count += 1
                                continue

                            # 插入新记录 - 使用类型标注避免 Pylance 错误
                            insert_data: list = [
                                {
                                    "personality_id": record.get("personality_id", ""),
                                    "session_id": record.get("session_id", ""),
                                    "content": content,
                                    "embedding": embedding,
                                    "create_time": record.get(
                                        "create_time", int(datetime.now().timestamp())
                                    ),
                                }
                            ]

                            result = self.milvus_manager.insert(
                                collection_name, insert_data
                            )
                            if result:
                                success_count += 1
                            else:
                                fail_count += 1

                            # 每10条记录报告一次进度
                            if (i + 1) % 10 == 0:
                                yield event.plain_result(
                                    f"进度: {i + 1}/{record_count} "
                                    f"(成功: {success_count}, 失败: {fail_count})"
                                )

                        except Exception as e:
                            logger.error(f"处理记录 {i} 时出错: {e}")
                            fail_count += 1

                    # Flush 确保数据持久化
                    self.milvus_manager.flush([collection_name])

                    yield event.plain_result(
                        f"✅ 数据迁移完成！\n"
                        f"成功: {success_count} 条\n"
                        f"失败: {fail_count} 条\n"
                        f"新维度: {current_dim}"
                    )
                else:
                    yield event.plain_result("✅ 迁移完成（无旧数据）")

            else:
                # 维度匹配，无需迁移
                yield event.plain_result(
                    f"✅ 集合 '{collection_name}' 已存在且维度匹配 ({current_dim})。\n"
                    "无需重新初始化。"
                )
        else:
            # 集合不存在，创建新集合
            yield event.plain_result(f"📝 集合 '{collection_name}' 不存在，正在创建...")

            self.config["embedding_dim"] = current_dim
            from . import initialization

            initialization.initialize_config_and_schema(self)
            initialization.setup_milvus_collection_and_index(
                self, skip_if_not_ready=False
            )

            yield event.plain_result(
                f"✅ 已成功创建集合 '{collection_name}' (维度: {current_dim})\n"
                "记忆系统已就绪！"
            )

    except Exception as e:
        logger.error(f"执行 'memory init' 命令失败: {str(e)}", exc_info=True)
        yield event.plain_result(f"⚠️ 初始化失败: {str(e)}")


async def debug_summary_cmd_impl(self: "Mnemosyne", event: AstrMessageEvent):
    """[实现] 强制触发当前会话的记忆总结（调试模式，含数据库写入）"""
    session_id = event.unified_msg_origin
    if not session_id:
        yield event.plain_result("⚠️ 无法获取 Session ID，无法执行调试。")
        return

    logger.info(f"🔧 [Debug] 开始对 Session: {session_id} 执行强制总结...")
    
    # 1. 获取会话上下文
    if not self.context_manager:
        yield event.plain_result("⚠️ Context Manager 未初始化。")
        return
    
    # 尝试从内存获取历史
    history_list = self.context_manager.get_history(session_id)
    
    # 如果内存为空（例如插件刚启动），尝试从 AstrBot 核心数据库拉取
    if not history_list:
        logger.info(f"🔧 [Debug] 内存历史为空，尝试从 AstrBot 核心拉取...")
        try:
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(session_id)
            if curr_cid:
                conversation = await conv_mgr.get_conversation(session_id, curr_cid)
                if conversation and conversation.history:
                    # AstrBot 存储的 history 是 JSON 字符串或 list of dicts
                    import json
                    if isinstance(conversation.history, str):
                        history_list = json.loads(conversation.history)
                    elif isinstance(conversation.history, list):
                        history_list = conversation.history
                    
                    logger.info(f"🔧 [Debug] 成功从核心数据库拉取到 {len(history_list)} 条记录。")
        except Exception as e:
            logger.error(f"从核心拉取历史失败: {e}")

    if not history_list:
        yield event.plain_result("⚠️ 无法获取到任何历史记录（内存与核心数据库均为空）。请先与机器人进行几轮对话后再试。")
        return
        
    yield event.plain_result(f"🔍 正在读取最近 {len(history_list)} 条历史记录...")

    # 2. 格式化历史 (Input)
    from .tools import format_context_to_string
    # 传递足够大的轮数以包含所有历史
    history_str = format_context_to_string(history_list, len(history_list) * 2)
    
    logger.info("="*20 + " [Debug] History Input " + "="*20)
    logger.info(history_str)
    logger.info("="*60)

    # 3. 获取 Persona ID
    from .memory_operations import _get_persona_id, handle_summary_long_memory
    persona_id = await _get_persona_id(self, event)

    yield event.plain_result("⏳ 正在执行总结流水线 (生成 -> 向量化 -> 存储)...")
    
    try:
        # 直接调用核心业务函数，复用其内部的所有逻辑（包括 embedding 和 milvus insert）
        # 注意：这是一个后台任务函数，这里我们需要 await 它
        await handle_summary_long_memory(self, persona_id, session_id, history_str)
        
        yield event.plain_result(
            "✅ 总结流程执行完毕！\n"
            "请检查控制台日志确认 LLM 输出内容。\n"
            "验证方法：请执行 `/memory list_records` 查看最新的一条记录是否为刚才生成的总结。"
        )
        
    except Exception as e:
        logger.error(f"Debug 总结过程出错: {e}", exc_info=True)
        yield event.plain_result(f"❌ 发生异常: {e}")
