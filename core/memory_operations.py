"""
Mnemosyne 插件核心记忆操作逻辑
包括 RAG 查询、LLM 响应处理、记忆总结与存储。
"""

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pymilvus.exceptions import MilvusException

from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.log import LogManager

from .chatroom_parser import ChatroomContextParser
from .constants import (
    DEFAULT_MILVUS_TIMEOUT,
    DEFAULT_PERSONA_ON_NONE,
    DEFAULT_TOP_K,
    VECTOR_FIELD_NAME,
)
from .security_utils import (
    safe_build_milvus_expression,
    validate_personality_id,
    validate_session_id,
)
from .tools import (
    format_context_to_string,
    remove_mnemosyne_tags,
    remove_system_content,
    remove_system_mnemosyne_tags,
)

# 类型提示，避免循环导入
if TYPE_CHECKING:
    from ..main import Mnemosyne

logger = LogManager.GetLogger(__name__)


async def handle_query_memory(
    plugin: "Mnemosyne", event: AstrMessageEvent, req: ProviderRequest
):
    """
    处理 LLM 请求前的 RAG 检索逻辑。
    检索相关的长期记忆，并将其注入到 ProviderRequest 中。
    """
    # logger = plugin.logger

    # --- 前置检查 ---
    if not await _check_rag_prerequisites(plugin):
        # [Fallback] 即使前置检查失败，也要确保重置当前会话的记忆缓存，防止脏读
        if hasattr(plugin, "set_memory_data"):
            plugin.set_memory_data(event.unified_msg_origin, "")
        return

    try:
        # [Init] 初始化缓存为空，确保默认安全
        if hasattr(plugin, "set_memory_data"):
            plugin.set_memory_data(event.unified_msg_origin, "")

        # --- 获取会话和人格信息 ---
        persona_id = await _get_persona_id(plugin, event)
        # 直接使用 unified_msg_origin 作为 session_id，确保多Bot场景下的记忆隔离
        session_id = event.unified_msg_origin

        # 【新增】触发运行时自动迁移
        if session_id and ":" in session_id:
            # 异步触发迁移，不阻塞查询
            from .migration_utils import migrate_session_data_if_needed

            asyncio.create_task(
                migrate_session_data_if_needed(
                    plugin, session_id, plugin.collection_name
                )
            )

        # M12 修复: 加强 session_id 空值检查，确保类型和内容都有效
        if (
            session_id is None
            or not isinstance(session_id, str)
            or not session_id.strip()
        ):
            logger.error(
                f"无法获取有效的 session_id (值: {session_id}, 类型: {type(session_id).__name__})，跳过记忆查询操作"
            )
            return

        # 检查 context_manager 和 msg_counter 是否可用
        if not plugin.context_manager or not plugin.msg_counter:
            logger.warning("context_manager 或 msg_counter 不可用，跳过记忆查询")
            return

        # 判断是否在历史会话管理器中，如果不在，则进行初始化
        if session_id not in plugin.context_manager.conversations:
            plugin.context_manager.init_conv(session_id, req.contexts, event)

        # 清理记忆标签
        clean_contexts(plugin, req)

        # 添加用户消息
        plugin.context_manager.add_message(session_id, "user", req.prompt)
        # 计数器+1
        plugin.msg_counter.increment_counter(session_id)

        # --- RAG 搜索 ---
        detailed_results = []
        try:
            # 1. 向量化用户查询
            # 使用 AstrBot EmbeddingProvider（异步）
            try:
                # 等待 Embedding Provider 就绪
                if (
                    not plugin.embedding_provider
                    and not plugin._embedding_provider_ready
                ):
                    logger.warning("Embedding Provider 不可用，无法执行 RAG 搜索")
                    return

                # ===== 提取真实用户消息用于 RAG 搜索 =====
                # 自动检测并提取（如果不是特殊格式则返回原值）
                actual_query = ChatroomContextParser.extract_actual_message(req.prompt)

                if actual_query != req.prompt:
                    logger.info(
                        f"检测到群聊上下文格式，已提取真实消息用于 RAG 搜索 "
                        f"(原始: {len(req.prompt)}字符 → 提取: {len(actual_query)}字符)"
                    )

                # 使用 AstrBot EmbeddingProvider 的 embed 方法
                if plugin.embedding_provider:
                    # 使用提取的真实消息进行向量化
                    query_vector = await plugin.embedding_provider.get_embedding(
                        actual_query
                    )
                else:
                    logger.error("Embedding Provider 未正确初始化")
                    return

                if not query_vector:
                    logger.error("无法获取用户查询的 Embedding 向量。")
                    return

            except ConnectionError as e:
                logger.error(f"网络连接错误，无法获取 Embedding: {e}", exc_info=True)
                return
            except ValueError as e:
                logger.error(f"输入参数错误，无法获取 Embedding: {e}", exc_info=True)
                return
            except RuntimeError as e:
                logger.error(f"运行时错误，无法获取 Embedding: {e}", exc_info=True)
                return
            except Exception as e:
                logger.error(f"获取 Embedding 时发生未知错误: {e}", exc_info=True)
                return

            # 2. 执行 Milvus 搜索
            detailed_results = await _perform_milvus_search(
                plugin, query_vector, session_id, persona_id
            )

            # 3. 格式化结果并注入到提示中
            if detailed_results:
                _format_and_inject_memory(plugin, detailed_results, req, event)

        except Exception as e:
            logger.error(f"处理长期记忆 RAG 查询时发生错误: {e}", exc_info=True)
            return

    except Exception as e:
        logger.error(f"处理 LLM 请求前的记忆查询流程失败: {e}", exc_info=True)


async def handle_on_llm_resp(
    plugin: "Mnemosyne", event: AstrMessageEvent, resp: LLMResponse
):
    """
    处理 LLM 响应后的逻辑。更新计数器。
    """
    if resp.role != "assistant":
        logger.warning("LLM 响应不是助手角色，不进行记录。")
        return

    # 检查是否有 context_manager 和 msg_counter
    if not plugin.context_manager or not plugin.msg_counter:
        logger.warning("context_manager 或 msg_counter 不可用，跳过记忆记录")
        return

    try:
        # 直接使用 unified_msg_origin 作为 session_id
        session_id = event.unified_msg_origin
        if not session_id:
            logger.error("无法获取当前 session_id,无法记录 LLM 响应到Mnemosyne。")
            return
        persona_id = await _get_persona_id(plugin, event)

        # 判断是否需要总结
        await _check_and_trigger_summary(
            plugin,
            session_id,
            plugin.context_manager.get_history(session_id),
            persona_id,
        )

        logger.debug(f"返回的内容：{resp.completion_text}")
        plugin.context_manager.add_message(
            session_id, "assistant", resp.completion_text
        )
        plugin.msg_counter.increment_counter(session_id)

    except Exception as e:
        logger.error(f"处理 LLM 响应后的记忆记录失败: {e}", exc_info=True)


# 记忆查询 (RAG) 相关函数
async def _check_rag_prerequisites(plugin: "Mnemosyne") -> bool:
    """
    检查 RAG 查询的前提条件是否满足。

    Args:
        plugin: Mnemosyne 插件实例。

    Returns:
        True 如果前提条件满足，False 否则。
    """
    # logger = plugin.logger
    if not plugin.milvus_manager:
        logger.warning("Milvus 管理器未初始化，无法查询长期记忆。")
        return False
    if not plugin.milvus_manager.is_connected():
        logger.warning("Milvus 服务未连接，无法查询长期记忆。")
        return False
    # 检查 Embedding Provider 是否就绪，支持延迟加载
    if not plugin.embedding_provider and not plugin._embedding_provider_ready:
        logger.warning("Embedding Provider 未初始化，部分功能可能受限。")
        return False
    if not plugin.msg_counter:
        logger.error("消息计数器未初始化，将无法实现记忆总结")
        return False
    return True


async def _get_persona_id(plugin: "Mnemosyne", event: AstrMessageEvent) -> str | None:
    """
    获取当前会话的人格 ID。

    Args:
        plugin: Mnemosyne 插件实例。
        event: 消息事件。

    Returns:
        人格 ID 字符串，如果没有人格或发生错误则为 None。
    """
    # logger = plugin.logger
    # 获取 conversation_id 用于获取人格配置
    conversation_id = (
        await plugin.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
    )
    conversation = await plugin.context.conversation_manager.get_conversation(
        event.unified_msg_origin, str(conversation_id)
    )
    persona_id = conversation.persona_id if conversation else None

    if not persona_id or persona_id == "[%None]":
        # 不使用默认人格，避免记忆错乱
        # 当会话没有配置人格时，使用占位符或None，而不是回退到默认人格
        logger.warning(
            f"当前会话 (ID: {event.unified_msg_origin}) 未配置人格，将使用占位符 '{DEFAULT_PERSONA_ON_NONE}' 进行记忆操作（如果启用人格过滤）。"
        )
        if plugin.config.get("use_personality_filtering", False):
            persona_id = DEFAULT_PERSONA_ON_NONE
        else:
            persona_id = None
    return persona_id


async def _check_and_trigger_summary(
    plugin: "Mnemosyne",
    session_id: str,
    context: list[dict],
    persona_id: str | None,
):
    """
    检查是否满足总结条件并触发总结任务。

    Args:
        plugin: Mnemosyne 插件实例。
        session_id: 会话 ID。
        context: 请求上下文列表。
        persona_id: 人格 ID.
    """
    # M24 修复: 添加 msg_counter 的类型检查
    # num_pairs 是对话轮数，msg_counter 计数的是消息条数（一问一答=2条消息）
    # 所以需要用 num_pairs * 2 来比较
    num_pairs = plugin.config.get("num_pairs", 5)
    if (
        plugin.msg_counter
        and plugin.msg_counter.adjust_counter_if_necessary(session_id, context)
        and plugin.msg_counter.get_counter(session_id) >= num_pairs * 2
    ):
        logger.info(f"对话已达到 {num_pairs} 轮，开始总结历史对话...")
        # M24 修复: 添加类型忽略，context 来自运行时的上下文
        history_contents = format_context_to_string(
            context,  # type: ignore
            num_pairs * 2,  # 传递消息条数而不是轮数
        )

        # M19 修复: 为后台任务添加异常处理回调
        task = asyncio.create_task(
            handle_summary_long_memory(plugin, persona_id, session_id, history_contents)
        )

        # [Visual Log] 显式日志：对话轮数触发总结
        logger.info("="*40)
        logger.info(f"⏰ [Mnemosyne] 触发记忆总结 (机制: 对话轮数)")
        logger.info(f"📊 当前计数: {plugin.msg_counter.get_counter(session_id)}")
        logger.info(f"🎯 触发阈值: {num_pairs * 2} (即 {num_pairs} 轮)")
        logger.info(f"🆔 Session: {session_id}")
        logger.info("="*40)

        def task_done_callback(t: asyncio.Task):
            """后台任务完成时的回调，用于捕获未处理的异常"""
            try:
                # 获取任务结果，如果有异常会在这里抛出
                t.result()
            except asyncio.CancelledError:
                logger.info(f"总结任务被取消 (session: {session_id})")
            except Exception as e:
                logger.error(
                    f"后台总结任务执行失败 (session: {session_id}): {e}", exc_info=True
                )

        task.add_done_callback(task_done_callback)
        logger.info("总结历史对话任务已提交到后台执行。")
        # M24 修复: 添加类型检查
        if plugin.msg_counter:
            plugin.msg_counter.reset_counter(session_id)


async def _perform_milvus_search(
    plugin: "Mnemosyne",
    query_vector: list[float],
    session_id: str | None,
    persona_id: str | None,
) -> list[dict] | None:
    """
    执行 Milvus 向量搜索。

    Args:
        plugin: Mnemosyne 插件实例。
        query_vector: 查询向量。
        session_id: 会话 ID。
        persona_id: 人格 ID。

    Returns:
        Milvus 搜索结果列表，如果没有找到或出错则为 None。
    """
    # logger = plugin.logger
    # 防止没有过滤条件引发的潜在错误
    filters = ["memory_id > 0"]

    # 检查是否启用了会话过滤
    use_session_filtering = plugin.config.get("use_session_filtering", True)
    
    if use_session_filtering:
        if session_id:
            # 安全检查：验证 session_id 格式
            if not validate_session_id(session_id):
                logger.error(f"session_id 格式验证失败: {session_id}")
                return None

            # 使用安全的表达式构建方法
            try:
                session_filter = safe_build_milvus_expression(
                    "session_id", session_id, "=="
                )
                filters.append(session_filter)
                logger.debug(f"已启用会话过滤，将使用会话 '{session_id}' 过滤记忆。")
            except ValueError as e:
                logger.error(f"构建 session_id 过滤表达式失败: {e}")
                return None
        else:
            logger.warning("无法获取当前 session_id，将不按 session 过滤记忆！")
    else:
        logger.info("会话过滤已禁用，将在所有会话中搜索记忆。")

    use_personality_filtering = plugin.config.get("use_personality_filtering", False)
    effective_persona_id_for_filter = persona_id
    if use_personality_filtering and effective_persona_id_for_filter:
        # 安全检查：验证 personality_id 格式
        if not validate_personality_id(effective_persona_id_for_filter):
            logger.warning(
                f"personality_id 格式验证失败: {effective_persona_id_for_filter}，跳过人格过滤"
            )
        else:
            # 使用安全的表达式构建方法
            try:
                persona_filter = safe_build_milvus_expression(
                    "personality_id", effective_persona_id_for_filter, "=="
                )
                filters.append(persona_filter)
                logger.debug(
                    f"将使用人格 '{effective_persona_id_for_filter}' 过滤记忆。"
                )
            except ValueError as e:
                logger.error(f"构建 personality_id 过滤表达式失败: {e}")
    elif use_personality_filtering:
        logger.debug("启用了人格过滤，但当前无有效人格 ID，不按人格过滤。")

    search_expression = " and ".join(filters) if filters else ""
    collection_name = plugin.collection_name
    top_k = plugin.config.get("top_k", DEFAULT_TOP_K)
    timeout_seconds = plugin.config.get("milvus_search_timeout", DEFAULT_MILVUS_TIMEOUT)

    logger.info(
        f"开始在集合 '{collection_name}' 中搜索相关记忆 (TopK: {top_k}, Filter: '{search_expression or '无'}')"
    )

    # M24 修复: 添加 milvus_manager 的类型检查
    if not plugin.milvus_manager:
        logger.error("Milvus 管理器不可用")
        return None

    try:
        search_results = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: plugin.milvus_manager.search(  # type: ignore
                    collection_name=collection_name,
                    query_vectors=[query_vector],
                    vector_field=VECTOR_FIELD_NAME,
                    search_params=plugin.search_params,
                    limit=top_k,
                    expression=search_expression,
                    output_fields=plugin.output_fields_for_query,
                ),
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(f"Milvus 搜索超时 ({timeout_seconds} 秒)，操作已取消。")
        return None
    except MilvusException as me:
        logger.error(f"Milvus 搜索操作失败: {me}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"执行 Milvus 搜索时发生未知错误: {e}", exc_info=True)
        return None

    if not search_results or not search_results[0]:
        logger.info("向量搜索未找到相关记忆。")
        return None
    else:
        # 从 search_results 中获取 Hits 对象
        hits = search_results[0]
        # 调用新的辅助函数来处理 Hits 对象并提取详细结果
        detailed_results = _process_milvus_hits(hits)
        return detailed_results


def _process_milvus_hits(hits) -> list[dict[str, Any]]:
    """
    处理 Milvus SearchResults 中的 Hits 对象，使用基于索引的遍历方式
    提取有效的记忆实体数据。

    Args:
        hits: 从 Milvus 搜索结果 search_results[0] 中获取的 Hits 对象。

    Returns:
        一个包含提取到的记忆实体字典的列表。如果没有任何有效实体被提取，
        则返回空列表 []。
    """
    detailed_results: list[dict[str, Any]] = []  # 初始化结果列表，指定类型

    # 使用索引遍历 hits 对象，以绕过 SequenceIterator 的迭代问题
    if hits:  # 确保 hits 对象不是空的或 None
        try:
            num_hits = len(hits)  # 获取命中数量
            logger.debug(f"Milvus 返回了 {num_hits} 条原始命中结果。")

            # 使用索引进行遍历
            for i in range(num_hits):
                try:
                    hit = hits[i]  # 通过索引获取单个 Hit 对象

                    # 检查 hit 对象及其 entity 属性是否存在且有效
                    # 使用 hasattr 更健壮，避免在 entity 属性不存在时报错
                    if hit and hasattr(hit, "entity") and hit.entity:
                        # 提取 entity 数据，使用 .get() 避免 KeyError
                        # 假设 entity.to_dict() 返回的字典中有 "entity" 键
                        entity_data = hit.entity.to_dict().get("entity")
                        # 如果成功提取到数据，则添加到结果列表
                        if entity_data:
                            detailed_results.append(entity_data)
                        else:
                            # 如果 entity 存在但提取的数据为空，可能是数据结构问题
                            logger.warning(
                                f"命中结果索引 {i} 处的 entity 数据为空或无效，已跳过。"
                            )
                    else:
                        # 如果 hit 或 entity 无效，则跳过
                        logger.debug(f"命中结果索引 {i} 处对象或 entity 无效，已跳过。")

                except Exception as e:
                    # 处理访问或处理单个 hit 时可能出现的错误
                    logger.error(
                        f"处理索引 {i} 处的命中结果时发生错误: {e}", exc_info=True
                    )
                    # 发生错误时继续处理下一个 hit，不中断整个流程

        except Exception as e:
            # 处理获取长度或设置循环时可能出现的更严重的错误
            # 如果在这里发生错误，detailed_results 可能不完整或为空
            logger.error(f"执行基于索引的命中结果处理时发生错误: {e}", exc_info=True)

    # 记录成功处理并提取记忆的记录数
    logger.debug(f"成功处理并提取记忆的记录数: {len(detailed_results)} 条。")

    return detailed_results


# LLM 响应处理相关函数
def _format_and_inject_memory(
    plugin: "Mnemosyne", detailed_results: list[dict], req: ProviderRequest, event: AstrMessageEvent
):
    """
    格式化搜索结果并注入到 ProviderRequest 中。

    Args:
        plugin: Mnemosyne 插件实例。
        detailed_results: 详细的搜索结果列表。
        req: ProviderRequest 对象。
        event: 消息事件对象 (用于挂载 state)。
    """
    # logger = plugin.logger
    if not detailed_results:
        logger.info("未找到或获取到相关的长期记忆，不进行补充。")
        return

    long_memory_prefix = plugin.config.get(
        "long_memory_prefix", "<Mnemosyne> 长期记忆片段："
    )
    long_memory_suffix = plugin.config.get("long_memory_suffix", "</Mnemosyne>")
    long_memory = f"{long_memory_prefix}\n"

    for result in detailed_results:
        content = result.get("content", "内容缺失")
        ts = result.get("create_time")
        try:
            time_str = (
                datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                if ts
                else "未知时间"
            )
        except (TypeError, ValueError):
            time_str = f"时间戳: {ts}" if ts else "未知时间"

        memory_entry_format = plugin.config.get(
            "memory_entry_format", "- [{time}] {content}"
        )
        long_memory += memory_entry_format.format(time=time_str, content=content) + "\n"

    long_memory += long_memory_suffix

    # [Visual Log] 结构化展示记忆检索结果
    logger.info("="*50)
    logger.info(f"🧠 [Mnemosyne] 记忆检索成功")
    logger.info(f"📊 命中数量: {len(detailed_results)} 条")
    logger.info(f"🔗 注入模式: {plugin.config.get('memory_injection_method', 'user_prompt')}")
    logger.info("-" * 20 + " 记忆片段预览 " + "-" * 20)
    
    preview_len = 100
    preview_content = long_memory.replace("\n", " ")
    if len(preview_content) > preview_len:
        preview_content = preview_content[:preview_len] + "..."
    logger.info(f"📝 {preview_content}")
    logger.info("="*50)

    # [Fix] 使用插件共享缓存替代 event.state，避免 AttributeError
    # 直接使用 unified_msg_origin 作为 session_id
    if hasattr(plugin, "set_memory_data"):
        plugin.set_memory_data(event.unified_msg_origin, long_memory)
    else:
        logger.error("Mnemosyne 实例缺少 set_memory_data 方法，记忆无法传递！")

    injection_method = plugin.config.get("memory_injection_method", "user_prompt")

    # 清理插入的长期记忆内容
    clean_contexts(plugin, req)
    
    # [新增] state_only 模式：只挂载变量，不修改 prompt，避免重复
    if injection_method == "state_only":
        logger.info("检测到注入模式为 'state_only'，已跳过 Prompt 自动注入。")
        return

    if injection_method == "user_prompt":
        req.prompt = long_memory + "\n" + req.prompt

    elif injection_method == "system_prompt":
        req.system_prompt += long_memory

    elif injection_method == "insert_system_prompt":
        req.contexts.append({"role": "system", "content": long_memory})

    else:
        logger.warning(
            f"未知的记忆注入方法 '{injection_method}'，将默认追加到用户 prompt。"
        )
        req.prompt = long_memory + "\n" + req.prompt


# 删除补充的长期记忆函数
def clean_contexts(plugin: "Mnemosyne", req: ProviderRequest):
    """
    删除长期记忆中的标签
    """
    injection_method = plugin.config.get("memory_injection_method", "user_prompt")
    contexts_memory_len = plugin.config.get("contexts_memory_len", 0)
    if injection_method == "user_prompt":
        req.contexts = remove_mnemosyne_tags(req.contexts, contexts_memory_len)
    elif injection_method == "system_prompt":
        req.system_prompt = remove_system_mnemosyne_tags(
            req.system_prompt, contexts_memory_len
        )
    elif injection_method == "insert_system_prompt":
        req.contexts = remove_system_content(req.contexts, contexts_memory_len)
    return


# 记忆总结相关函数
async def _check_summary_prerequisites(plugin: "Mnemosyne", memory_text: str) -> bool:
    """
    检查记忆总结的前提条件是否满足。

    Args:
        plugin: Mnemosyne 插件实例。
        memory_text: 记忆文本。

    Returns:
        True 如果前提条件满足，False 否则。
    """
    # logger = plugin.logger
    if not plugin.milvus_manager or not plugin.milvus_manager.is_connected():
        logger.error("Milvus 服务不可用，无法存储总结后的长期记忆。")
        return False
    if not plugin.embedding_provider:
        logger.error("Embedding Provider 不可用，无法向量化总结记忆。")
        return False
    if not memory_text or not memory_text.strip():
        logger.warning("尝试总结空的或仅包含空白的记忆文本，跳过。")
        return False
    return True


async def _get_summary_llm_response(
    plugin: "Mnemosyne", memory_text: str
) -> LLMResponse | None:
    """
    请求 LLM 进行记忆总结。

    Args:
        plugin: Mnemosyne 插件实例。
        memory_text: 需要总结的记忆文本。

    Returns:
        LLMResponse 对象，如果请求失败则为 None。
    """
    # logger = plugin.logger
    llm_provider = plugin.provider
    # TODO: 优化LLM Provider获取逻辑，确保在plugin.provider不可用时能正确回退到当前使用的Provider
    try:
        if not llm_provider:
            # 如果plugin.provider不正确，在这时候，使用当前使用的LLM服务商，避免错误
            llm_provider = plugin.context.get_using_provider()
            if not llm_provider:
                logger.error("无法获取用于总结记忆的 LLM Provider。")
                return None
    except Exception as e:
        logger.error(f"获取 LLM Provider 时出错: {e}", exc_info=True)
        return None

    long_memory_prompt = plugin.config.get(
        "long_memory_prompt",
        "请将以下多轮对话历史总结为一段简洁、客观、包含关键信息的长期记忆条目:",
    )
    summary_llm_config = plugin.config.get("summary_llm_config", {})

    logger.debug(
        f"请求 LLM 总结短期记忆，提示: '{long_memory_prompt[:50]}...', 内容长度: {len(memory_text)}"
    )

    try:
        # M24 修复: 添加 text_chat 方法的类型忽略
        llm_response = await llm_provider.text_chat(  # type: ignore
            prompt=memory_text,
            contexts=[{"role": "system", "content": long_memory_prompt}],
            **summary_llm_config,
        )
        logger.debug(f"LLM 总结响应原始数据: {llm_response}")
        return llm_response
    except Exception as e:
        logger.error(f"LLM 总结请求失败: {e}", exc_info=True)
        return None


def _extract_summary_text(plugin: "Mnemosyne", llm_response: LLMResponse) -> str | None:
    """
    从 LLM 响应中提取总结文本并进行校验。

    Args:
        plugin: Mnemosyne 插件实例。
        llm_response: LLMResponse 对象。

    Returns:
        总结文本字符串，如果提取失败则为 None。
    """
    # logger = plugin.logger
    completion_text = None
    if isinstance(llm_response, LLMResponse):
        completion_text = llm_response.completion_text
        # role = llm_response.role
    elif isinstance(llm_response, dict):
        completion_text = llm_response.get("completion_text")
        # role = llm_response.get("role")
    else:
        logger.error(f"LLM 总结返回了未知类型的数据: {type(llm_response)}")
        return None

    if not completion_text or not completion_text.strip():
        logger.error(f"LLM 总结响应无效或内容为空。原始响应: {llm_response}")
        return None

    summary_text = completion_text.strip()
    logger.info(f"LLM 成功生成记忆总结，长度: {len(summary_text)}")
    return summary_text


async def _store_summary_to_milvus(
    plugin: "Mnemosyne",
    persona_id: str | None,
    session_id: str,
    summary_text: str,
    embedding_vector: list[float],
):
    """
    将总结文本和向量存储到 Milvus 中。

    Args:
        plugin: Mnemosyne 插件实例。
        persona_id: 人格 ID。
        session_id: 会话 ID。
        summary_text: 总结文本。
        embedding_vector: 总结文本的 Embedding 向量。
    """
    # logger = plugin.logger
    collection_name = plugin.collection_name
    current_timestamp = int(time.time())

    effective_persona_id = (
        persona_id
        if persona_id
        else plugin.config.get("default_persona_id_on_none", DEFAULT_PERSONA_ON_NONE)
    )

    data_to_insert = [
        {
            "personality_id": effective_persona_id,
            "session_id": session_id,
            "content": summary_text,
            VECTOR_FIELD_NAME: embedding_vector,
            "create_time": current_timestamp,
        }
    ]

    logger.info(
        f"准备向集合 '{collection_name}' 插入 1 条总结记忆 (Persona: {effective_persona_id}, Session: {session_id[:8]}...)"
    )
    # mutation_result = plugin.milvus_manager.insert(
    #     collection_name=collection_name,
    #     data=data_to_insert,
    # )
    # --- 修改 insert 调用 ---
    loop = asyncio.get_event_loop()
    mutation_result = None

    # M24 修复: 添加 milvus_manager 的类型检查
    if not plugin.milvus_manager:
        logger.error("Milvus 管理器不可用")
        return

    try:
        # M24 修复: 定义插入函数避免类型检查问题
        def _insert_data():
            return plugin.milvus_manager.insert(  # type: ignore
                collection_name=collection_name,
                data=data_to_insert,  # type: ignore
            )

        mutation_result = await loop.run_in_executor(
            None,  # 使用默认线程池
            _insert_data,
        )
    except (MilvusException, ConnectionError, ValueError) as e:
        logger.error(f"向 Milvus 插入总结记忆时出错: {e}", exc_info=True)
    finally:
        # 确保资源清理和错误日志记录
        if mutation_result is None:
            logger.error(
                f"Milvus 插入操作失败，未返回结果。集合: {collection_name}, 数据: {summary_text[:100]}..."
            )
        else:
            logger.debug("Milvus 插入操作完成，正在进行资源清理。")

    if mutation_result and mutation_result.insert_count > 0:
        inserted_ids = mutation_result.primary_keys
        logger.info(f"成功插入总结记忆到 Milvus。插入 ID: {inserted_ids}")

        try:
            logger.debug(
                f"正在刷新 (Flush) 集合 '{collection_name}' 以确保记忆立即可用..."
            )

            # plugin.milvus_manager.flush([collection_name])
            # M24 修复: 定义刷新函数避免类型检查问题
            def _flush_collection():
                return plugin.milvus_manager.flush([collection_name])  # type: ignore

            await loop.run_in_executor(
                None,  # 使用默认线程池
                _flush_collection,
            )
            logger.debug(f"集合 '{collection_name}' 刷新完成。")

        except Exception as flush_err:
            logger.error(
                f"刷新集合 '{collection_name}' 时出错: {flush_err}",
                exc_info=True,
            )
    else:
        logger.error(
            f"插入总结记忆到 Milvus 失败。MutationResult: {mutation_result}. LLM 回复: {summary_text[:100]}..."
        )


async def handle_summary_long_memory(
    plugin: "Mnemosyne", persona_id: str | None, session_id: str, memory_text: str
):
    """
    使用 LLM 总结短期对话历史形成长期记忆，并将其向量化后存入 Milvus。
    这是一个后台任务。
    """
    # logger = plugin.logger

    # [Optimization] Acquire semaphore from plugin instance to limit concurrency
    # Fallback to a temporary semaphore if not initialized (though it should be)
    semaphore = getattr(plugin, "summary_semaphore", asyncio.Semaphore(3))
    
    async with semaphore:
        # --- 前置检查 ---
        if not await _check_summary_prerequisites(plugin, memory_text):
            return

        try:
            # 1. 请求 LLM 进行总结
            llm_response = await _get_summary_llm_response(plugin, memory_text)
            if not llm_response:
                return

            # 2. 提取总结文本
            summary_text = _extract_summary_text(plugin, llm_response)
            if not summary_text:
                return

            # [Visual Log] 打印 LLM 生成的总结内容 (Debug 可见)
            logger.info("="*20 + " [Mnemosyne] Generated Summary " + "="*20)
            logger.info(f"{summary_text}")
            logger.info("="*60)

            # 3. 获取总结文本的 Embedding
            # 使用 AstrBot EmbeddingProvider（异步）
            try:
                if not plugin.embedding_provider:
                    logger.error("Embedding Provider 不可用，无法获取总结的 Embedding")
                    return

                # 使用 AstrBot EmbeddingProvider 的 get_embedding 方法
                embedding_vector = await plugin.embedding_provider.get_embedding(
                    summary_text
                )

                if not embedding_vector:
                    logger.error(f"无法获取总结文本的 Embedding: '{summary_text[:100]}...'")
                    return

            except (ConnectionError, ValueError, RuntimeError) as e:
                logger.error(
                    f"获取总结文本 Embedding 时出错: '{summary_text[:100]}...' - {e}",
                    exc_info=True,
                )
                return
            except Exception as e:
                logger.error(
                    f"获取总结文本 Embedding 时发生未知错误: '{summary_text[:100]}...' - {e}",
                    exc_info=True,
                )
                return

            # 4. 存储到 Milvus
            await _store_summary_to_milvus(
                plugin, persona_id, session_id, summary_text, embedding_vector
            )
            return
        except Exception as e:
            logger.error(f"在总结或存储长期记忆的过程中发生严重错误: {e}", exc_info=True)


# 计时器
async def _periodic_summarization_check(plugin: "Mnemosyne"):
    """
    [后台任务] 定期检查并触发超时的会话总结

    S0 优化: 添加异常恢复机制，防止任务崩溃
    """
    logger.info(
        f"启动定期总结检查任务，检查间隔: {plugin.summary_check_interval}秒, 总结时间阈值: {plugin.summary_time_threshold}秒。"
    )

    # S0 优化: 异常恢复计数器
    consecutive_errors = 0
    max_consecutive_errors = 5

    while True:
        try:
            await asyncio.sleep(plugin.summary_check_interval)  # <--- 等待指定间隔

            if not plugin.context_manager or plugin.summary_time_threshold == float(
                "inf"
            ):
                # 如果上下文管理器未初始化或阈值无效，则跳过本次检查
                continue

            current_time = time.time()
            session_ids_to_check = list(plugin.context_manager.conversations.keys())

            # logger.debug(f"开始检查 {len(session_ids_to_check)} 个会话的总结超时...")

            for session_id in session_ids_to_check:
                try:
                    session_context = plugin.context_manager.get_session_context(
                        session_id
                    )
                    if not session_context:  # 会话可能在检查期间被移除
                        continue
                    # M24 修复: 添加 msg_counter 的类型检查
                    if (
                        not plugin.msg_counter
                        or plugin.msg_counter.get_counter(session_id) <= 0
                    ):
                        logger.debug(f"会话 {session_id} 没有新消息，跳过检查。")
                        continue

                    last_summary_time = session_context["last_summary_time"]

                    if current_time - last_summary_time > plugin.summary_time_threshold:
                        # logger.debug(f"current_time {current_time} - last_summary_time {last_summary_time} : {current_time - last_summary_time}")
                        
                        # [Visual Log] 显式日志：时间阈值触发总结
                        time_diff = int(current_time - last_summary_time)
                        logger.info("="*40)
                        logger.info(f"⏰ [Mnemosyne] 触发记忆总结 (机制: 定时任务)")
                        logger.info(f"⌛ 距上次总结: {time_diff} 秒")
                        logger.info(f"🎯 超时阈值: {plugin.summary_time_threshold} 秒")
                        logger.info(f"🆔 Session: {session_id}")
                        logger.info("="*40)
                        
                        # 运行总结
                        logger.info("开始总结历史对话...")
                        # M24 修复: 添加 msg_counter 的类型检查和类型忽略
                        counter = (
                            plugin.msg_counter.get_counter(session_id)
                            if plugin.msg_counter
                            else 0
                        )
                        history_contents = format_context_to_string(
                            session_context["history"],
                            counter,  # type: ignore
                        )
                        persona_id = await _get_persona_id(
                            plugin, session_context["event"]
                        )
                        asyncio.create_task(
                            handle_summary_long_memory(
                                plugin, persona_id, session_id, history_contents
                            )
                        )
                        logger.info("总结历史对话任务已提交到后台执行。")

                        # M24 修复: 添加 msg_counter 的类型检查
                        if plugin.msg_counter:
                            plugin.msg_counter.reset_counter(session_id)
                        plugin.context_manager.update_summary_time(session_id)

                except KeyError:
                    # 会话在获取 keys 后、处理前被删除，是正常情况
                    logger.debug(f"检查会话 {session_id} 时，会话已被移除。")
                except Exception as e:
                    logger.error(
                        f"检查或总结会话 {session_id} 时发生错误: {e}", exc_info=True
                    )

            # S0 优化: 成功完成一次循环，重置错误计数器
            consecutive_errors = 0

        except asyncio.CancelledError:
            logger.info("定期总结检查任务被取消。")
            break  # 退出循环
        except Exception as e:
            # S0 优化: 增强的异常处理和恢复机制
            consecutive_errors += 1
            logger.error(
                f"定期总结检查任务主循环发生错误 (连续错误次数: {consecutive_errors}/{max_consecutive_errors}): {e}",
                exc_info=True,
            )

            # 指数退避策略：等待时间随错误次数增加
            backoff_time = min(
                plugin.summary_check_interval * (2 ** (consecutive_errors - 1)), 300
            )
            logger.warning(f"将在 {backoff_time} 秒后重试后台总结任务...")

            try:
                await asyncio.sleep(backoff_time)
            except asyncio.CancelledError:
                logger.info("等待重试期间任务被取消。")
                break

            # 如果连续错误次数过多，记录严重警告但继续尝试
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(
                    f"后台总结任务已连续失败 {consecutive_errors} 次，系统将继续尝试但可能存在严重问题，请检查日志并考虑重启插件。"
                )
                # 重置计数器以避免无限增长
                consecutive_errors = max_consecutive_errors - 1
