"""
Mnemosyne 插件工具函数
"""

import functools
import re
from typing import Any
from urllib.parse import urlparse

from astrbot.api.event import AstrMessageEvent
from astrbot.core.log import LogManager

logger = LogManager.GetLogger(__name__)


def parse_address(address: str):
    """
    解析地址，提取出主机名和端口号。
    如果地址没有协议前缀，则默认添加 "http://"
    """
    if not (address.startswith("http://") or address.startswith("https://")):
        address = "http://" + address
    parsed = urlparse(address)
    host = parsed.hostname
    port = (
        parsed.port if parsed.port is not None else 19530
    )  # 如果未指定端口，默认使用19530
    return host, port


def content_to_str(func):
    """
    实现一个装饰器，将输入的内容全部转化为字符串
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        str_args = [str(arg) for arg in args]
        str_kwargs = {k: str(v) for k, v in kwargs.items()}
        logger.debug(
            f"Function '{func.__name__}' called with arguments: args={str_args}, kwargs={str_kwargs}"
        )
        return func(*str_args, **str_kwargs)

    return wrapper


def remove_mnemosyne_tags(
    contents: list[dict[str, Any]], contexts_memory_len: int = 0
) -> list[dict[str, Any]]:
    """
    使用正则表达式去除LLM上下文中的<mnemosyne> </mnemosyne>标签对。
    - contexts_memory_len > 0: 保留最新的N个标签对。
    - contexts_memory_len == 0: 移除所有标签对。
    - contexts_memory_len < 0: 保留所有标签对，不作任何删除。
    """
    if contexts_memory_len < 0:
        return contents

    compiled_regex = re.compile(r"<Mnemosyne>.*?</Mnemosyne>", re.DOTALL)
    cleaned_contents: list[dict[str, Any]] = []

    if contexts_memory_len == 0:
        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == "user":
                original_text = content_item.get("content", "")
                # 关键修复：多模态内容（list/dict 等）不能强制转换为字符串。
                # 只有在 content 为 str 时才需要清理标签。
                if isinstance(original_text, str):
                    cleaned_text = compiled_regex.sub("", original_text)
                    cleaned_contents.append({"role": "user", "content": cleaned_text})
                else:
                    cleaned_contents.append(content_item)
            else:
                cleaned_contents.append(content_item)
    else:  # contexts_memory_len > 0
        all_mnemosyne_blocks: list[str] = []
        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == "user":
                original_text = content_item.get("content", "")
                if isinstance(original_text, str):
                    found_blocks = compiled_regex.findall(original_text)
                    all_mnemosyne_blocks.extend(found_blocks)

        blocks_to_keep: set[str] = set(all_mnemosyne_blocks[-contexts_memory_len:])

        def replace_logic(match: re.Match) -> str:
            block = match.group(0)
            return block if block in blocks_to_keep else ""

        for content_item in contents:
            if isinstance(content_item, dict) and content_item.get("role") == "user":
                original_text = content_item.get("content", "")

                # M14 修复: 改进逻辑流程，确保正确处理各种情况
                # 使用 elif 形成互斥逻辑，避免重复处理
                if isinstance(original_text, list):
                    # 1. 如果内容是列表（多模态消息），直接保留原样
                    cleaned_contents.append({"role": "user", "content": original_text})
                elif isinstance(original_text, str):
                    # 2. 如果内容是字符串，检查是否需要清理标签
                    if compiled_regex.search(original_text):
                        # 内容包含标签，进行清理
                        cleaned_text = compiled_regex.sub(replace_logic, original_text)
                        cleaned_contents.append(
                            {"role": "user", "content": cleaned_text}
                        )
                    else:
                        # 内容不包含标签，直接保留
                        cleaned_contents.append(content_item)
                else:
                    # 3. 其他类型（不应该出现），记录警告并保留原始内容
                    logger.warning(
                        f"遇到意外的 content 类型: {type(original_text).__name__}，将保留原始内容"
                    )
                    cleaned_contents.append(content_item)
            else:
                # 非 user 角色的消息，直接保留
                cleaned_contents.append(content_item)

    return cleaned_contents


def remove_system_mnemosyne_tags(text: str, contexts_memory_len: int = 0) -> str:
    """
    使用正则表达式去除LLM上下文系统提示中的<Mnemosyne> </Mnemosyne>标签对。
    如果 contexts_memory_len > 0，则仅保留最后 contexts_memory_len 个标签对。
    """
    if not isinstance(text, str):
        return text  # 如果输入不是字符串，直接返回

    if contexts_memory_len < 0:
        return text

    compiled_regex = re.compile(r"<Mnemosyne>.*?</Mnemosyne>", re.DOTALL)

    if contexts_memory_len == 0:
        cleaned_text = compiled_regex.sub("", text)
    else:
        all_mnemosyne_blocks: list[str] = compiled_regex.findall(text)
        blocks_to_keep: set[str] = set(all_mnemosyne_blocks[-contexts_memory_len:])

        def replace_logic(match: re.Match) -> str:
            block = match.group(0)
            return block if block in blocks_to_keep else ""

        if compiled_regex.search(text):
            cleaned_text = compiled_regex.sub(replace_logic, text)
        else:
            cleaned_text = text

    return cleaned_text


def remove_system_content(
    contents: list[dict[str, str]], contexts_memory_len: int = 0
) -> list[dict[str, str]]:
    """
    从LLM上下文中移除较旧的系统提示 ('role'='system' 的消息)，
    保留指定数量的最新的 system 消息，并维持整体消息顺序。
    """
    if not isinstance(contents, list):
        return []
    if contexts_memory_len < 0:
        return contents

    system_message_indices = [
        i
        for i, msg in enumerate(contents)
        if isinstance(msg, dict) and msg.get("role") == "system"
    ]
    indices_to_remove: set[int] = set()
    num_system_messages = len(system_message_indices)

    if num_system_messages > contexts_memory_len:
        num_to_remove = num_system_messages - contexts_memory_len
        indices_to_remove = set(system_message_indices[:num_to_remove])

    cleaned_contents = [
        msg for i, msg in enumerate(contents) if i not in indices_to_remove
    ]

    return cleaned_contents


def format_context_to_string(
    context_history: list[dict[str, Any] | str], length: int = 10
) -> str:
    """
    从上下文历史记录中提取最后 'length' 条用户和AI的对话消息，
    并将它们的内容转换为结构化的字符串，明确标注角色和时间戳（如果可用）。
    """
    if length <= 0 or not context_history:
        return ""

    def _truncate_text(text: str, max_chars: int = 2000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "...(truncated)"

    def _content_to_safe_text(content: Any) -> str:
        """将 AstrBot/OpenAI 风格上下文内容安全转为文本。"""
        # 1) 纯文本
        if isinstance(content, str):
            if content.startswith("base64://") or content.startswith("data:image"):
                return "[图片]"
            return _truncate_text(content)

        # 2) OpenAI 多模态 content blocks
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                if item_type == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(_truncate_text(text))
                    continue

                if item_type == "image_url" or "image_url" in item:
                    parts.append("[图片]")
                    continue

                if item_type == "audio_url" or "audio_url" in item:
                    parts.append("[音频]")
                    continue

                if item_type == "think":
                    continue

                if isinstance(item_type, str) and item_type:
                    parts.append(f"[{item_type}]")

            merged = " ".join(p for p in parts if p)
            return merged or ""

        # 3) 其他结构：避免展开潜在大对象
        if isinstance(content, dict):
            if "image_url" in content or "audio_url" in content:
                return "[图片]" if "image_url" in content else "[音频]"
            text = content.get("text")
            if isinstance(text, str):
                return _truncate_text(text)
            return ""

        return ""

    selected_contents: list[str] = []
    count = 0

    # 倒序遍历，从最新的消息开始提取
    for message in reversed(context_history):
        if count >= length:
            break

        role = "Unknown"
        content = ""
        timestamp_str = ""

        if isinstance(message, dict):
            # 1. 尝试获取发送者名称 (Multi-User Attribution)
            sender_name = message.get("name")
            
            raw_role = message.get("role", "").lower()
            # 明确映射角色名称，避免混淆
            if raw_role in ["user", "human"]:
                # 如果有具体名字，优先使用名字
                role = sender_name if sender_name else "User"
            elif raw_role in ["assistant", "model", "ai"]:
                role = "Rosa"
            elif raw_role == "system":
                continue # 跳过系统提示，专注于对话历史
            else:
                role = raw_role.capitalize() if raw_role else "Unknown"

            content_obj = message.get("content", "")
            content = _content_to_safe_text(content_obj)
            
            # 尝试获取时间戳
            ts = message.get("timestamp") or message.get("created_at") or message.get("time")
            if ts:
                # 简单格式化，假设是字符串或时间戳对象
                timestamp_str = f"[{ts}] "
            else:
                # 如果没有时间戳，留空
                timestamp_str = ""

        elif hasattr(message, "role") and hasattr(message, "content"):
             # 处理对象类型的消息 (如 AstrBot 的 MessageSegment)
            raw_role = getattr(message, "role", "").lower()
            
            # 尝试从对象属性中获取 name
            sender_name = getattr(message, "name", None)
            
            if raw_role == "user": 
                role = sender_name if sender_name else "User"
            elif raw_role == "assistant": 
                role = "Rosa"
            else: role = "Unknown"
            
            content = _content_to_safe_text(getattr(message, "content", ""))
            timestamp_str = "" # 对象通常不带时间戳，除非我们去查库

        if content:
            # 构建单条结构化消息: [Time] Role: Content
            msg_line = f"{timestamp_str}{role}: {content}"
            selected_contents.insert(0, msg_line)
            count += 1

    return "\n".join(selected_contents)


def is_group_chat(event: AstrMessageEvent) -> bool:
    """
    判断消息是否来自群聊。
    """
    return event.get_group_id() != ""
