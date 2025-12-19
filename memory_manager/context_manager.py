import threading
import time

from astrbot.api.event import AstrMessageEvent


class ConversationContextManager:
    """
    会话上下文管理器

    M18 修复: 改进并发安全性
    - 在异步环境中，如果所有操作都在同一个事件循环线程中执行，threading.RLock 是安全的
    - 保留 RLock 用于同步代码路径
    - 添加注释说明并发安全策略
    """

    def __init__(self):
        self.conversations: dict[str, dict] = {}
        # 使用 RLock 保证线程安全
        # 注意: 这个类的方法主要在 asyncio 事件循环中调用
        # RLock 可以保护同步代码路径，对于异步代码，Python 的 GIL 和单线程事件循环提供了基本保护
        # 如果未来需要真正的异步锁，应该使用 asyncio.Lock
        self._lock = threading.RLock()

    def init_conv(self, session_id: str, contexts: list[dict], event: AstrMessageEvent):
        """
        初始化会话上下文 (仅存储元数据)
        """
        with self._lock:
            if session_id in self.conversations:
                return
            self.conversations[session_id] = {}
            # [Optimization] 移除内存历史记录
            # self.conversations[session_id]["history"] = contexts
            self.conversations[session_id]["event"] = event
            # 初始化最后一次总结的时间
            self.conversations[session_id]["last_summary_time"] = time.time()
            return

    def add_message(self, session_id: str, role: str, content: str) -> str | None:
        """
        添加对话消息 (仅确保会话存在，不再存储内容)
        :param session_id: 会话ID
        :param role: 角色（user/assistant）
        :param content: 对话内容
        :return: None
        """
        with self._lock:
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    # "history": [], # [Optimization] 移除
                    "last_summary_time": time.time(),
                }
            
            # [Optimization] 不再追加历史记录
            # conversation = self.conversations[session_id]
            # conversation["history"].append(...)
            pass

    def get_summary_time(self, session_id: str) -> float:
        """
        获取最后一次总结时间
        """
        with self._lock:
            if session_id in self.conversations:
                return self.conversations[session_id]["last_summary_time"]
            else:
                return 0

    def update_summary_time(self, session_id: str):
        """
        更新最后一次总结时间
        """
        with self._lock:
            if session_id in self.conversations:
                self.conversations[session_id]["last_summary_time"] = time.time()

    def get_history(self, session_id: str) -> list[dict]:
        """
        获取对话历史记录 (已废弃，始终返回空列表)
        :param session_id: 会话ID
        :return: 空列表
        """
        return []
        # with self._lock:
        #     if session_id in self.conversations:
        #         return self.conversations[session_id]["history"]
        #     else:
        #         return []

    def get_session_context(self, session_id: str):
        """
        获取session_id对应的所有信息
        """
        with self._lock:
            if session_id in self.conversations:
                return self.conversations[session_id]
            else:
                return {}
