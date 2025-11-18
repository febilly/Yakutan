"""
OSC (Open Sound Control) 管理模块
负责处理VRChat的OSC通信，包括接收静音消息和发送聊天框消息
"""
import asyncio
import logging
import time
import threading
from typing import Optional
from dataclasses import dataclass
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

__all__ = ["OSCManager", "osc_manager"]

logger = logging.getLogger(__name__)

# 定义发送到VRChat聊天框的最大文本长度
MAX_LENGTH = 144

# 消息优先级
PRIORITY_HIGH = 1  # 最终确认的消息
PRIORITY_LOW = 2   # ongoing 消息

@dataclass
class QueuedMessage:
    """待发送的消息实体"""
    text: str
    ongoing: bool
    priority: int
    timestamp: float

class OSCManager:
    """OSC管理器单例类，负责OSC服务器和客户端的管理"""
    
    _instance = None
    _server = None
    _client = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OSCManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, truncate_messages: Optional[bool] = None):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._server = None
            self._transport = None  # 保存transport用于关闭
            self._client = None
            self._mute_callback = None  # 静音状态变化的回调函数
            
            # OSC客户端配置（发送到VRChat）
            self._osc_client_host = "127.0.0.1"
            self._osc_client_port = 9000
            
            # OSC服务器配置（接收来自VRChat）
            self._osc_server_host = "127.0.0.1"
            self._osc_server_port = 9001
            
            # 发送节流配置（仅保留一个待发消息）
            self._cooldown_seconds = 1.5  # 发送冷却时间（秒）
            self._last_send_time = 0.0  # 上次发送时间
            self._pending_message: Optional[QueuedMessage] = None
            self._pending_timer: Optional[threading.Timer] = None
            self._state_lock = threading.Lock()
            self._truncate_enabled = True
            
            logger.info("[OSC] OSC管理器已初始化")
        if truncate_messages is not None:
            self._truncate_enabled = bool(truncate_messages)
    
    def set_mute_callback(self, callback):
        """
        设置静音状态变化的回调函数
        
        Args:
            callback: 回调函数，接收一个布尔参数 (mute_value)
                     当收到 MuteSelf=True 时调用 callback(True)
                     当收到 MuteSelf=False 时调用 callback(False)
        """
        self._mute_callback = callback
        logger.info("[OSC] 已设置静音状态回调函数")
    
    def clear_mute_callback(self):
        """清除静音状态回调函数"""
        self._mute_callback = None
        logger.info("[OSC] 已清除静音状态回调函数")
    
    def get_udp_client(self):
        """获取OSC UDP客户端实例（用于发送消息）"""
        if self._client is None:
            self._client = udp_client.SimpleUDPClient(
                self._osc_client_host,
                self._osc_client_port
            )
            logger.info(f"[OSC] OSC客户端已创建，目标地址: {self._osc_client_host}:{self._osc_client_port}")
        return self._client
    
    def _handle_mute_self(self, address, *args):
        """处理来自OSC的MuteSelf消息"""
        if args and len(args) > 0:
            mute_value = args[0]
            logger.info(f"[OSC] 收到MuteSelf消息: {mute_value}")
            
            # 如果设置了回调函数，则调用它
            if self._mute_callback is not None:
                try:
                    # 如果回调是协程函数，需要创建任务
                    if asyncio.iscoroutinefunction(self._mute_callback):
                        asyncio.create_task(self._mute_callback(mute_value))
                    else:
                        self._mute_callback(mute_value)
                except Exception as e:
                    logger.error(f"[OSC] 调用静音回调函数时出错: {e}")
            else:
                logger.debug(f"[OSC] 未设置静音回调函数，忽略MuteSelf消息")
    
    async def start_server(self):
        """启动OSC服务器监听（全局单例）"""
        if self._server is not None:
            logger.info("[OSC] OSC服务器已在运行中")
            return
        
        dispatcher = Dispatcher()
        dispatcher.map("/avatar/parameters/MuteSelf", self._handle_mute_self)
        
        self._server = AsyncIOOSCUDPServer(
            (self._osc_server_host, self._osc_server_port),
            dispatcher,
            asyncio.get_event_loop()
        )
        
        self._transport, protocol = await self._server.create_serve_endpoint()
        logger.info(f"[OSC] OSC服务器已启动，监听地址: {self._osc_server_host}:{self._osc_server_port}")
        return self._transport
    
    async def stop_server(self):
        """停止OSC服务器"""
        # 取消待处理消息
        with self._state_lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            self._pending_message = None
        
        if self._transport is not None:
            self._transport.close()
            logger.info("[OSC] OSC服务器transport已关闭")
            self._transport = None
        
        if self._server is not None:
            self._server = None
            logger.info("[OSC] OSC服务器已停止")
    
    def _truncate_text(self, text: str, max_length: int = 144) -> str:
        """
        截断过长的文本，优先删除前面的句子
        
        Args:
            text: 需要截断的文本
            max_length: 最大长度限制
            
        Returns:
            截断后的文本
        """
        if not getattr(self, "_truncate_enabled", True):
            return text

        if len(text) <= max_length:
            return text
        
        # 句子结束标记
        SENTENCE_ENDERS = [
            '.', '?', '!', ',',           # Common
            '。', '？', '！', '，',        # CJK
            '…', '...', '‽',             # Stylistic & Special (includes 3-dot ellipsis)
            '։', '؟', ';', '،',           # Armenian, Arabic, Greek (as question mark), Arabic comma
            '।', '॥', '።', '။', '།',    # Indic, Ethiopic, Myanmar, Tibetan
            '、', '‚', '٫'               # Japanese enumeration comma, low comma, Arabic decimal separator
        ]
        
        # 当文本超长时，删除最前面的句子而不是截断末尾
        while len(text) > max_length:
            # 尝试找到第一个句子的结束位置
            first_sentence_end = -1
            for ender in SENTENCE_ENDERS:
                idx = text.find(ender)
                if idx != -1 and (first_sentence_end == -1 or idx < first_sentence_end):
                    first_sentence_end = idx
            
            if first_sentence_end != -1:
                # 删除第一个句子（包括标点符号后的空格）
                text = text[first_sentence_end + 1:].lstrip()
            else:
                # 如果没有找到标点符号，删除前面的字符直到长度合适
                text = text[len(text) - max_length:]
                break
        
        return text
    
    def _schedule_pending_send_locked(self):
        """在锁内调用，安排发送待处理消息"""
        if self._pending_message is None:
            self._pending_timer = None
            return

        if self._pending_timer is not None:
            self._pending_timer.cancel()

        wait = self._cooldown_seconds - (time.time() - self._last_send_time)
        if wait <= 0:
            wait = 0.01  # 避免忙等待，稍作延迟

        timer = threading.Timer(wait, self._flush_pending_message)
        timer.daemon = True
        self._pending_timer = timer
        timer.start()

    def _flush_pending_message(self):
        """发送当前待处理的消息（如果冷却结束）"""
        with self._state_lock:
            message = self._pending_message
            if not message:
                self._pending_timer = None
                return

            elapsed = time.time() - self._last_send_time
            if elapsed < self._cooldown_seconds:
                # 冷却尚未结束，重新安排
                logger.debug("[OSC] 冷却未结束，延后发送待处理消息")
                self._schedule_pending_send_locked()
                return

            # 可以发送
            self._pending_message = None
            self._pending_timer = None
            self._last_send_time = time.time()
            text = message.text
            ongoing = message.ongoing

        self._send_message_immediately(text, ongoing)
    
    def _send_message_immediately(self, text: str, ongoing: bool):
        """
        立即发送消息到VRChat聊天框（内部方法）
        
        Args:
            text: 要发送的文本
            ongoing: 是否正在输入中
        """
        try:
            client = self.get_udp_client()
            client.send_message("/chatbox/typing", ongoing)
            client.send_message("/chatbox/input", [text, True, not ongoing])
            logger.info(f"[OSC] 发送聊天框消息: '{text}' (ongoing={ongoing})")
        except Exception as e:
            logger.error(f"[OSC] 发送OSC消息失败: {e}")
    
    async def set_typing(self, typing: bool):
        """兼容旧调用方式的异步接口"""
        if hasattr(asyncio, "to_thread"):
            await asyncio.to_thread(self.set_typing_sync, typing)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.set_typing_sync, typing)

    def set_typing_sync(self, typing: bool):
        """
        设置 VRChat 聊天框的 typing 状态
        
        Args:
            typing: True 表示正在输入，False 表示停止输入
        """
        try:
            client = self.get_udp_client()
            client.send_message("/chatbox/typing", typing)
            logger.debug(f"[OSC] 设置 typing 状态: {typing}")
        except Exception as e:
            logger.error(f"[OSC] 设置 typing 状态失败: {e}")
    
    async def send_text(self, text: str, ongoing: bool):
        """兼容旧调用方式的异步接口"""
        if hasattr(asyncio, "to_thread"):
            await asyncio.to_thread(self.send_text_sync, text, ongoing)
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.send_text_sync, text, ongoing)

    def send_text_sync(self, text: str, ongoing: bool):
        """发送文本到 VRChat（带冷却，最多保留一个待发消息）"""
        # 截断过长的文本
        text = self._truncate_text(text, max_length=MAX_LENGTH)
        
        # 确定优先级
        priority = PRIORITY_LOW if ongoing else PRIORITY_HIGH

        message = QueuedMessage(
            text=text,
            ongoing=ongoing,
            priority=priority,
            timestamp=time.time(),
        )

        send_now = None

        with self._state_lock:
            now = time.time()
            elapsed = now - self._last_send_time
            can_send_now = elapsed >= self._cooldown_seconds and self._pending_message is None

            if can_send_now:
                self._last_send_time = now
                send_now = message
            else:
                if self._pending_message is not None:
                    if priority == PRIORITY_LOW and self._pending_message.priority == PRIORITY_HIGH:
                        logger.debug("[OSC] 丢弃低优先级消息，已有高优先级待发送")
                        return
                    logger.debug(
                        "[OSC] 替换待发送消息 priority %s -> %s",
                        self._pending_message.priority,
                        priority,
                    )
                else:
                    logger.debug("[OSC] 新增待发送消息 (priority=%s)", priority)

                self._pending_message = message
                self._schedule_pending_send_locked()

        if send_now is not None:
            self._send_message_immediately(send_now.text, send_now.ongoing)


# 创建全局单例实例
osc_manager = OSCManager()
