"""
OSC (Open Sound Control) 管理模块
负责处理VRChat的OSC通信，包括接收静音消息和发送聊天框消息
"""
import asyncio
import logging
import time
import threading
import os
from typing import Optional
from dataclasses import dataclass
from vrchat_oscquery.common import dict_to_dispatcher, vrc_client
import vrchat_oscquery.common as vrchat_osc_common
from vrchat_oscquery.threaded import vrc_osc

__all__ = ["OSCManager", "osc_manager"]

logger = logging.getLogger(__name__)

# 定义发送到VRChat聊天框的最大文本长度
MAX_LENGTH = 144

# 翻译头部行（None 或空字符串表示禁用）
TRANSLATION_HEADER = ""

# 消息优先级
PRIORITY_HIGH = 1  # 最终确认的消息
PRIORITY_LOW = 2   # ongoing 消息

VRCHAT_MUTE_PATH = "/avatar/parameters/MuteSelf"

@dataclass
class QueuedMessage:
    """待发送的消息实体"""
    text: str
    ongoing: bool
    priority: int
    timestamp: float


@dataclass
class HistoryMessage:
    """历史消息实体，用于拼接发送"""
    text: str
    timestamp: float
    speaker: str = "?"


class OSCManager:
    """OSC管理器单例类，负责OSC服务器和客户端的管理"""
    
    _instance = None
    _client = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OSCManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, truncate_messages: Optional[bool] = None):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._client = None
            self._mute_callback = None  # 静音状态变化的回调函数
            self._oscquery_enabled = str(os.environ.get("OSC_QUERY_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")
            self._oscquery_app_name = str(os.environ.get("OSCQUERY_APP_NAME", "DeafaultAppName")).strip() or "DeafaultAppName"
            
            self._last_mute_value: Optional[bool] = None
            self._oscquery_lock = threading.Lock()
            self._oscquery_connected = False
            self._oscquery_httpd = None
            self._vrchat_linked_logged = False
            
            # 发送节流配置（仅保留一个待发消息）
            self._cooldown_seconds = 1.5  # 发送冷却时间（秒）
            self._last_send_time = 0.0  # 上次发送时间
            self._pending_message: Optional[QueuedMessage] = None
            self._pending_timer: Optional[threading.Timer] = None
            self._state_lock = threading.Lock()
            self._truncate_enabled = True
            self._message_history: list[HistoryMessage] = []
            self._history_ttl_seconds = 10.0
            self._header_line = TRANSLATION_HEADER
            
            self._emit("[OSC] OSC manager initialized")
        if truncate_messages is not None:
            self._truncate_enabled = bool(truncate_messages)

    def _emit(self, message: str, level: str = "info"):
        print(message)
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
        else:
            logger.info(message)
    
    def set_mute_callback(self, callback):
        """
        设置静音状态变化的回调函数
        
        Args:
            callback: 回调函数，接收一个布尔参数 (mute_value)
                     当收到 MuteSelf=True 时调用 callback(True)
                     当收到 MuteSelf=False 时调用 callback(False)
        """
        self._mute_callback = callback
        self._emit("[OSC] Mute callback registered")
    
    def clear_mute_callback(self):
        """清除静音状态回调函数"""
        self._mute_callback = None
        self._emit("[OSC] Mute callback cleared")
    
    def get_udp_client(self):
        """获取OSC UDP客户端实例（用于发送消息）"""
        if self._client is None:
            self._client = vrc_client()
            self._emit("[OSC] UDP client created")
        return self._client

    def _notify_mute_callback(self, mute_value: bool):
        if self._mute_callback is None:
            logger.debug("[OSC] Mute callback is not set; ignoring MuteSelf update")
            return

        try:
            if asyncio.iscoroutinefunction(self._mute_callback):
                try:
                    running_loop = asyncio.get_running_loop()
                    running_loop.create_task(self._mute_callback(mute_value))
                except RuntimeError:
                    asyncio.run(self._mute_callback(mute_value))
            else:
                self._mute_callback(mute_value)
        except Exception as e:
            self._emit(f"[OSC] Error while invoking mute callback: {e}", level="error")
    
    def _handle_mute_self(self, address, *args):
        """处理来自OSC的MuteSelf消息"""
        if args and len(args) > 0:
            mute_value = bool(args[0])
            previous = self._last_mute_value
            self._last_mute_value = mute_value
            if previous == mute_value:
                return
            self._emit(f"[OSC] Received MuteSelf: {mute_value}")
            if not self._vrchat_linked_logged:
                self._vrchat_linked_logged = True
                self._oscquery_connected = True
                self._emit("[OSCQuery] Linked with VRChat (received first MuteSelf event)")
            self._notify_mute_callback(mute_value)
    
    async def start_server(self, app_name: Optional[str] = None):
        """Start OSCQuery service and wait for VRChat callbacks."""
        if not self._oscquery_enabled:
            self._emit("[OSC] OSCQuery is disabled by config; skipping startup", level="warning")
            return None

        if app_name is not None:
            normalized_name = str(app_name).strip()
            if normalized_name:
                self._oscquery_app_name = normalized_name

        with self._oscquery_lock:
            if self._oscquery_httpd is not None:
                self._emit("[OSC] OSCQuery service is already running")
                return None
            try:
                self._oscquery_httpd = await asyncio.to_thread(self._start_oscquery_service_blocking)
                self._vrchat_linked_logged = False
                self._oscquery_connected = False
                self._last_mute_value = None
                self._emit(f"[OSCQuery] Service published as '{self._oscquery_app_name}'. Waiting for VRChat to connect...")
            except Exception as error:
                self._emit(f"[OSCQuery] Failed to publish OSCQuery service: {error!r}", level="error")
                self._oscquery_httpd = None
            return None

    def _start_oscquery_service_blocking(self):
        vrchat_osc_common.APP_HOST = "127.0.0.1"
        dispatcher = dict_to_dispatcher({VRCHAT_MUTE_PATH: self._handle_mute_self})
        return vrc_osc(self._oscquery_app_name, dispatcher, foreground=False)
    
    async def stop_server(self):
        """Stop OSCQuery service."""
        # 取消待处理消息
        with self._state_lock:
            if self._pending_timer is not None:
                self._pending_timer.cancel()
                self._pending_timer = None
            self._pending_message = None

        with self._oscquery_lock:
            httpd = self._oscquery_httpd
            self._oscquery_httpd = None
        if httpd is not None:
            try:
                httpd.shutdown()
                httpd.server_close()
                self._emit("[OSCQuery] Service stopped")
            except Exception as error:
                self._emit(f"[OSCQuery] Error while stopping service: {error}", level="warning")
        self._oscquery_connected = False
        self._vrchat_linked_logged = False
        self._last_mute_value = None
    
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

    def _prune_history_locked(self, now: float):
        """移除超过 TTL 的历史消息（需要在锁内调用）"""
        ttl = getattr(self, "_history_ttl_seconds", 10.0)
        self._message_history = [msg for msg in self._message_history if now - msg.timestamp <= ttl]

    def _build_combined_history_locked(self) -> str:
        """组合历史消息，最多9行且超长时优先丢弃旧消息，始终保留头部（需要在锁内调用）"""
        header = getattr(self, "_header_line", "") or ""
        header_enabled = bool(header)
        max_lines = 9

        # 构造行：前缀含说话人
        lines = []
        if header_enabled:
            lines.append(header)

        for msg in self._message_history:
            prefix = f"S{msg.speaker}：" if msg.speaker else "S？："
            lines.append(f"{prefix}{msg.text}")

        # 限制行数（保留最新），头部算一行
        while len(lines) > max_lines:
            # 如果有头部，先尝试移除最早的消息行（不能移除头部）
            if header_enabled and len(lines) == 1:
                break
            if header_enabled and len(lines) > 1:
                # 删除 header 之后的第一条消息
                lines.pop(1)
                if self._message_history:
                    self._message_history.pop(0)
            else:
                lines.pop(0)
                if self._message_history:
                    self._message_history.pop(0)

        def assemble(line_list):
            return "\n".join(line_list)

        combined = assemble(lines)
        if not combined:
            return ""

        # 超长时优先删除旧的整条消息（不删除头部）
        while len(combined) > MAX_LENGTH and len(lines) > 1:
            # 删除 header 后的第一条消息，如果没有 header 就删第一条
            drop_index = 1 if header_enabled and len(lines) > 1 else 0
            if drop_index < len(lines):
                lines.pop(drop_index)
            if self._message_history:
                self._message_history.pop(0)
            combined = assemble(lines)

        if len(combined) > MAX_LENGTH and len(lines) >= 1:
            # 仅剩头部 + 最新一条或只有一条消息仍然超长，截断最新消息
            header_overhead = len(lines[0]) + 1 if header_enabled and len(lines) > 1 else (len(lines[0]) if header_enabled else 0)
            if header_enabled and len(lines) > 1:
                latest_idx = len(lines) - 1
                budget = max(0, MAX_LENGTH - header_overhead)
                body = lines[latest_idx]
                truncated_body = self._truncate_text(body, max_length=budget if budget > 0 else 0)
                lines[latest_idx] = truncated_body
                if self._message_history:
                    self._message_history[-1] = HistoryMessage(
                        text=truncated_body.split("：", 1)[-1] if "：" in truncated_body else truncated_body,
                        timestamp=self._message_history[-1].timestamp,
                        speaker=self._message_history[-1].speaker,
                    )
                combined = assemble(lines)
            elif not header_enabled and lines:
                latest_idx = len(lines) - 1
                budget = MAX_LENGTH
                body = lines[latest_idx]
                truncated_body = self._truncate_text(body, max_length=budget)
                lines[latest_idx] = truncated_body
                if self._message_history:
                    self._message_history[-1] = HistoryMessage(
                        text=truncated_body.split("：", 1)[-1] if "：" in truncated_body else truncated_body,
                        timestamp=self._message_history[-1].timestamp,
                        speaker=self._message_history[-1].speaker,
                    )
                combined = assemble(lines)

        return combined

    def clear_history(self):
        """清空历史消息（线程安全）"""
        with self._state_lock:
            self._message_history.clear()

    def add_message_and_send(self, text: str, ongoing: bool = False, speaker: Optional[str] = None):
        """记录消息并按历史拼接发送，自动清理过期消息"""
        safe_text = (text or "").strip()
        if not safe_text:
            return

        speaker_label = (speaker or "").strip()
        speaker_label = speaker_label if speaker_label else "?"

        now = time.time()
        with self._state_lock:
            self._prune_history_locked(now)
            self._message_history.append(HistoryMessage(text=safe_text, timestamp=now, speaker=speaker_label))
            combined = self._build_combined_history_locked()

        if combined:
            # 使用已有的发送节流逻辑
            self.send_text_sync(combined, ongoing)
    
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
                logger.debug("[OSC] Cooldown active; delaying pending message")
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
            logger.info(f"[OSC] Sent chatbox message: '{text}' (ongoing={ongoing})")
        except Exception as e:
            logger.error(f"[OSC] Failed to send OSC message: {e}")
    
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
            logger.debug(f"[OSC] Set typing state: {typing}")
        except Exception as e:
            logger.error(f"[OSC] Failed to set typing state: {e}")
    
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
                        logger.debug("[OSC] Dropped low-priority message; high-priority pending exists")
                        return
                    logger.debug(
                        "[OSC] Replaced pending message priority %s -> %s",
                        self._pending_message.priority,
                        priority,
                    )
                else:
                    logger.debug("[OSC] Added pending message (priority=%s)", priority)

                self._pending_message = message
                self._schedule_pending_send_locked()

        if send_now is not None:
            self._send_message_immediately(send_now.text, send_now.ongoing)


# 创建全局单例实例
osc_manager = OSCManager()
