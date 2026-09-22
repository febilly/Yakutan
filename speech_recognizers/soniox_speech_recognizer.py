"""Soniox Speech Recognizer - 使用原生 WebSocket 实现的语音识别器"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import suppress
from typing import Any, Dict, List, Optional
import config as app_config
from resource_path import get_resource_path, get_user_data_path, ensure_dir
from proxy_detector import refresh_system_proxy_env
from vrcx_context_bridge import build_asr_context_text, get_asr_context_terms

try:
    from websockets.sync.client import connect as ws_connect
    from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from .base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
    SpeechRecognizer,
)

__all__ = ["SonioxSpeechRecognizer", "WEBSOCKETS_AVAILABLE"]

# Soniox WebSocket API 端点
SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# 断线重连策略：指数退避，封顶 30s，在服务运行期间无限重试
SONIOX_RECONNECT_INITIAL_DELAY = 1.0
SONIOX_RECONNECT_MAX_DELAY = 30.0
# 音频帧被静默丢弃时的日志节流间隔（每 5s 最多一条）
SONIOX_DROP_LOG_INTERVAL = 5.0


class SonioxSpeechRecognizer(SpeechRecognizer):
    """Speech recognizer backed by the Soniox WebSocket API.
    
    使用原生 websockets 库实现，不依赖 Soniox SDK。
    """

    def __init__(
        self,
        callback: SpeechRecognitionCallback,
        api_key: Optional[str] = None,
        model: str = "stt-rt-v3",
        sample_rate: int = 16000,
        num_channels: int = 1,
        audio_format: str = "pcm_s16le",
        language_hints: Optional[List[str]] = None,
        enable_endpoint_detection: bool = True,
        enable_language_identification: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **extra_kwargs: Any
    ) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets 库未安装，请运行: pip install websockets")
        
        self._lock = threading.Lock()
        self._callback: Optional[SpeechRecognitionCallback] = None
        self._ws = None
        self._recv_thread: Optional[threading.Thread] = None
        self._recv_stop_event = threading.Event()
        
        # 连接状态
        self._connected: bool = False
        self._should_run: bool = False
        self._paused: bool = False
        self._session_id: Optional[str] = None
        # 重连状态
        self._reconnecting: bool = False
        self._last_drop_log_time: float = 0.0
        
        # 配置参数
        self._api_key = os.environ.get("SONIOX_API_KEY", "") if api_key is None else api_key
        self._model = model
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._audio_format = audio_format
        self._language_hints = language_hints or ["en", "zh", "ja", "ko"]
        self._enable_endpoint_detection = enable_endpoint_detection
        self._enable_language_identification = enable_language_identification
        self._context = context
        self._extra_kwargs = extra_kwargs
        
        # Token 累积
        self._final_tokens: List[Dict[str, Any]] = []
        self._current_text: str = ""
        
        self.set_callback(callback)

    def set_callback(self, callback: SpeechRecognitionCallback) -> None:
        if callback is None:
            raise ValueError("callback must not be None")
        with self._lock:
            if self._ws is not None:
                raise RuntimeError("Callback already configured; create a new recognizer instance instead.")
            self._callback = callback

    def start(self) -> None:
        with self._lock:
            # 仅在连接确实存活时短路；连接已断开（_connected=False）时必须重建，
            # 否则残留的 _ws 会让 start() 永远无法恢复（审查 P1-4）。
            if self._connected and self._ws is not None:
                return
            self._should_run = True
            self._paused = False
            self._final_tokens = []
            self._current_text = ""
        
        self._connect()

    def _connect(self) -> None:
        """建立 WebSocket 连接并发送配置。
        
        会话语义（语言提示/上下文/热词/VRCX 上下文）通过 _build_config()
        在每次建连时重放，因此重连后自动恢复。
        """
        try:
            print("[Soniox] Connecting to Soniox...")
            refresh_system_proxy_env()
            ws = ws_connect(SONIOX_WEBSOCKET_URL)
            
            # 构建配置消息
            config = self._build_config()
            ws.send(json.dumps(config))
            
            # 启动接收线程（stop_event 与 ws 绑定，避免新旧连接串扰）
            stop_event = threading.Event()
            recv_thread = threading.Thread(
                target=self._recv_worker,
                args=(ws, stop_event),
                daemon=True,
                name="SonioxRecvThread"
            )
            with self._lock:
                # 锁内提交连接状态：_ws 赋值必须在锁内（审查 P1-4），
                # 并防御并发建连/已停止的竞态。
                if not self._should_run or (self._connected and self._ws is not None):
                    stale = True
                else:
                    self._recv_stop_event = stop_event
                    self._ws = ws
                    self._recv_thread = recv_thread
                    self._connected = True
                    stale = False
            if stale:
                # 已停止或已有其他路径完成连接：丢弃本次握手
                with suppress(Exception):
                    ws.close()
                return
            recv_thread.start()
            
            print("[Soniox] Connection established successfully.")
            
            if self._callback:
                self._callback.on_session_started()
                
        except Exception as e:
            print(f"[Soniox] Connection failed: {e}")
            self._cleanup()
            raise
    
    def _maybe_start_reconnect_thread(self) -> None:
        """在服务仍应运行且连接已断开时，启动后台受控重连线程（幂等）。"""
        with self._lock:
            if not self._should_run:
                return
            if self._connected and self._ws is not None:
                return
            if self._reconnecting:
                return
            self._reconnecting = True
        thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="SonioxReconnectThread"
        )
        thread.start()
    
    def _reconnect_loop(self) -> None:
        """后台重连循环：指数退避（封顶 SONIOX_RECONNECT_MAX_DELAY），
        在 _should_run 为真期间无限重试，成功或服务停止后退出。
        """
        delay = SONIOX_RECONNECT_INITIAL_DELAY
        try:
            while True:
                with self._lock:
                    if not self._should_run:
                        print("[Soniox] Service stopped; cancel reconnection.")
                        return
                    if self._connected and self._ws is not None:
                        return
                print(f"[Soniox] Connection lost; retrying in {delay:.1f}s (auto reconnect, backoff capped at {SONIOX_RECONNECT_MAX_DELAY:.0f}s)...")
                time.sleep(delay)
                delay = min(delay * 2.0, SONIOX_RECONNECT_MAX_DELAY)
                with self._lock:
                    if not self._should_run:
                        return
                try:
                    self._connect()
                    print("[Soniox] Reconnected successfully.")
                    return
                except Exception as e:
                    print(f"[Soniox] Reconnect attempt failed: {e}")
        finally:
            with self._lock:
                self._reconnecting = False

    def _build_config(self) -> Dict[str, Any]:
        """构建发送给 Soniox 的配置消息"""
        config: Dict[str, Any] = {
            "api_key": self._api_key,
            "model": self._model,
            "audio_format": self._audio_format,
            "sample_rate": self._sample_rate,
            "num_channels": self._num_channels,
            "enable_endpoint_detection": self._enable_endpoint_detection,
        }
        
        if self._language_hints:
            config["language_hints"] = self._language_hints
        
        if self._enable_language_identification:
            config["enable_language_identification"] = True
        
        # Context handling:
        # 1. If user provided `context` via constructor, honor it (dict or list).
        # 2. Otherwise, if hot words are enabled in config, load terms from
        #    hot_words/ and hot_words_private/ text files and pass them as
        #    context {"terms": [...]} to Soniox.
        context_payload: Optional[Dict[str, Any]] = None
        if self._context:
            if isinstance(self._context, dict):
                # Allow either {'context': {...}} or a direct dict with keys like 'terms'/'text'
                if "context" in self._context and isinstance(self._context["context"], dict):
                    context_payload = self._context["context"]
                else:
                    context_payload = self._context
            elif isinstance(self._context, list):
                context_payload = {"terms": self._context}

        # Try loading hot words from resource files when no explicit context provided
        if context_payload is None and getattr(app_config, "ENABLE_HOT_WORDS", False):
            terms: List[str] = []
            try:
                hot_dir = get_resource_path(app_config.HOT_WORDS_DIR)
                hot_private_dir = get_user_data_path(app_config.HOT_WORDS_PRIVATE_DIR)
                # Ensure private dir exists (may be created by user code elsewhere)
                ensure_dir(hot_private_dir)

                for dirpath in (hot_dir, hot_private_dir):
                    if not os.path.isdir(dirpath):
                        continue
                    for fname in sorted(os.listdir(dirpath)):
                        if not fname.lower().endswith(".txt"):
                            continue
                        fpath = os.path.join(dirpath, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8") as fh:
                                for line in fh:
                                    w = line.strip()
                                    if not w or w.startswith("#"):
                                        continue
                                    if w not in terms:
                                        terms.append(w)
                        except Exception:
                            # ignore read errors per-file
                            continue

                if terms:
                    # limit to reasonable size (align with HotWordsManager limits)
                    context_payload = {"terms": terms[:500]}
            except Exception:
                # Fail silently; do not prevent recognizer from starting
                context_payload = None

        vrcx_terms = get_asr_context_terms()
        vrcx_text = build_asr_context_text("")
        if vrcx_terms or vrcx_text:
            if context_payload is None:
                context_payload = {}
            elif not isinstance(context_payload, dict):
                context_payload = {"terms": list(context_payload)}

            merged_terms: List[str] = []
            seen_terms = set()
            existing_terms = context_payload.get("terms")
            if isinstance(existing_terms, list):
                for term in existing_terms:
                    text = str(term or "").strip()
                    if text and text not in seen_terms:
                        seen_terms.add(text)
                        merged_terms.append(text)
            for term in vrcx_terms:
                if term and term not in seen_terms:
                    seen_terms.add(term)
                    merged_terms.append(term)
            if merged_terms:
                context_payload["terms"] = merged_terms[:500]

            if vrcx_text:
                existing_text = str(context_payload.get("text") or "").strip()
                context_payload["text"] = (
                    f"{existing_text}\n\n{vrcx_text}" if existing_text else vrcx_text
                )

        if context_payload:
            config["context"] = context_payload
        
        # 合并额外参数
        config.update(self._extra_kwargs)
        
        return config

    def _recv_worker(self, ws: Any, stop_event: threading.Event) -> None:
        """接收线程：从 WebSocket 读取消息并处理。
        
        退出时（无论正常关闭还是异常断开）在锁内置 _ws=None、_connected=False；
        若服务仍应运行（_should_run），则触发受控自动重连。
        """
        error: Optional[Exception] = None
        try:
            while not stop_event.is_set():
                try:
                    message = ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                except ConnectionClosedOK:
                    print("[Soniox] Connection closed normally.")
                    break
                except ConnectionClosedError as e:
                    print(f"[Soniox] Connection closed with error: {e}")
                    break
                except Exception as e:
                    print(f"[Soniox] Error receiving message: {e}")
                    break
                
                self._handle_message(message)
                
        except Exception as e:
            error = e
            print(f"[Soniox] Receive thread error: {e}")
        finally:
            was_current = False
            with self._lock:
                if self._ws is ws:
                    self._ws = None
                    self._connected = False
                    was_current = True
            if was_current:
                # 仅当本线程仍拥有当前连接时才回调，避免旧连接的退出
                # 干扰已经重建的新会话。
                if error is not None and self._callback:
                    self._callback.on_error(error)
                if self._callback:
                    self._callback.on_session_stopped()
            # 服务仍在运行时触发受控重连（指数退避，无限重试）
            self._maybe_start_reconnect_thread()

    def _handle_message(self, message: str) -> None:
        """处理从 Soniox 接收的消息"""
        try:
            res = json.loads(message)
        except json.JSONDecodeError as e:
            print(f"[Soniox] Failed to parse message: {e}")
            return
        
        # 检查错误
        if res.get("error_code") is not None:
            error_msg = f"Soniox error: {res.get('error_code')} - {res.get('error_message', 'Unknown error')}"
            print(f"[Soniox] {error_msg}")
            if self._callback:
                self._callback.on_error(RuntimeError(error_msg))
            return
        
        # 处理 tokens
        tokens = res.get("tokens", [])
        if not tokens:
            # 检查是否结束
            if res.get("finished"):
                print("[Soniox] Session finished.")
            return
        
        # 分离 final 和 non-final tokens
        non_final_tokens: List[Dict[str, Any]] = []
        new_final_tokens: List[Dict[str, Any]] = []
        
        for token in tokens:
            text = token.get("text", "")
            if not text:
                continue
            
            if token.get("is_final"):
                new_final_tokens.append(token)
            else:
                non_final_tokens.append(token)
        
        # 累积 final tokens
        self._final_tokens.extend(new_final_tokens)
        
        # 需要过滤的特殊 token
        SPECIAL_TOKENS = {"<end>", "<fin>"}
        
        # 检查是否有 <end> 或 <fin> token（endpoint detection 或 manual finalization）
        has_endpoint = any(t.get("text") in SPECIAL_TOKENS for t in new_final_tokens)
        
        # 构建当前文本（过滤掉特殊 token）
        final_text = "".join(t.get("text", "") for t in self._final_tokens if t.get("text") not in SPECIAL_TOKENS)
        non_final_text = "".join(t.get("text", "") for t in non_final_tokens if t.get("text") not in SPECIAL_TOKENS)
        
        combined_text = final_text + non_final_text
        combined_text = combined_text.strip()
        
        # 如果有 endpoint，发送 final 事件并重置
        if has_endpoint and final_text.strip():
            event = RecognitionEvent(
                text=final_text.strip(),
                is_final=True,
                raw={"tokens": self._final_tokens}
            )
            if self._callback:
                self._callback.on_result(event)
            
            # 重置 final tokens
            self._final_tokens = []
            self._current_text = ""
        elif combined_text and combined_text != self._current_text:
            # 发送部分结果
            self._current_text = combined_text
            event = RecognitionEvent(
                text=combined_text,
                is_final=False,
                raw={"tokens": tokens}
            )
            if self._callback:
                self._callback.on_result(event)

    def stop(self) -> None:
        with self._lock:
            self._should_run = False
        
        self._cleanup()
        
        with self._lock:
            self._paused = False

    def _cleanup(self) -> None:
        """清理资源"""
        # 停止接收线程
        self._recv_stop_event.set()
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=2.0)
        self._recv_thread = None
        
        # 关闭 WebSocket（接收线程的 finally 通常已先清空 _ws）
        ws = self._ws
        if ws:
            with suppress(Exception):
                # 发送空字符串表示结束
                ws.send("")
            with suppress(Exception):
                ws.close()
        
        with self._lock:
            self._ws = None
            self._connected = False

    def send_audio_frame(self, data: bytes) -> None:
        if not data:
            return
        
        with self._lock:
            if self._paused:
                return
            if not self._connected or self._ws is None:
                # 未连接时静默丢弃会掩盖断线（审查 P1-4）：改为节流日志提示
                self._log_throttled_drop()
                return
            ws = self._ws
        
        try:
            # Soniox 接收原始 PCM 字节数据
            ws.send(data)
        except Exception as e:
            print(f"[Soniox] Error sending audio: {e}")
            with self._lock:
                if self._ws is ws:
                    self._connected = False
            # 主动关闭以唤醒接收线程，尽快走断线重连路径
            with suppress(Exception):
                ws.close()
    
    def _log_throttled_drop(self) -> None:
        """连接未就绪丢弃音频帧时的节流日志（默认每 5s 最多一条）。调用方需持有 _lock。"""
        now = time.monotonic()
        if now - self._last_drop_log_time < SONIOX_DROP_LOG_INTERVAL:
            return
        self._last_drop_log_time = now
        print("[Soniox] Not connected; dropping audio frame (auto reconnect in progress)")

    def pause(self) -> None:
        with self._lock:
            if self._paused:
                return
            self._paused = True
            ws = self._ws if self._connected else None
        
        # 发送 finalize 消息强制结束当前句子
        if ws is not None:
            try:
                finalize_msg = json.dumps({"type": "finalize"})
                ws.send(finalize_msg)
            except Exception as e:
                print(f"[Soniox] Error sending finalize: {e}")

    def resume(self) -> None:
        with self._lock:
            if not self._paused:
                return
            self._paused = False
            needs_reconnect = not (self._connected and self._ws is not None)
        
        if needs_reconnect:
            # 暂停期间连接已断开：恢复时重建连接（审查 P1-4：resume 不再只清标志位）
            try:
                self._connect()
            except Exception as e:
                print(f"[Soniox] resume() reconnect failed: {e}; falling back to background reconnect")
                self._maybe_start_reconnect_thread()

    def get_last_request_id(self) -> Optional[str]:
        with self._lock:
            return self._session_id

    def get_first_package_delay(self) -> Optional[int]:
        # Soniox API 不提供此信息
        return None

    def get_last_package_delay(self) -> Optional[int]:
        # Soniox API 不提供此信息
        return None
