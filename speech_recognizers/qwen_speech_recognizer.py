from __future__ import annotations

import base64
from contextlib import suppress
import threading
import time
from typing import Any, Dict, Optional

from dashscope.audio.qwen_omni import (
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)

try:
    from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
except ImportError:  # pragma: no cover
    class TranscriptionParams:  # type: ignore[override]
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

from .base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
    SpeechRecognizer,
)

__all__ = ["QwenSpeechRecognizer"]


class _QwenOmniCallbackAdapter(OmniRealtimeCallback):
    """Bridge Qwen realtime callbacks to the generic recognizer callback."""

    def __init__(
        self,
        recognizer: "QwenSpeechRecognizer",
        user_callback: SpeechRecognitionCallback,
    ) -> None:
        self._recognizer = recognizer
        self._user_callback = user_callback
        self._conversation: Optional[OmniRealtimeConversation] = None
        self._items: Dict[str, Dict[str, str]] = {}

    def attach_conversation(self, conversation: OmniRealtimeConversation) -> None:
        self._conversation = conversation

    def detach(self) -> None:
        self._conversation = None

    # ------------------------------------------------------------------
    # OmniRealtimeCallback interface
    # ------------------------------------------------------------------
    def on_open(self) -> None:  # type: ignore[override]
        self._user_callback.on_session_started()

    def on_close(self, code, msg) -> None:  # type: ignore[override]
        print(f"[WebSocket] Connection closed: code={code}, msg={msg}")
        self._user_callback.on_session_stopped()
        self._recognizer._notify_closed()

    def on_event(self, message: Dict[str, Any]) -> None:  # type: ignore[override]
        if not isinstance(message, dict):
            return
        event_type = message.get("type")
        if event_type == "session.created":
            self._handle_session_created(message)
        elif event_type == "session.updated":
            self._handle_session_updated(message)
        elif event_type == "conversation.item.input_audio_transcription.text":
            self._handle_transcription_text(message)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._handle_transcription_completed(message)
        elif event_type == "conversation.item.input_audio_transcription.failed":
            self._handle_transcription_failed(message)
        elif event_type == "response.done":
            self._recognizer._update_metrics(self._conversation)
        elif event_type == "error":
            self._handle_error(message)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _handle_session_created(self, message: Dict[str, Any]) -> None:
        session = message.get("session") or {}
        session_id = session.get("id")
        if session_id:
            self._recognizer._update_session_id(str(session_id))

    def _handle_session_updated(self, message: Dict[str, Any]) -> None:
        session = message.get("session") or {}
        session_id = session.get("id")
        if session_id:
            self._recognizer._update_session_id(str(session_id))

    def _handle_transcription_text(self, message: Dict[str, Any]) -> None:
        item_id = message.get("item_id")
        if not item_id:
            return
        fixed = message.get("text") or ""
        stash = message.get("stash") or ""
        combined = f"{fixed}{stash}"
        # print(f"Intermediate recognized text: {combined}")
        self._items[item_id] = {"fixed": fixed, "stash": stash}
        if not combined:
            return
        
        if combined.startswith("<asr_text>"):
            result = combined[len("<asr_text>") :]
        else:
            result = combined

        event = RecognitionEvent(text=result, is_final=False, raw=message)
        self._user_callback.on_result(event)

    def _handle_transcription_completed(self, message: Dict[str, Any]) -> None:
        item_id = message.get("item_id")
        transcript = message.get("transcript") or ""
        if not transcript and item_id and item_id in self._items:
            cache = self._items[item_id]
            transcript = f"{cache.get('fixed', '')}{cache.get('stash', '')}"
        if transcript:
            if transcript.startswith("<asr_text>"):
                result = transcript[len("<asr_text>") :]
            else:
                result = transcript

            event = RecognitionEvent(text=result, is_final=True, raw=message)
            self._user_callback.on_result(event)
        if item_id:
            self._items.pop(item_id, None)

    def _handle_transcription_failed(self, message: Dict[str, Any]) -> None:
        error = message.get("error") or {}
        detail = error.get("message") or "Recognition failed"
        code = error.get("code")
        if code:
            detail = f"{detail} (code={code})"
        self._user_callback.on_error(RuntimeError(detail))

    def _handle_error(self, message: Dict[str, Any]) -> None:
        error = message.get("error") or {}
        detail = error.get("message") or "Unknown error"
        code = error.get("code")
        if code:
            detail = f"{detail} (code={code})"
        event_id = error.get("event_id")
        if event_id:
            detail = f"{detail} [event_id={event_id}]"
        self._user_callback.on_error(RuntimeError(detail))


class QwenSpeechRecognizer(SpeechRecognizer):
    """Speech recognizer backed by the Qwen realtime ASR API."""

    def __init__(self, callback: SpeechRecognitionCallback, **recognition_kwargs: Any) -> None:
        self._lock = threading.Lock()
        self._conversation: Optional[OmniRealtimeConversation] = None
        self._adapter: Optional[_QwenOmniCallbackAdapter] = None
        self._callback: Optional[SpeechRecognitionCallback] = None
        self._session_id: Optional[str] = None
        self._last_response_id: Optional[str] = None
        self._last_first_text_delay: Optional[int] = None
        self._last_first_audio_delay: Optional[int] = None
        self._paused: bool = False
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop_event = threading.Event()
        self._connection_closed: bool = False  # 标记连接是否已关闭
        self._should_run: bool = False  # 标记服务是否应该运行（用于自动重连）

        options = dict(recognition_kwargs)
        self._model = options.pop("model", "qwen3-asr-flash-realtime")
        self._url = options.pop("url", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
        self._enable_turn_detection: Optional[bool] = options.pop("enable_turn_detection", True)
        self._turn_detection_threshold: Optional[float] = options.pop("turn_detection_threshold", 0.2)
        self._turn_detection_silence_duration_ms: Optional[int] = options.pop(
            "turn_detection_silence_duration_ms",
            800,
        )
        self._input_audio_format = options.pop("input_audio_format", "pcm")
        self._sample_rate = options.pop("sample_rate", 16000)
        self._language = options.pop("language", None)
        self._corpus_text = options.pop("corpus_text", None)
        self._enable_input_audio_transcription = options.pop("enable_input_audio_transcription", True)
        self._transcription_params: Optional[TranscriptionParams] = options.pop("transcription_params", None)
        self._keepalive_interval = options.pop("keepalive_interval", 30)  # 心跳间隔（秒）
        self._conversation_kwargs = dict(options.pop("conversation_kwargs", {}))
        self._update_session_overrides = dict(options.pop("update_session_kwargs", {}))
        if options:
            self._update_session_overrides.update(options)
        if "model" not in self._conversation_kwargs:
            self._conversation_kwargs["model"] = self._model
        if "url" not in self._conversation_kwargs:
            self._conversation_kwargs["url"] = self._url

        self.set_callback(callback)

    # ------------------------------------------------------------------
    # SpeechRecognizer interface
    # ------------------------------------------------------------------
    def set_callback(self, callback: SpeechRecognitionCallback) -> None:
        if callback is None:
            raise ValueError("callback must not be None")
        with self._lock:
            if self._conversation is not None:
                raise RuntimeError("Callback already configured; create a new recognizer instance instead.")
            self._callback = callback

    def start(self) -> None:
        with self._lock:
            if self._conversation is not None:
                return
            self._should_run = True  # 标记服务应该运行
            self._connection_closed = False
            adapter = self._create_adapter()
            conversation = OmniRealtimeConversation(callback=adapter, **self._conversation_kwargs)
            adapter.attach_conversation(conversation)
            self._conversation = conversation
            self._adapter = adapter
            self._session_id = None
            self._last_response_id = None
            self._last_first_text_delay = None
            self._last_first_audio_delay = None
            self._paused = False

        conversation = self._conversation
        assert conversation is not None  # for type checkers
        try:
            conversation.connect()
            transcription_params = self._resolve_transcription_params()
            update_kwargs: Dict[str, Any] = {
                "output_modalities": [MultiModality.TEXT],
                "enable_input_audio_transcription": self._enable_input_audio_transcription,
                "transcription_params": transcription_params,
            }
            if self._enable_turn_detection is not None:
                update_kwargs["enable_turn_detection"] = self._enable_turn_detection
            if self._enable_turn_detection:
                update_kwargs.setdefault("turn_detection_type", "server_vad")
                if self._turn_detection_threshold is not None:
                    update_kwargs.setdefault("turn_detection_threshold", self._turn_detection_threshold)
                if self._turn_detection_silence_duration_ms is not None:
                    update_kwargs.setdefault(
                        "turn_detection_silence_duration_ms",
                        self._turn_detection_silence_duration_ms,
                    )
            else:
                update_kwargs.setdefault("turn_detection_type", None)
            update_kwargs.update(self._update_session_overrides)
            conversation.update_session(**update_kwargs)
            
            # 启动心跳线程
            self._start_keepalive()
            print("[WebSocket] Connection established successfully.")
        except Exception:
            self._teardown_conversation(close=True)
            raise

    def stop(self) -> None:
        # 标记服务不应该运行（禁用自动重连）
        with self._lock:
            self._should_run = False
        
        # 停止心跳线程
        self._stop_keepalive()
        
        conversation = self._teardown_conversation(close=False)
        if conversation is None:
            return
        if not self._enable_turn_detection:
            with suppress(Exception):
                conversation.commit()
        with suppress(Exception):
            conversation.close()
        with self._lock:
            self._paused = False

    def send_audio_frame(self, data: bytes) -> None:
        if not data:
            return
        
        # 检查是否需要重连
        should_reconnect = False
        with self._lock:
            if self._paused:
                return
            # 检查连接是否已关闭且应该运行
            if self._connection_closed and self._should_run:
                should_reconnect = True
                self._connection_closed = False  # 重置标志，准备重连
        
        # 如果需要重连，在锁外执行重连
        if should_reconnect:
            try:
                self._reconnect()
            except Exception as e:
                print(f"[WebSocket] Reconnection failed: {e}")
                with self._lock:
                    self._connection_closed = True  # 重连失败，重新标记
                return
        
        conversation = self._require_conversation()
        audio_b64 = base64.b64encode(data).decode("ascii")
        try:
            conversation.append_audio(audio_b64)
        except Exception as e:
            print(f"[WebSocket] Error sending audio: {e}")
            # 标记连接已关闭，下次发送时会尝试重连
            with self._lock:
                self._connection_closed = True
            raise

    def pause(self) -> None:
        conversation: Optional[OmniRealtimeConversation] = None
        with self._lock:
            if self._paused:
                return
            self._paused = True
            conversation = self._conversation
        if conversation is not None:
            # VAD和手动commit不能同时使用
            if self._enable_turn_detection:
                # 启用VAD时，发送静音音频触发断句
                # 静音时长应比VAD的静音检测时长稍长，确保触发断句
                with suppress(Exception):
                    silence_duration_ms = self._turn_detection_silence_duration_ms or 800
                    # 多加200ms确保触发
                    silence_duration_ms += 200
                    sample_rate = self._sample_rate or 16000
                    # 计算需要的静音帧数
                    silence_frames = int(sample_rate * silence_duration_ms / 1000)
                    # 生成静音数据（16位PCM，单声道）
                    silence_data = b'\x00' * (silence_frames * 2)
                    audio_b64 = base64.b64encode(silence_data).decode("ascii")
                    conversation.append_audio(audio_b64)
            else:
                # 禁用VAD时，手动调用commit触发断句
                with suppress(Exception):
                    conversation.commit()

    def resume(self) -> None:
        should_reconnect = False
        with self._lock:
            if not self._paused:
                return  # 已经在运行
            self._paused = False
            # 检查连接是否已关闭
            if self._connection_closed and self._should_run:
                should_reconnect = True
                self._connection_closed = False
        
        # 如果连接已关闭，尝试重连
        if should_reconnect:
            try:
                print("[WebSocket] Connection was closed during pause, reconnecting...")
                self._reconnect()
            except Exception as e:
                print(f"[WebSocket] Reconnection on resume failed: {e}")
                with self._lock:
                    self._connection_closed = True

    def get_last_request_id(self) -> Optional[str]:
        with self._lock:
            return self._last_response_id or self._session_id

    def get_first_package_delay(self) -> Optional[int]:
        with self._lock:
            return self._last_first_text_delay

    def get_last_package_delay(self) -> Optional[int]:
        with self._lock:
            return self._last_first_audio_delay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_adapter(self) -> _QwenOmniCallbackAdapter:
        if self._callback is None:
            raise RuntimeError("Callback not configured; call set_callback first.")
        return _QwenOmniCallbackAdapter(self, self._callback)

    def _require_conversation(self) -> OmniRealtimeConversation:
        with self._lock:
            conversation = self._conversation
        if conversation is None:
            raise RuntimeError("Speech recognizer not started; call start() first.")
        return conversation

    def _resolve_transcription_params(self) -> TranscriptionParams:
        if self._transcription_params is not None:
            return self._transcription_params
        params: Dict[str, Any] = {
            "sample_rate": self._sample_rate,
            "input_audio_format": self._input_audio_format,
        }
        if self._language:
            params["language"] = self._language
        if self._corpus_text:
            params["corpus_text"] = self._corpus_text
        return TranscriptionParams(**params)

    def _teardown_conversation(self, *, close: bool) -> Optional[OmniRealtimeConversation]:
        # 先停止心跳
        self._stop_keepalive()
        
        with self._lock:
            conversation = self._conversation
            if conversation is None:
                return None
            adapter = self._adapter
            if adapter is not None:
                adapter.detach()
            self._conversation = None
            self._adapter = None
            self._paused = False
        if close:
            with suppress(Exception):
                conversation.close()
            return None
        return conversation

    def _update_metrics(self, conversation: Optional[OmniRealtimeConversation]) -> None:
        conv = conversation or self._conversation
        if conv is None:
            return
        response_id: Optional[str] = None
        first_text_delay: Optional[int] = None
        first_audio_delay: Optional[int] = None
        with suppress(Exception):
            response_id = conv.get_last_response_id()
        with suppress(Exception):
            first_text_delay = conv.get_last_first_text_delay()
        with suppress(Exception):
            first_audio_delay = conv.get_last_first_audio_delay()
        with self._lock:
            if response_id is not None:
                self._last_response_id = response_id
            if first_text_delay is not None:
                self._last_first_text_delay = first_text_delay
            if first_audio_delay is not None:
                self._last_first_audio_delay = first_audio_delay

    def _update_session_id(self, session_id: str) -> None:
        with self._lock:
            self._session_id = session_id

    def _notify_closed(self) -> None:
        with self._lock:
            self._connection_closed = True  # 标记连接已关闭
            self._conversation = None
            self._adapter = None
            # 注意：不要重置 _paused，因为可能需要保持暂停状态
            # 也不要重置 _should_run，因为可能需要自动重连

    def _reconnect(self) -> None:
        """重新建立WebSocket连接"""
        print("[WebSocket] Starting reconnection...")
        
        # 清理旧连接
        self._teardown_conversation(close=True)
        
        with self._lock:
            if not self._should_run:
                print("[WebSocket] Service is stopped, cancelling reconnection.")
                return
            
            self._connection_closed = False
            adapter = self._create_adapter()
            conversation = OmniRealtimeConversation(callback=adapter, **self._conversation_kwargs)
            adapter.attach_conversation(conversation)
            self._conversation = conversation
            self._adapter = adapter
            # 保留之前的暂停状态
            paused = self._paused

        conversation = self._conversation
        assert conversation is not None
        
        try:
            conversation.connect()
            transcription_params = self._resolve_transcription_params()
            update_kwargs: Dict[str, Any] = {
                "output_modalities": [MultiModality.TEXT],
                "enable_input_audio_transcription": self._enable_input_audio_transcription,
                "transcription_params": transcription_params,
            }
            if self._enable_turn_detection is not None:
                update_kwargs["enable_turn_detection"] = self._enable_turn_detection
            if self._enable_turn_detection:
                update_kwargs.setdefault("turn_detection_type", "server_vad")
                if self._turn_detection_threshold is not None:
                    update_kwargs.setdefault("turn_detection_threshold", self._turn_detection_threshold)
                if self._turn_detection_silence_duration_ms is not None:
                    update_kwargs.setdefault(
                        "turn_detection_silence_duration_ms",
                        self._turn_detection_silence_duration_ms,
                    )
            else:
                update_kwargs.setdefault("turn_detection_type", None)
            update_kwargs.update(self._update_session_overrides)
            conversation.update_session(**update_kwargs)
            
            # 重新启动心跳线程
            self._start_keepalive()
            print("[WebSocket] Reconnection successful!")
        except Exception as e:
            print(f"[WebSocket] Reconnection failed: {e}")
            self._teardown_conversation(close=True)
            raise

    def _start_keepalive(self) -> None:
        """启动心跳线程以保持WebSocket连接活跃"""
        with self._lock:
            if self._keepalive_thread is not None:
                return  # 已经运行
            self._keepalive_stop_event.clear()
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_worker,
                daemon=True,
                name="QwenKeepalive"
            )
            self._keepalive_thread.start()

    def _stop_keepalive(self) -> None:
        """停止心跳线程"""
        with self._lock:
            if self._keepalive_thread is None:
                return
            self._keepalive_stop_event.set()
            thread = self._keepalive_thread
            self._keepalive_thread = None
        
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _keepalive_worker(self) -> None:
        """心跳工作线程：定期发送极短的静音音频以保持连接活跃"""
        while not self._keepalive_stop_event.wait(timeout=self._keepalive_interval):
            try:
                with self._lock:
                    conversation = self._conversation
                    paused = self._paused
                
                # 只在会话存在且处于暂停状态时发送心跳
                # 如果正在活跃发送音频，不需要额外的心跳
                if conversation is not None and paused:
                    # 发送一个极短的静音音频帧作为心跳
                    # 使用100ms的静音（远小于VAD检测时长）
                    sample_rate = self._sample_rate or 16000
                    keepalive_frames = int(sample_rate * 0.1)  # 100ms
                    silence_data = b'\x00' * (keepalive_frames * 2)  # 16-bit PCM
                    audio_b64 = base64.b64encode(silence_data).decode("ascii")
                    
                    with suppress(Exception):
                        conversation.append_audio(audio_b64)
                        # print("[Keepalive] Sent heartbeat audio frame.")
            except Exception:
                # 忽略心跳过程中的错误，继续下一次心跳
                pass
