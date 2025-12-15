"""Soniox Speech Recognizer - 使用原生 WebSocket 实现的语音识别器"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import suppress
from typing import Any, Dict, List, Optional

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
        
        # 配置参数
        self._api_key = api_key or os.environ.get("SONIOX_API_KEY", "")
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
            if self._ws is not None:
                return
            self._should_run = True
            self._paused = False
            self._final_tokens = []
            self._current_text = ""
        
        self._connect()

    def _connect(self) -> None:
        """建立 WebSocket 连接并发送配置"""
        try:
            print("[Soniox] Connecting to Soniox...")
            self._ws = ws_connect(SONIOX_WEBSOCKET_URL)
            
            # 构建配置消息
            config = self._build_config()
            self._ws.send(json.dumps(config))
            
            # 启动接收线程
            self._recv_stop_event.clear()
            self._recv_thread = threading.Thread(
                target=self._recv_worker,
                daemon=True,
                name="SonioxRecvThread"
            )
            self._recv_thread.start()
            
            with self._lock:
                self._connected = True
            
            print("[Soniox] Connection established successfully.")
            
            if self._callback:
                self._callback.on_session_started()
                
        except Exception as e:
            print(f"[Soniox] Connection failed: {e}")
            self._cleanup()
            raise

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
        
        if self._context:
            config["context"] = self._context
        
        # 合并额外参数
        config.update(self._extra_kwargs)
        
        return config

    def _recv_worker(self) -> None:
        """接收线程：从 WebSocket 读取消息并处理"""
        try:
            while not self._recv_stop_event.is_set():
                if self._ws is None:
                    break
                
                try:
                    message = self._ws.recv(timeout=1.0)
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
            print(f"[Soniox] Receive thread error: {e}")
            if self._callback:
                self._callback.on_error(e)
        finally:
            with self._lock:
                self._connected = False
            if self._callback:
                self._callback.on_session_stopped()

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
        
        # 检查是否有 <end> token（endpoint detection）
        has_endpoint = any(t.get("text") == "<end>" for t in new_final_tokens)
        
        # 构建当前文本
        final_text = "".join(t.get("text", "") for t in self._final_tokens if t.get("text") != "<end>")
        non_final_text = "".join(t.get("text", "") for t in non_final_tokens)
        
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
        
        # 关闭 WebSocket
        if self._ws:
            with suppress(Exception):
                # 发送空字符串表示结束
                self._ws.send("")
            with suppress(Exception):
                self._ws.close()
            self._ws = None
        
        with self._lock:
            self._connected = False

    def send_audio_frame(self, data: bytes) -> None:
        if not data:
            return
        
        with self._lock:
            if self._paused:
                return
            if not self._connected or self._ws is None:
                return
        
        try:
            # Soniox 接收原始 PCM 字节数据
            self._ws.send(data)
        except Exception as e:
            print(f"[Soniox] Error sending audio: {e}")
            with self._lock:
                self._connected = False

    def pause(self) -> None:
        with self._lock:
            if self._paused:
                return
            self._paused = True
        
        # 发送 finalize 消息强制结束当前句子
        if self._ws and self._connected:
            try:
                finalize_msg = json.dumps({"type": "finalize"})
                self._ws.send(finalize_msg)
            except Exception as e:
                print(f"[Soniox] Error sending finalize: {e}")

    def resume(self) -> None:
        with self._lock:
            if not self._paused:
                return
            self._paused = False

    def get_last_request_id(self) -> Optional[str]:
        with self._lock:
            return self._session_id

    def get_first_package_delay(self) -> Optional[int]:
        # Soniox API 不提供此信息
        return None

    def get_last_package_delay(self) -> Optional[int]:
        # Soniox API 不提供此信息
        return None
