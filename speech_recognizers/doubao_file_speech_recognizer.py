from __future__ import annotations

import base64
import io
import json
import threading
import urllib.error
import urllib.request
import uuid
import wave
from typing import Any, Dict, Optional

from .base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
    SpeechRecognizer,
)


class DoubaoFileSpeechRecognizer(SpeechRecognizer):
    """豆包录音文件识别（极速版）伪实时实现。

    行为说明：
    - start/resume: 开始录制（缓存）音频帧
    - pause: 结束本段录制并调用录音文件识别接口
    - 识别完成后仅回调一条 is_final=True 的结果
    """

    def __init__(
        self,
        callback: SpeechRecognitionCallback,
        api_key: Optional[str] = None,
        api_app_key: Optional[str] = None,
        api_access_key: Optional[str] = None,
        uid: Optional[str] = None,
        resource_id: str = "volc.seedasr.auc",
        url: str = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
        model_name: str = "bigmodel",
        sample_rate: int = 16000,
        channels: int = 1,
        bits: int = 16,
        timeout_seconds: int = 60,
        min_audio_bytes: int = 3200,
        max_buffer_seconds: int = 60,
        request_options: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        resolved_api_key = str(api_key or "").strip()
        resolved_app_key = str(api_app_key or "").strip()
        resolved_access_key = str(api_access_key or "").strip()
        if not resolved_api_key and (not resolved_app_key or not resolved_access_key):
            raise ValueError("豆包录音文件识别需要 x-api-key 或 api_app_key+api_access_key")

        self._lock = threading.Lock()
        self._callback: Optional[SpeechRecognitionCallback] = None
        self._running = False
        self._recording = False
        self._audio_buffer = bytearray()
        self._last_request_id: Optional[str] = None
        self._frames_in_segment = 0
        self._segment_index = 0

        self._api_key = resolved_api_key
        self._api_app_key = resolved_app_key
        self._api_access_key = resolved_access_key
        self._uid = str(uid or "").strip() or self._api_app_key or "doubao-file-asr"
        self._resource_id = str(resource_id).strip() or "volc.seedasr.auc"
        self._url = str(url).strip()
        self._model_name = str(model_name).strip() or "bigmodel"
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._bits = int(bits)
        self._timeout_seconds = int(timeout_seconds)
        self._min_audio_bytes = int(min_audio_bytes)
        self._max_buffer_seconds = max(1, int(max_buffer_seconds))
        bytes_per_second = self._sample_rate * self._channels * max(1, self._bits // 8)
        self._max_audio_buffer_bytes = max(bytes_per_second, bytes_per_second * self._max_buffer_seconds)
        self._request_options = dict(request_options or {})

        self.set_callback(callback)

    def set_callback(self, callback: SpeechRecognitionCallback) -> None:
        if callback is None:
            raise ValueError("callback must not be None")
        with self._lock:
            self._callback = callback

    def start(self) -> None:
        callback: Optional[SpeechRecognitionCallback]
        with self._lock:
            if self._running:
                self._recording = True
                self._audio_buffer.clear()
                self._frames_in_segment = 0
                self._segment_index += 1
                return
            self._running = True
            self._recording = True
            self._audio_buffer.clear()
            self._frames_in_segment = 0
            self._segment_index = 1
            callback = self._callback
        if callback is not None:
            callback.on_session_started()

    def stop(self) -> None:
        callback: Optional[SpeechRecognitionCallback]
        audio_data = b""
        with self._lock:
            if not self._running:
                return
            self._running = False
            callback = self._callback
            if self._recording and self._audio_buffer:
                audio_data = bytes(self._audio_buffer)
            self._recording = False
            self._audio_buffer.clear()
            self._frames_in_segment = 0

        if audio_data and len(audio_data) >= self._min_audio_bytes:
            self._recognize_and_emit(audio_data)

        if callback is not None:
            callback.on_session_stopped()

    def send_audio_frame(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if not self._running or not self._recording:
                return
            self._audio_buffer.extend(data)
            if len(self._audio_buffer) > self._max_audio_buffer_bytes:
                overflow_bytes = len(self._audio_buffer) - self._max_audio_buffer_bytes
                del self._audio_buffer[:overflow_bytes]
            self._frames_in_segment += 1

    def pause(self) -> None:
        audio_data = b""
        with self._lock:
            if not self._running or not self._recording:
                return
            if self._audio_buffer:
                audio_data = bytes(self._audio_buffer)
            self._recording = False
            self._audio_buffer.clear()
            self._frames_in_segment = 0

        if not audio_data or len(audio_data) < self._min_audio_bytes:
            return

        self._recognize_and_emit(audio_data)

    def resume(self) -> None:
        with self._lock:
            if not self._running:
                return
            if self._recording:
                return
            self._recording = True
            self._audio_buffer.clear()
            self._frames_in_segment = 0
            self._segment_index += 1

    def get_last_request_id(self) -> Optional[str]:
        with self._lock:
            return self._last_request_id

    def get_first_package_delay(self) -> Optional[int]:
        return None

    def get_last_package_delay(self) -> Optional[int]:
        return None

    def _recognize_and_emit(self, pcm_bytes: bytes) -> None:
        callback: Optional[SpeechRecognitionCallback]
        with self._lock:
            callback = self._callback

        if callback is None:
            return

        try:
            response_payload = self._recognize_once(pcm_bytes)
            text = (response_payload.get("result") or {}).get("text") or ""
            text = str(text).strip()
            if not text:
                return
            event = RecognitionEvent(text=text, is_final=True, raw=response_payload)
            callback.on_result(event)
        except Exception as e:
            callback.on_error(e)

    def _recognize_once(self, pcm_bytes: bytes) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        with self._lock:
            self._last_request_id = request_id

        wav_bytes = self._pcm_to_wav(pcm_bytes)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        payload: Dict[str, Any] = {
            "user": {"uid": self._uid},
            "audio": {
                "data": audio_b64,
                "format": "wav",
                "codec": "raw",
                "rate": self._sample_rate,
                "bits": self._bits,
                "channel": self._channels,
            },
            "request": {
                "model_name": self._model_name,
                "enable_itn": True,
                "enable_punc": True,
            },
        }
        if self._request_options:
            payload["request"].update(self._request_options)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Api-Resource-Id": self._resource_id,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        else:
            headers["X-Api-App-Key"] = self._api_app_key
            headers["X-Api-Access-Key"] = self._api_access_key

        request = urllib.request.Request(
            self._url,
            data=body,
            headers=headers,
            method="POST",
        )

        response_headers: Dict[str, str] = {}
        raw_text = ""
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                response_headers = {k: v for k, v in response.headers.items()}
                raw_text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"豆包识别请求失败: HTTP {e.code}, body={err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"豆包识别请求失败: {e}") from e

        status_code = response_headers.get("X-Api-Status-Code", "")
        status_message = response_headers.get("X-Api-Message", "")
        logid = response_headers.get("X-Tt-Logid", "")

        response_payload: Dict[str, Any] = {}
        if raw_text:
            try:
                response_payload = json.loads(raw_text)
            except json.JSONDecodeError:
                response_payload = {"raw_text": raw_text}

        if status_code and status_code != "20000000":
            raise RuntimeError(
                f"豆包识别失败: code={status_code}, message={status_message}, logid={logid}"
            )

        response_payload.setdefault("_meta", {})
        response_payload["_meta"].update(
            {
                "status_code": status_code,
                "status_message": status_message,
                "logid": logid,
                "request_id": request_id,
            }
        )
        return response_payload

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(self._channels)
                wav_file.setsampwidth(max(1, self._bits // 8))
                wav_file.setframerate(self._sample_rate)
                wav_file.writeframes(pcm_bytes)
            return wav_buffer.getvalue()
