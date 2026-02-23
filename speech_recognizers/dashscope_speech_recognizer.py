from __future__ import annotations

from contextlib import suppress
from typing import Any, Optional

from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

from .base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
    SpeechRecognizer,
)


class _DashscopeCallbackAdapter(RecognitionCallback):
    """Adapter that normalizes DashScope events into generic recognition events."""

    def __init__(self, user_callback: SpeechRecognitionCallback) -> None:
        self._user_callback = user_callback

    def on_open(self) -> None:
        self._user_callback.on_session_started()

    def on_close(self) -> None:
        self._user_callback.on_session_stopped()

    def on_complete(self) -> None:
        self._user_callback.on_session_stopped()

    def on_error(self, message) -> None:  # type: ignore[override]
        description = getattr(message, "message", str(message))
        request_id = getattr(message, "request_id", None)
        error_message = description if request_id is None else f"{description} (request_id={request_id})"
        self._user_callback.on_error(RuntimeError(error_message))

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if not sentence:
            return

        text = sentence.get("text")
        if not text:
            return

        is_final = RecognitionResult.is_sentence_end(sentence)
        confidence = sentence.get("confidence")

        event = RecognitionEvent(
            text=text,
            is_final=is_final,
            confidence=confidence,
            raw=sentence,
        )
        self._user_callback.on_result(event)


class DashscopeSpeechRecognizer(SpeechRecognizer):
    """DashScope-backed implementation of the speech recognizer interface."""

    def __init__(self, callback: SpeechRecognitionCallback, **recognition_kwargs: Any) -> None:
        self._recognition_kwargs = recognition_kwargs
        self._recognition: Optional[Recognition] = None
        self._adapter: Optional[_DashscopeCallbackAdapter] = None
        self._callback: Optional[SpeechRecognitionCallback] = None
        self.set_callback(callback)

    def set_callback(self, callback: SpeechRecognitionCallback) -> None:
        if callback is None:
            raise ValueError("callback must not be None")

        if self._recognition is not None:
            raise RuntimeError("Callback already configured; create a new recognizer instance instead.")

        self._callback = callback
        self._adapter = _DashscopeCallbackAdapter(callback)
        self._recognition = Recognition(callback=self._adapter, **self._recognition_kwargs)

    def _require_recognition(self) -> Recognition:
        if self._recognition is None:
            raise RuntimeError("Speech recognizer not initialized; call set_callback first.")
        return self._recognition

    def start(self) -> None:
        self._require_recognition().start()

    def stop(self) -> None:
        self._require_recognition().stop()

    def send_audio_frame(self, data: bytes) -> None:
        self._require_recognition().send_audio_frame(data)

    def pause(self) -> None:
        recognition = self._require_recognition()

        # 在暂停时主动发送少量静音帧，避免无音频直接 stop 触发后端报错。
        # 这部分原先在 main.py 中处理，现在下沉到具体后端实现。
        sample_rate = int(self._recognition_kwargs.get('sample_rate', 16000) or 16000)
        channels = int(self._recognition_kwargs.get('channels', 1) or 1)
        bytes_per_sample = 2  # int16
        silence_ms = 100
        silence_frames = max(1, int(sample_rate * silence_ms / 1000))
        silence_data = b'\x00' * (silence_frames * channels * bytes_per_sample)

        with suppress(Exception):
            recognition.send_audio_frame(silence_data)

        with suppress(Exception):
            recognition.stop()

    def resume(self) -> None:
        self.start()

    def get_last_request_id(self) -> Optional[str]:
        return self._require_recognition().get_last_request_id()

    def get_first_package_delay(self) -> Optional[int]:
        return self._require_recognition().get_first_package_delay()

    def get_last_package_delay(self) -> Optional[int]:
        return self._require_recognition().get_last_package_delay()
