from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

import config
from local_asr import get_engine_runtime_issues
from local_asr.model_manager import is_asr_cached, is_asr_models_ready, is_silero_cached
from local_asr.vad_processor import VADProcessor
from vrcx_context_bridge import build_asr_context_text

from .base_speech_recognizer import RecognitionEvent, SpeechRecognitionCallback, SpeechRecognizer

logger = logging.getLogger(__name__)

LOCAL_VAD_SAMPLE_RATE = 16000
LOCAL_VAD_CHUNK_SAMPLES = 512
LOCAL_VAD_CHUNK_DURATION = LOCAL_VAD_CHUNK_SAMPLES / LOCAL_VAD_SAMPLE_RATE


class LocalSpeechRecognizer(SpeechRecognizer):
    """Local VAD + on-device ASR wrapped as the existing recognizer interface."""

    def __init__(
        self,
        callback: SpeechRecognitionCallback,
        sample_rate: int = 16000,
        source_language: str = "auto",
        corpus_text: str | None = None,
    ) -> None:
        self._callback = callback
        self._sample_rate = sample_rate
        self._source_language = source_language
        self._engine_name = getattr(config, "LOCAL_ASR_ENGINE", "sensevoice")
        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=128)
        self._worker: threading.Thread | None = None
        self._asr_executor: ThreadPoolExecutor | None = None
        self._active_transcribe_future: Future | None = None
        self._waiting_partial_audio: np.ndarray | None = None
        self._waiting_final_audio: np.ndarray | None = None
        self._running = False
        self._paused = False
        self._lock = threading.RLock()
        self._engine = None
        self._vad = self._create_vad()
        self._pending_samples = np.array([], dtype=np.float32)
        self._last_partial_text = ""
        self._last_partial_time = 0.0
        self._last_request_id = f"local-{self._engine_name}"
        self._stream_id = 0
        self._corpus_text = (corpus_text or "").strip()

    def _input_cap_samples(self) -> int:
        sec = float(getattr(config, "LOCAL_VAD_MAX_SPEECH_DURATION", 30.0))
        sec = max(1.0, sec)
        return int(sec * LOCAL_VAD_SAMPLE_RATE)

    def _create_vad(self) -> VADProcessor:
        vad = VADProcessor(
            sample_rate=LOCAL_VAD_SAMPLE_RATE,
            threshold=float(getattr(config, "LOCAL_VAD_THRESHOLD", 0.50)),
            min_speech_duration=float(getattr(config, "LOCAL_VAD_MIN_SPEECH_DURATION", 1.0)),
            chunk_duration=LOCAL_VAD_CHUNK_DURATION,
            pre_speech_duration=float(getattr(config, "LOCAL_VAD_PRE_SPEECH_DURATION", 0.2)),
        )
        vad.update_settings(
            {
                "vad_mode": getattr(config, "LOCAL_VAD_MODE", "silero"),
                "vad_threshold": float(getattr(config, "LOCAL_VAD_THRESHOLD", 0.50)),
                "min_speech_duration": float(getattr(config, "LOCAL_VAD_MIN_SPEECH_DURATION", 1.0)),
                "silence_duration": float(getattr(config, "LOCAL_VAD_SILENCE_DURATION", 0.8)),
                "pre_speech_duration": float(getattr(config, "LOCAL_VAD_PRE_SPEECH_DURATION", 0.2)),
            }
        )
        return vad

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        if not is_asr_cached(self._engine_name):
            if not is_silero_cached():
                raise RuntimeError(
                    "本地 VAD（Silero ONNX）未就绪。请在「本地音频识别」中点击下载，或检查 local_asr/models 下是否有 silero_vad。"
                )
            if is_asr_models_ready(self._engine_name) and get_engine_runtime_issues(
                self._engine_name
            ):
                missing = ", ".join(get_engine_runtime_issues(self._engine_name))
                raise RuntimeError(
                    f"模型文件已在本地，但缺少 Python 依赖: {missing}。请安装: pip install -r requirements-local-asr.txt"
                )
            raise RuntimeError(
                f"本地识别主模型未就绪。请在「本地音频识别」中点击下载 {self._engine_name} 所需资源。"
            )

        if self._engine_name == "sensevoice":
            from local_asr.asr_sensevoice import SenseVoiceEngine

            engine = SenseVoiceEngine()
        elif self._engine_name == "qwen3-asr":
            from local_asr.asr_qwen3 import Qwen3ASREngine

            engine = Qwen3ASREngine(corpus_text=build_asr_context_text(self._corpus_text) or None)
        else:
            raise RuntimeError(f"未知的本地识别引擎: {self._engine_name}")

        engine.set_language(self._source_language or "auto")
        self._engine = engine
        return engine

    def _emit_result(self, text: str, is_final: bool, raw: Optional[dict] = None) -> None:
        if not text:
            return
        self._callback.on_result(
            RecognitionEvent(
                text=text,
                is_final=is_final,
                raw=raw,
            )
        )

    def _transcribe(self, audio: np.ndarray, *, is_final: bool = True) -> tuple[str, dict] | None:
        engine = self._ensure_engine()
        if hasattr(engine, "set_corpus_text"):
            engine.set_corpus_text(build_asr_context_text(self._corpus_text) or None)
        kwargs = {}
        if hasattr(engine, "transcribe") and "update_context" in engine.transcribe.__code__.co_varnames:
            kwargs["update_context"] = is_final
        result = engine.transcribe(audio, **kwargs)
        if not result:
            return None
        text = (result.get("text") or "").strip()
        if not text:
            return None
        return text, result

    def _on_transcription_done(self, future: Future, *, stream_id: int, is_final: bool) -> None:
        try:
            payload = future.result()
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("Local ASR transcription failed")
            self._callback.on_error(exc)
            payload = None

        if payload is not None:
            text, raw = payload
            with self._lock:
                if is_final:
                    self._last_partial_text = ""
                    self._stream_id += 1
                    self._emit_result(text, is_final=True, raw=raw)
                elif stream_id == self._stream_id and text != self._last_partial_text:
                    self._last_partial_text = text
                    self._emit_result(text, is_final=False, raw=raw)

        with self._lock:
            self._try_start_transcribe_locked()

    def _try_start_transcribe_locked(self) -> None:
        if self._asr_executor is None:
            return
        fut = self._active_transcribe_future
        if fut is not None and not fut.done():
            return

        if self._waiting_final_audio is not None:
            audio = self._waiting_final_audio
            self._waiting_final_audio = None
            is_final = True
        elif self._waiting_partial_audio is not None:
            audio = self._waiting_partial_audio
            self._waiting_partial_audio = None
            is_final = False
        else:
            self._active_transcribe_future = None
            return

        stream_id = self._stream_id
        self._active_transcribe_future = self._asr_executor.submit(self._transcribe, audio, is_final=is_final)
        self._active_transcribe_future.add_done_callback(
            lambda done_future, _sid=stream_id, _fin=is_final: self._on_transcription_done(
                done_future,
                stream_id=_sid,
                is_final=_fin,
            )
        )

    def _enqueue_transcribe(self, audio: np.ndarray, *, is_final: bool) -> None:
        if audio.size == 0:
            return
        copy = audio.copy()
        with self._lock:
            if self._asr_executor is None:
                self._asr_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="yakutan-local-asr",
                )
            if is_final:
                self._waiting_final_audio = copy
            else:
                self._waiting_partial_audio = copy
            self._try_start_transcribe_locked()

    def _maybe_emit_partial(self) -> None:
        if not getattr(config, "LOCAL_INCREMENTAL_ASR", True):
            return
        if self._audio_queue.qsize() >= 8:
            return
        peek = self._vad.peek_buffer()
        if peek is None:
            return
        audio, duration = peek
        if duration < 1.5:
            return
        now = time.monotonic()
        if now - self._last_partial_time < float(getattr(config, "LOCAL_INTERIM_INTERVAL", 2.0)):
            return
        self._last_partial_time = time.monotonic()
        self._enqueue_transcribe(audio, is_final=False)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        if self._vad._is_speaking and self._vad._speech_samples >= self._input_cap_samples():
            chunk = np.zeros_like(chunk)
        speech_segment = self._vad.process_chunk(chunk)
        if speech_segment is not None:
            self._enqueue_transcribe(speech_segment, is_final=True)
            return
        if self._vad._is_speaking:
            self._maybe_emit_partial()

    def _feed_samples(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        if self._pending_samples.size:
            samples = np.concatenate([self._pending_samples, samples])
        chunk_size = LOCAL_VAD_CHUNK_SAMPLES
        offset = 0
        while offset + chunk_size <= len(samples):
            chunk = samples[offset : offset + chunk_size]
            self._process_chunk(chunk)
            offset += chunk_size
        self._pending_samples = samples[offset:].copy()

    def _drain_queue_locked(self) -> None:
        buffered: list[np.ndarray] = []
        while True:
            try:
                data = self._audio_queue.get_nowait()
            except queue.Empty:
                break
            buffered.append(self._pcm_to_float32(data))
        if buffered:
            self._feed_samples(np.concatenate(buffered))

    def _finalize_current_segment_locked(self) -> None:
        if self._pending_samples.size:
            padded = np.pad(
                self._pending_samples,
                (0, LOCAL_VAD_CHUNK_SAMPLES - len(self._pending_samples)),
            )
            self._process_chunk(padded)
            self._pending_samples = np.array([], dtype=np.float32)
        segment = self._vad.force_flush() if self._vad._is_speaking else self._vad.flush()
        if segment is not None:
            self._enqueue_transcribe(segment, is_final=True)
        self._last_partial_text = ""
        self._last_partial_time = 0.0

    @staticmethod
    def _pcm_to_float32(data: bytes) -> np.ndarray:
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        return audio / 32768.0

    def _worker_loop(self) -> None:
        try:
            while self._running:
                try:
                    data = self._audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._paused:
                    continue
                samples = self._pcm_to_float32(data)
                with self._lock:
                    self._feed_samples(samples)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("Local ASR worker failed")
            self._callback.on_error(exc)

    def set_callback(self, callback: SpeechRecognitionCallback) -> None:
        self._callback = callback

    def start(self) -> None:
        with self._lock:
            self._ensure_engine()
            self._paused = False
            if self._running:
                return
            self._running = True
            self._stream_id = 0
            self._last_partial_text = ""
            self._waiting_partial_audio = None
            self._waiting_final_audio = None
            self._active_transcribe_future = None
            if self._asr_executor is None:
                self._asr_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="yakutan-local-asr",
                )
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
            self._callback.on_session_started()

    def stop(self) -> None:
        with self._lock:
            self._drain_queue_locked()
            self._finalize_current_segment_locked()
            self._running = False
            self._paused = False
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None
        if self._asr_executor is not None:
            self._asr_executor.shutdown(wait=True)
            self._asr_executor = None
        self._active_transcribe_future = None
        self._waiting_partial_audio = None
        self._waiting_final_audio = None
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.unload()
                except Exception:
                    pass
                self._engine = None
            self._callback.on_session_stopped()

    def send_audio_frame(self, data: bytes) -> None:
        if not self._running or self._paused or not data:
            return
        try:
            self._audio_queue.put_nowait(data)
        except queue.Full:
            try:
                _ = self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            self._audio_queue.put_nowait(data)

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._drain_queue_locked()
            self._finalize_current_segment_locked()

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def get_last_request_id(self) -> Optional[str]:
        return self._last_request_id

    def get_first_package_delay(self) -> Optional[int]:
        return None

    def get_last_package_delay(self) -> Optional[int]:
        return None
