from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

import config
from local_inference import get_engine_runtime_issues
from local_inference.model_manager import is_asr_cached, is_asr_models_ready, is_silero_cached
from local_inference.vad_processor import VADProcessor
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
        self._engine_name = getattr(config, "LOCAL_INFERENCE_ENGINE", "sensevoice")
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
        self._silence_trigger_armed = True
        # 有声内容计数：每处理到一个有声 VAD 分块 +1。
        # 最近一次"已完成"中间结果入队时的快照值若与当前一致，
        # 说明该快照之后缓冲里只追加了静音——中间结果已覆盖整句。
        self._voiced_chunk_seq = 0
        self._last_partial_voiced_seq = -1
        self._partial_pending_voiced_seq = -1
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
            pre_speech_duration=float(getattr(config, "VAD_PRE_SPEECH_DURATION", 0.5)),
        )
        vad_mode = (
            getattr(config, "LOCAL_VAD_MODE", "silero")
            if getattr(config, "VAD_ENABLED", True)
            else "disabled"
        )
        vad.update_settings(
            {
                "vad_mode": vad_mode,
                "vad_threshold": float(getattr(config, "LOCAL_VAD_THRESHOLD", 0.50)),
                "min_speech_duration": float(getattr(config, "LOCAL_VAD_MIN_SPEECH_DURATION", 1.0)),
                "silence_duration": config.clamp_vad_silence_duration(
                    float(getattr(config, "LOCAL_VAD_SILENCE_DURATION", 0.8))
                ),
                "pre_speech_duration": float(getattr(config, "VAD_PRE_SPEECH_DURATION", 0.5)),
            }
        )
        return vad

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        if not is_asr_cached(self._engine_name):
            if not is_silero_cached():
                raise RuntimeError(
                    "本地 VAD（Silero ONNX）未就绪。请在「本地音频识别」中点击下载，或检查 local_inference/models 下是否有 silero_vad。"
                )
            if is_asr_models_ready(self._engine_name) and get_engine_runtime_issues(
                self._engine_name
            ):
                missing = ", ".join(get_engine_runtime_issues(self._engine_name))
                raise RuntimeError(
                    f"模型文件已在本地，但缺少 Python 依赖: {missing}。请安装: pip install -r requirements-local-inference.txt"
                )
            raise RuntimeError(
                f"本地识别主模型未就绪。请在「本地音频识别」中点击下载 {self._engine_name} 所需资源。"
            )

        if self._engine_name == "sensevoice":
            from local_inference.asr_sensevoice import SenseVoiceEngine

            engine = SenseVoiceEngine()
        elif self._engine_name == "qwen3-asr":
            from local_inference.asr_qwen3 import Qwen3ASREngine

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
        recognition_started = time.monotonic()
        result = engine.transcribe(audio, **kwargs)
        # 临时：识别完成后输出一行日志，记录本次识别耗时（毫秒）
        print(
            f"[本地ASR] 识别完成（{'最终' if is_final else '中间'}）: "
            f"耗时 {(time.monotonic() - recognition_started) * 1000.0:.0f}ms"
        )
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
                    self._emit_final_result(text, raw)
                elif stream_id == self._stream_id:
                    # 中间结果完成：记录其快照对应的有声计数，供整句复用时比对。
                    self._last_partial_voiced_seq = self._partial_pending_voiced_seq
                    if text != self._last_partial_text:
                        self._last_partial_text = text
                        self._emit_result(text, is_final=False, raw=raw)
        else:
            with self._lock:
                if is_final or stream_id == self._stream_id:
                    # 空结果/失败：作废"可复用"资格，避免用上更早的旧文本。
                    self._last_partial_voiced_seq = -1

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
        # 临时：本地 ASR 触发时输出一行日志
        print(
            f"[本地ASR] 触发识别（{'最终' if is_final else '中间'}）: "
            f"音频时长 {audio.size / LOCAL_VAD_SAMPLE_RATE:.2f}s"
        )
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
                    thread_name_prefix="yakutan-local-inference",
                )
            if is_final:
                self._waiting_final_audio = copy
            else:
                self._waiting_partial_audio = copy
                # 记录这份快照入队时的有声计数（完成时回写到 _last_partial_voiced_seq）。
                self._partial_pending_voiced_seq = self._voiced_chunk_seq
            self._try_start_transcribe_locked()

    def _maybe_emit_partial(self) -> None:
        if not getattr(config, "LOCAL_INCREMENTAL_ASR", True):
            return
        if self._audio_queue.qsize() >= 8:
            return
        vad = self._vad
        silence_sec = vad.current_silence_duration
        if silence_sec <= 0.0:
            # 正在说话（无停顿）：确保下一次停顿可以触发；继续走保底判定。
            self._silence_trigger_armed = True
        peek = vad.peek_buffer()
        if peek is None:
            return
        audio, duration = peek
        if duration < 1.0:
            return
        now = time.monotonic()
        elapsed = now - self._last_partial_time
        min_interval = max(0.0, float(
            getattr(config, "LOCAL_INCREMENTAL_MIN_UPDATE_INTERVAL", 3.0),
        ))
        fallback_interval = max(min_interval, float(
            getattr(config, "LOCAL_INCREMENTAL_MAX_UPDATE_INTERVAL", 4.0),
        ))
        trigger_silence = max(
            0.0, float(getattr(config, "LOCAL_INCREMENTAL_TRIGGER_SILENCE_MS", 10)),
        ) / 1000.0

        # 短停顿触发：说话中出现达到阈值的短静音视为一个分句（逗号位置），
        # 同一次停顿只触发一次（重新开口后才会再次武装）。
        silence_triggered = (
            trigger_silence > 0.0
            and self._silence_trigger_armed
            and silence_sec >= trigger_silence
        )
        # 保底：连续很久没有任何增量更新时强制刷新一次（含完全无停顿的连续说话）。
        fallback_triggered = elapsed >= fallback_interval
        if not silence_triggered and not fallback_triggered:
            return
        # 限流：短停顿触发至少间隔 min_interval 一次。
        if silence_triggered and elapsed < min_interval:
            return
        if silence_triggered:
            self._silence_trigger_armed = False
        self._last_partial_time = now
        self._enqueue_transcribe(audio, is_final=False)

    def _try_reuse_partial_as_final(self, segment: np.ndarray) -> str | None:
        """整句收束时，若最近一次已完成的中间结果之后没有任何新的有声内容
        （缓冲只是又追加了静音），该中间结果已覆盖整句，直接复用它作为
        最终结果，省掉一次重复的整句识别。"""
        if not getattr(config, "LOCAL_INCREMENTAL_ASR", True):
            return None
        if self._voiced_chunk_seq != self._last_partial_voiced_seq:
            return None
        text = (self._last_partial_text or "").strip()
        if not text:
            return None
        # print("[本地ASR] 终句复用中间结果（未出现新的有声内容，省一次识别）")
        return text

    def _emit_final_result(self, text: str, raw: dict | None) -> None:
        # 调用方需持有 self._lock
        self._last_partial_text = ""
        self._last_partial_voiced_seq = -1
        self._stream_id += 1
        self._reset_engine_draft()
        self._emit_result(text, is_final=True, raw=raw)

    def _reset_engine_draft(self) -> None:
        """整句翻篇：让引擎丢掉上一句的推测解码草稿，别拿它去猜下一句。"""
        engine = self._engine
        if engine is not None and hasattr(engine, "reset_draft"):
            try:
                engine.reset_draft()
            except Exception:  # pragma: no cover - 草稿只影响速度，失败不该打断识别
                logger.debug("reset_draft failed", exc_info=True)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        if self._vad._is_speaking and self._vad._speech_samples >= self._input_cap_samples():
            chunk = np.zeros_like(chunk)
        was_speaking = self._vad._is_speaking
        speech_segment = self._vad.process_chunk(chunk)
        if self._vad._is_speaking and self._vad._silence_counter == 0:
            # 本分块是有声内容：此后缓冲与任何既有中间结果快照不再等价。
            self._voiced_chunk_seq += 1
        if speech_segment is not None:
            # 整句提交（断句）：下一句的保底计时与短停顿触发从此处重新计。
            self._last_partial_time = time.monotonic()
            self._silence_trigger_armed = True
            reused = self._try_reuse_partial_as_final(speech_segment)
            if reused is not None:
                self._emit_final_result(reused, None)
            else:
                self._enqueue_transcribe(speech_segment, is_final=True)
            return
        if self._vad._is_speaking:
            if not was_speaking:
                # 新段刚进入说话状态：保底计时从"开口时刻"重新锚定，
                # 否则开口前的长静音（> 保底间隔）会让开口瞬间被误判为"连续说话很久未更新"而立即触发。
                self._last_partial_time = time.monotonic()
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
            reused = self._try_reuse_partial_as_final(segment)
            if reused is not None:
                self._emit_final_result(reused, None)
            else:
                self._enqueue_transcribe(segment, is_final=True)
        self._last_partial_text = ""
        # 新句子从此刻重新计时：保底间隔从新句子开始计而不是从进程启动计。
        self._last_partial_time = time.monotonic()
        self._silence_trigger_armed = True

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
            self._reset_engine_draft()
            self._last_partial_text = ""
            self._last_partial_time = time.monotonic()
            self._silence_trigger_armed = True
            self._voiced_chunk_seq = 0
            self._last_partial_voiced_seq = -1
            self._partial_pending_voiced_seq = -1
            self._waiting_partial_audio = None
            self._waiting_final_audio = None
            self._active_transcribe_future = None
            if self._asr_executor is None:
                self._asr_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="yakutan-local-inference",
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
