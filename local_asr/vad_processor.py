from __future__ import annotations

import collections
import logging
import numpy as np

from .model_manager import apply_cache_env, silero_onnx_path

logger = logging.getLogger(__name__)


class _SileroOnnxVAD:
    """Silero VAD via `silero_vad_16k_op15.onnx` (no PyTorch)."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset_states()

    def reset_states(self, batch_size: int = 1) -> None:
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context: np.ndarray | None = None
        self._last_sr = 0
        self._last_batch_size = 0

    def probability(self, audio_chunk: np.ndarray, sample_rate: int) -> float:
        batch_size = 1
        num_samples = 512 if sample_rate == 16000 else 256
        context_size = 64 if sample_rate == 16000 else 32

        x = np.asarray(audio_chunk, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] != batch_size:
            raise ValueError("Silero ONNX path supports batch size 1 only")

        if x.shape[1] < num_samples:
            x = np.pad(x, ((0, 0), (0, num_samples - x.shape[1])))
        elif x.shape[1] > num_samples:
            x = x[:, :num_samples]

        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sample_rate:
            self.reset_states(batch_size)

        if self._context is None or self._context.shape != (batch_size, context_size):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        inp = np.concatenate([self._context, x], axis=1)
        ort_inputs = {
            "input": inp.astype(np.float32, copy=False),
            "state": self._state,
            "sr": np.array(sample_rate, dtype=np.int64),
        }
        prob_arr, self._state = self._session.run(None, ort_inputs)
        self._context = inp[..., -context_size:]
        self._last_sr = sample_rate
        self._last_batch_size = batch_size
        return float(prob_arr.reshape(-1)[0])


class VADProcessor:
    """Voice activity detection with Silero (ONNX), energy, or disabled mode."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.50,
        min_speech_duration: float = 1.0,
        chunk_duration: float = 0.032,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.energy_threshold = 0.02
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self._chunk_duration = chunk_duration
        self.mode = "silero"

        apply_cache_env()
        self._silero: _SileroOnnxVAD | None = None

        self._speech_buffer: list[np.ndarray] = []
        self._confidence_history: list[float] = []
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0

        self._pre_speech_chunks = 3
        self._pre_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=self._pre_speech_chunks
        )

        self._silence_limit = self._seconds_to_chunks(0.8)
        self.last_confidence = 0.0

    def _ensure_silero(self) -> _SileroOnnxVAD:
        if self._silero is not None:
            return self._silero
        path = silero_onnx_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Silero VAD ONNX 未找到: {path}。请先运行 download_silero() 或 prepare_engine()。"
            )
        self._silero = _SileroOnnxVAD(str(path))
        return self._silero

    def _seconds_to_chunks(self, seconds: float) -> int:
        return max(1, round(seconds / self._chunk_duration))

    def update_settings(self, settings: dict) -> None:
        if "vad_mode" in settings:
            self.mode = settings["vad_mode"]
        if "vad_threshold" in settings:
            self.threshold = settings["vad_threshold"]
        if "energy_threshold" in settings:
            self.energy_threshold = settings["energy_threshold"]
        if "min_speech_duration" in settings:
            self.min_speech_samples = int(settings["min_speech_duration"] * self.sample_rate)
        if "silence_duration" in settings:
            self._silence_limit = self._seconds_to_chunks(float(settings["silence_duration"]))

    def _silero_confidence(self, audio_chunk: np.ndarray) -> float:
        window_size = 512 if self.sample_rate == 16000 else 256
        chunk = audio_chunk[:window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        return self._ensure_silero().probability(chunk.astype(np.float32, copy=False), self.sample_rate)

    def _energy_confidence(self, audio_chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        return min(1.0, rms / (self.energy_threshold * 2))

    def _get_confidence(self, audio_chunk: np.ndarray) -> float:
        if self.mode == "silero":
            return self._silero_confidence(audio_chunk)
        if self.mode == "energy":
            return self._energy_confidence(audio_chunk)
        return 1.0

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray | None:
        confidence = self._get_confidence(audio_chunk)
        self.last_confidence = confidence

        effective_threshold = self.threshold if self.mode == "silero" else 0.5

        if confidence >= effective_threshold:
            if not self._is_speaking:
                for pre_chunk in self._pre_buffer:
                    self._speech_buffer.append(pre_chunk)
                    self._confidence_history.append(effective_threshold)
                    self._speech_samples += len(pre_chunk)
                self._pre_buffer.clear()

            self._is_speaking = True
            self._silence_counter = 0
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)
        elif self._is_speaking:
            self._silence_counter += 1
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)
        else:
            self._pre_buffer.append(audio_chunk)

        if self._is_speaking and self._silence_counter >= self._silence_limit:
            if self._speech_samples >= self.min_speech_samples:
                return self._flush_segment()
            logger.debug(
                "Short segment %.1fs < min %.1fs, keeping for merge",
                self._speech_samples / self.sample_rate,
                self.min_speech_samples / self.sample_rate,
            )
            self._is_speaking = False
            self._silence_counter = 0
            return None

        return None

    def _flush_segment(self) -> np.ndarray | None:
        if not self._speech_buffer:
            return None
        if len(self._confidence_history) >= 4:
            effective_threshold = self.threshold if self.mode == "silero" else 0.5
            voiced = sum(1 for confidence in self._confidence_history if confidence >= effective_threshold)
            density = voiced / len(self._confidence_history)
            if density < 0.25:
                logger.debug(
                    "Low speech density %.0f%%, discarding %.1fs",
                    density * 100,
                    self._speech_samples / self.sample_rate,
                )
                self._reset()
                return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def _reset(self) -> None:
        self._speech_buffer = []
        self._confidence_history = []
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0

    def peek_buffer(self) -> tuple[np.ndarray, float] | None:
        if not self._speech_buffer or not self._is_speaking:
            return None
        audio = np.concatenate(self._speech_buffer)
        return audio, self._speech_samples / self.sample_rate

    def force_flush(self) -> np.ndarray | None:
        if not self._speech_buffer:
            return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def flush(self) -> np.ndarray | None:
        if self._speech_samples >= self.min_speech_samples:
            return self._flush_segment()
        self._reset()
        return None
