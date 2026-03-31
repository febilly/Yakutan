from __future__ import annotations

import collections
import logging

import numpy as np
import torch

from .model_manager import apply_cache_env

torch.set_num_threads(1)

logger = logging.getLogger(__name__)


class VADProcessor:
    """Voice activity detection with Silero, energy, or disabled mode."""

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
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()

        self._speech_buffer: list[np.ndarray] = []
        self._confidence_history: list[float] = []
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0

        self._pre_speech_chunks = 3
        self._pre_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=self._pre_speech_chunks
        )

        self._silence_mode = "auto"
        self._fixed_silence_dur = 0.8
        self._silence_limit = self._seconds_to_chunks(0.8)

        self._progressive_tiers = [
            (3.0, 1.0),
            (6.0, 0.5),
            (10.0, 0.25),
        ]

        self._pause_history: collections.deque[float] = collections.deque(maxlen=50)
        self._adaptive_min = 0.3
        self._adaptive_max = 2.0
        self.last_confidence = 0.0

    def _seconds_to_chunks(self, seconds: float) -> int:
        return max(1, round(seconds / self._chunk_duration))

    def _update_adaptive_limit(self) -> None:
        if len(self._pause_history) < 3:
            return
        pauses = sorted(self._pause_history)
        idx = int(len(pauses) * 0.75)
        p75 = pauses[min(idx, len(pauses) - 1)]
        target = max(self._adaptive_min, min(self._adaptive_max, p75 * 1.2))
        new_limit = self._seconds_to_chunks(target)
        if new_limit != self._silence_limit:
            logger.debug(
                "Adaptive silence: %.2fs (%s chunks), P75=%.2fs",
                target,
                new_limit,
                p75,
            )
            self._silence_limit = new_limit

    def update_settings(self, settings: dict) -> None:
        if "vad_mode" in settings:
            self.mode = settings["vad_mode"]
        if "vad_threshold" in settings:
            self.threshold = settings["vad_threshold"]
        if "energy_threshold" in settings:
            self.energy_threshold = settings["energy_threshold"]
        if "min_speech_duration" in settings:
            self.min_speech_samples = int(settings["min_speech_duration"] * self.sample_rate)
        if "silence_mode" in settings:
            self._silence_mode = settings["silence_mode"]
        if "silence_duration" in settings:
            self._fixed_silence_dur = settings["silence_duration"]
        if self._silence_mode == "fixed":
            self._silence_limit = self._seconds_to_chunks(self._fixed_silence_dur)

    def _silero_confidence(self, audio_chunk: np.ndarray) -> float:
        window_size = 512 if self.sample_rate == 16000 else 256
        chunk = audio_chunk[:window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        tensor = torch.from_numpy(chunk).float()
        return self._model(tensor, self.sample_rate).item()

    def _energy_confidence(self, audio_chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        return min(1.0, rms / (self.energy_threshold * 2))

    def _get_confidence(self, audio_chunk: np.ndarray) -> float:
        if self.mode == "silero":
            return self._silero_confidence(audio_chunk)
        if self.mode == "energy":
            return self._energy_confidence(audio_chunk)
        return 1.0

    def _get_effective_silence_limit(self) -> int:
        buf_seconds = self._speech_samples / self.sample_rate
        multiplier = 1.0
        for tier_seconds, tier_multiplier in self._progressive_tiers:
            if buf_seconds < tier_seconds:
                break
            multiplier = tier_multiplier
        return max(1, round(self._silence_limit * multiplier))

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray | None:
        confidence = self._get_confidence(audio_chunk)
        self.last_confidence = confidence

        effective_threshold = self.threshold if self.mode == "silero" else 0.5
        effective_silence_limit = self._get_effective_silence_limit()

        if confidence >= effective_threshold:
            if self._is_speaking and self._silence_counter > 0:
                pause_duration = self._silence_counter * self._chunk_duration
                if pause_duration >= 0.1:
                    self._pause_history.append(pause_duration)
                    if self._silence_mode == "auto":
                        self._update_adaptive_limit()

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

        if self._is_speaking and self._silence_counter >= effective_silence_limit:
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
