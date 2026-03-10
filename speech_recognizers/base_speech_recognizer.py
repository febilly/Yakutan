from __future__ import annotations

from array import array
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
from typing import Any, Optional


@dataclass
class RecognitionEvent:
    """Container for a single incremental recognition result."""

    text: str
    is_final: bool
    confidence: Optional[float] = None
    raw: Optional[Any] = None


class SpeechRecognitionCallback(ABC):
    """Callback interface for speech recognition events."""

    def on_session_started(self) -> None:  # pragma: no cover - optional hook
        """Called when the recognizer session starts."""
        pass

    def on_session_stopped(self) -> None:  # pragma: no cover - optional hook
        """Called when the recognizer session stops."""
        pass

    def on_error(self, error: Exception) -> None:  # pragma: no cover - optional hook
        """Called when an unrecoverable error occurs."""
        pass

    @abstractmethod
    def on_result(self, event: RecognitionEvent) -> None:
        """Called when new recognition text becomes available."""


class SpeechRecognizer(ABC):
    """Abstract base class for speech recognition backends."""

    @abstractmethod
    def set_callback(self, callback: SpeechRecognitionCallback) -> None:
        """Register the callback that will receive recognition events."""

    @abstractmethod
    def start(self) -> None:
        """Start the recognition session."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the recognition session."""

    @abstractmethod
    def send_audio_frame(self, data: bytes) -> None:
        """Send a chunk of audio data to the recognizer."""

    @abstractmethod
    def pause(self) -> None:
        """Temporarily pause recognition while keeping the session alive if possible."""

    @abstractmethod
    def resume(self) -> None:
        """Resume recognition after a previous pause."""

    @abstractmethod
    def get_last_request_id(self) -> Optional[str]:
        """Return the request identifier for the most recent session."""

    @abstractmethod
    def get_first_package_delay(self) -> Optional[int]:
        """Latency in milliseconds for the first package, if available."""

    @abstractmethod
    def get_last_package_delay(self) -> Optional[int]:
        """Latency in milliseconds for the last package, if available."""


def mix_pcm16le_to_mono(data: bytes, channels: int) -> bytes:
    """Downmix little-endian 16-bit PCM audio to mono."""
    normalized_channels = max(1, int(channels))
    if not data or normalized_channels == 1:
        return data

    frame_width = normalized_channels * 2
    usable_bytes = len(data) - (len(data) % frame_width)
    if usable_bytes <= 0:
        return b""

    samples = array("h")
    samples.frombytes(data[:usable_bytes])
    if sys.byteorder != "little":
        samples.byteswap()

    mono_samples = array("h")
    for offset in range(0, len(samples), normalized_channels):
        total = 0
        for channel_index in range(normalized_channels):
            total += int(samples[offset + channel_index])
        mixed = int(round(total / normalized_channels))
        if mixed > 32767:
            mixed = 32767
        elif mixed < -32768:
            mixed = -32768
        mono_samples.append(mixed)

    if sys.byteorder != "little":
        mono_samples.byteswap()
    return mono_samples.tobytes()


class MonoAudioSpeechRecognizer(SpeechRecognizer):
    """Recognizer wrapper that guarantees outgoing PCM frames are mono."""

    def __init__(self, recognizer: SpeechRecognizer, input_channels: int = 1) -> None:
        self._recognizer = recognizer
        self._input_channels = max(1, int(input_channels))

    def set_callback(self, callback: SpeechRecognitionCallback) -> None:
        self._recognizer.set_callback(callback)

    def start(self) -> None:
        self._recognizer.start()

    def stop(self) -> None:
        self._recognizer.stop()

    def send_audio_frame(self, data: bytes) -> None:
        mono_data = mix_pcm16le_to_mono(data, self._input_channels)
        if mono_data:
            self._recognizer.send_audio_frame(mono_data)

    def pause(self) -> None:
        self._recognizer.pause()

    def resume(self) -> None:
        self._recognizer.resume()

    def get_last_request_id(self) -> Optional[str]:
        return self._recognizer.get_last_request_id()

    def get_first_package_delay(self) -> Optional[int]:
        return self._recognizer.get_first_package_delay()

    def get_last_package_delay(self) -> Optional[int]:
        return self._recognizer.get_last_package_delay()
