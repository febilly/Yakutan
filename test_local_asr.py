from __future__ import annotations

import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

import config
from local_asr.model_manager import prepare_engine
from local_asr.vad_processor import VADProcessor
from speech_recognizers.base_speech_recognizer import SpeechRecognitionCallback
from speech_recognizers.local_speech_recognizer import (
    LOCAL_VAD_CHUNK_SAMPLES,
    LocalSpeechRecognizer,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_WAV = PROJECT_ROOT / "temp" / "录音.wav"
LONG_AUDIO = PROJECT_ROOT / "temp" / "tests" / "14min_01.flac"


def load_audio_file(path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            if max_seconds is not None:
                frames = min(frames, int(sample_rate * max_seconds))
            raw = wav_file.readframes(frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            channels = wav_file.getnchannels()
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return audio, sample_rate

    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if max_seconds is not None:
        audio = audio[: int(sample_rate * max_seconds)]
    return audio.astype(np.float32), sample_rate


def chunk_pcm16(audio: np.ndarray, chunk_samples: int = LOCAL_VAD_CHUNK_SAMPLES) -> list[bytes]:
    audio16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    chunks: list[bytes] = []
    for offset in range(0, len(audio16), chunk_samples):
        chunk = audio16[offset : offset + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk.tobytes())
    return chunks


class CollectingCallback(SpeechRecognitionCallback):
    def __init__(self) -> None:
        self.events = []
        self.errors = []
        self.started = 0
        self.stopped = 0

    def on_session_started(self) -> None:
        self.started += 1

    def on_session_stopped(self) -> None:
        self.stopped += 1

    def on_error(self, error: Exception) -> None:
        self.errors.append(error)

    def on_result(self, event) -> None:
        self.events.append(event)


class _StubEngine:
    def __init__(self) -> None:
        self.calls = 0
        self.language = "auto"

    def set_language(self, language: str) -> None:
        self.language = language

    def unload(self) -> None:
        return None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        self.calls += 1
        if self.calls == 1:
            return {"text": "第一句，第二句", "language": "zh", "language_name": "zh"}
        return {"text": "最终一句", "language": "zh", "language_name": "zh"}


class LocalAsrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SAMPLE_WAV.exists():
            raise unittest.SkipTest(f"缺少测试音频: {SAMPLE_WAV}")
        prepare_engine("sensevoice")

    def setUp(self) -> None:
        self._original_values = {
            "LOCAL_ASR_ENGINE": config.LOCAL_ASR_ENGINE,
            "LOCAL_ASR_LANGUAGE": config.LOCAL_ASR_LANGUAGE,
            "LOCAL_INCREMENTAL_ASR": config.LOCAL_INCREMENTAL_ASR,
            "LOCAL_INTERIM_INTERVAL": config.LOCAL_INTERIM_INTERVAL,
            "LOCAL_VAD_MODE": config.LOCAL_VAD_MODE,
            "LOCAL_VAD_THRESHOLD": config.LOCAL_VAD_THRESHOLD,
            "LOCAL_VAD_MIN_SPEECH_DURATION": config.LOCAL_VAD_MIN_SPEECH_DURATION,
            "LOCAL_VAD_MAX_SPEECH_DURATION": config.LOCAL_VAD_MAX_SPEECH_DURATION,
            "LOCAL_VAD_SILENCE_MODE": config.LOCAL_VAD_SILENCE_MODE,
            "LOCAL_VAD_SILENCE_DURATION": config.LOCAL_VAD_SILENCE_DURATION,
        }
        config.LOCAL_ASR_ENGINE = "sensevoice"
        config.LOCAL_ASR_LANGUAGE = "auto"
        config.LOCAL_INCREMENTAL_ASR = True
        config.LOCAL_INTERIM_INTERVAL = 1.5
        config.LOCAL_VAD_MODE = "silero"
        config.LOCAL_VAD_THRESHOLD = 0.50
        config.LOCAL_VAD_MIN_SPEECH_DURATION = 1.0
        config.LOCAL_VAD_MAX_SPEECH_DURATION = 30.0
        config.LOCAL_VAD_SILENCE_MODE = "auto"
        config.LOCAL_VAD_SILENCE_DURATION = 0.8

    def tearDown(self) -> None:
        for key, value in self._original_values.items():
            setattr(config, key, value)

    def test_vad_detects_segments_from_sample_audio(self) -> None:
        audio, sample_rate = load_audio_file(SAMPLE_WAV)
        if sample_rate != 16000:
            self.skipTest("测试音频采样率不是 16kHz，当前测试只覆盖 16kHz 路径")

        vad = VADProcessor(
            sample_rate=16000,
            threshold=0.50,
            min_speech_duration=1.0,
            chunk_duration=LOCAL_VAD_CHUNK_SAMPLES / 16000,
        )
        segments = []
        for chunk in np.array_split(audio[: len(audio) - (len(audio) % LOCAL_VAD_CHUNK_SAMPLES)], len(audio) // LOCAL_VAD_CHUNK_SAMPLES):
            seg = vad.process_chunk(chunk.astype(np.float32))
            if seg is not None:
                segments.append(seg)
        flushed = vad.force_flush() if vad._is_speaking else vad.flush()
        if flushed is not None:
            segments.append(flushed)

        self.assertGreaterEqual(len(segments), 1)
        self.assertTrue(any(len(segment) > 16000 for segment in segments))

    def test_local_recognizer_emits_final_result(self) -> None:
        audio, sample_rate = load_audio_file(SAMPLE_WAV)
        if sample_rate != 16000:
            self.skipTest("测试音频采样率不是 16kHz")

        callback = CollectingCallback()
        recognizer = LocalSpeechRecognizer(callback=callback, sample_rate=16000, source_language="auto")
        recognizer.start()
        try:
            for chunk in chunk_pcm16(audio):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.002)
            time.sleep(1.5)
        finally:
            recognizer.stop()

        self.assertFalse(callback.errors)
        self.assertGreaterEqual(callback.started, 1)
        self.assertGreaterEqual(callback.stopped, 1)
        self.assertTrue(any(event.is_final for event in callback.events))
        self.assertTrue(any((event.text or "").strip() for event in callback.events))

    def test_local_recognizer_partial_and_final_path_with_stub_engine(self) -> None:
        audio, sample_rate = load_audio_file(SAMPLE_WAV)
        if sample_rate != 16000:
            self.skipTest("测试音频采样率不是 16kHz")

        callback = CollectingCallback()
        recognizer = LocalSpeechRecognizer(callback=callback, sample_rate=16000, source_language="auto")
        recognizer._engine = _StubEngine()
        recognizer._ensure_engine = lambda: recognizer._engine  # type: ignore[assignment]
        recognizer.start()
        try:
            for chunk in chunk_pcm16(audio[: 16000 * 6]):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.002)
            time.sleep(1.0)
            recognizer.pause()
            recognizer.resume()
        finally:
            recognizer.stop()

        self.assertFalse(callback.errors)
        self.assertTrue(any(not event.is_final for event in callback.events))
        self.assertTrue(any(event.is_final for event in callback.events))

    def test_long_audio_excerpt_vad_or_recognizer_smoke(self) -> None:
        if not LONG_AUDIO.exists():
            self.skipTest(f"缺少长音频样本: {LONG_AUDIO}")
        audio, sample_rate = load_audio_file(LONG_AUDIO, max_seconds=20)
        if sample_rate != 16000:
            self.skipTest("长音频样本采样率不是 16kHz")
        callback = CollectingCallback()
        recognizer = LocalSpeechRecognizer(callback=callback, sample_rate=16000, source_language="auto")
        recognizer._engine = _StubEngine()
        recognizer._ensure_engine = lambda: recognizer._engine  # type: ignore[assignment]
        recognizer.start()
        try:
            for chunk in chunk_pcm16(audio):
                recognizer.send_audio_frame(chunk)
            time.sleep(1.0)
        finally:
            recognizer.stop()
        self.assertTrue(callback.events)


if __name__ == "__main__":
    unittest.main()
