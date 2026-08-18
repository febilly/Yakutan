from __future__ import annotations

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
            "LOCAL_INCREMENTAL_ASR": config.LOCAL_INCREMENTAL_ASR,
            "LOCAL_INCREMENTAL_TRIGGER_SILENCE_MS": config.LOCAL_INCREMENTAL_TRIGGER_SILENCE_MS,
            "LOCAL_INCREMENTAL_MIN_UPDATE_INTERVAL": config.LOCAL_INCREMENTAL_MIN_UPDATE_INTERVAL,
            "LOCAL_INCREMENTAL_MAX_UPDATE_INTERVAL": config.LOCAL_INCREMENTAL_MAX_UPDATE_INTERVAL,
            "LOCAL_VAD_MODE": config.LOCAL_VAD_MODE,
            "LOCAL_VAD_THRESHOLD": config.LOCAL_VAD_THRESHOLD,
            "LOCAL_VAD_MIN_SPEECH_DURATION": config.LOCAL_VAD_MIN_SPEECH_DURATION,
            "LOCAL_VAD_MAX_SPEECH_DURATION": config.LOCAL_VAD_MAX_SPEECH_DURATION,
            "LOCAL_VAD_SILENCE_DURATION": config.LOCAL_VAD_SILENCE_DURATION,
        }
        config.LOCAL_ASR_ENGINE = "sensevoice"
        config.LOCAL_INCREMENTAL_ASR = True
        config.LOCAL_INCREMENTAL_TRIGGER_SILENCE_MS = 100
        config.LOCAL_INCREMENTAL_MIN_UPDATE_INTERVAL = 0.1
        config.LOCAL_INCREMENTAL_MAX_UPDATE_INTERVAL = 1.0
        config.LOCAL_VAD_MODE = "silero"
        config.LOCAL_VAD_THRESHOLD = 0.50
        config.LOCAL_VAD_MIN_SPEECH_DURATION = 1.0
        config.LOCAL_VAD_MAX_SPEECH_DURATION = 30.0
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

        sr = 16000
        silence_400ms = np.zeros(sr // 100 * 4, dtype=np.float32)
        silence_1200ms = np.zeros(sr // 100 * 12, dtype=np.float32)

        callback = CollectingCallback()
        recognizer = LocalSpeechRecognizer(callback=callback, sample_rate=16000, source_language="auto")
        recognizer._engine = _StubEngine()
        recognizer._ensure_engine = lambda: recognizer._engine  # type: ignore[assignment]
        recognizer.start()
        try:
            # 阶段一：喂入真实语音，直到当前 VAD 段内累积了足够的说话内容
            head_end = min(6 * sr, len(audio))
            offset = 0
            peek = None
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if offset < head_end:
                    end = min(offset + sr, head_end)
                    for chunk in chunk_pcm16(audio[offset:end]):
                        recognizer.send_audio_frame(chunk)
                        time.sleep(0.005)
                    offset = end
                with recognizer._lock:
                    peek = recognizer._vad.peek_buffer()
                if peek is not None and peek[1] >= 1.2:
                    break
                time.sleep(0.05)
            self.assertIsNotNone(peek, "样本音频前 6 秒内未出现足够长的连续语音段")

            # 阶段二：400ms 短静音（小于断句阈值）→ 应触发一次增量更新（逗号位置）
            for chunk in chunk_pcm16(silence_400ms):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.01)

            # 阶段三：继续说话，再以 1.2s 尾静音结束整句 → 应提交最终结果
            more = audio[: 3 * sr]
            for chunk in chunk_pcm16(more):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.005)
            for chunk in chunk_pcm16(silence_1200ms):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.01)

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                has_partial = any(not event.is_final for event in callback.events)
                has_final = any(event.is_final for event in callback.events)
                if has_partial and has_final:
                    break
                time.sleep(0.05)
        finally:
            recognizer.stop()

        self.assertFalse(callback.errors)
        self.assertTrue(any(not event.is_final for event in callback.events))
        self.assertTrue(any(event.is_final for event in callback.events))

    def test_fallback_not_triggered_by_stale_timer_before_speech(self) -> None:
        """回归：开口前的长静音不应算作"连续说话"，保底计时应从开口时刻锚定。"""
        audio, sample_rate = load_audio_file(SAMPLE_WAV)
        if sample_rate != 16000:
            self.skipTest("测试音频采样率不是 16kHz")

        sr = 16000
        # 本测试只观察保底路径：关闭短停顿触发，并把保底间隔调大到 5 秒
        config.LOCAL_INCREMENTAL_TRIGGER_SILENCE_MS = 0
        config.LOCAL_INCREMENTAL_MAX_UPDATE_INTERVAL = 5.0

        callback = CollectingCallback()
        recognizer = LocalSpeechRecognizer(callback=callback, sample_rate=16000, source_language="auto")
        recognizer._engine = _StubEngine()
        recognizer._ensure_engine = lambda: recognizer._engine  # type: ignore[assignment]
        recognizer.start()
        try:
            audio = audio.astype(np.float32)
            # 阶段一：制造"启动已久、一直沉默"的陈旧计时（回拨 10 秒 > 5 秒保底间隔）
            recognizer._last_partial_time = time.monotonic() - 10.0

            offset = 0
            end_limit = min(6 * sr, len(audio))
            while offset < end_limit and not recognizer._vad._is_speaking:
                end = min(offset + sr // 10, end_limit)
                with recognizer._lock:
                    recognizer._feed_samples(audio[offset:end])
                offset = end
            self.assertTrue(recognizer._vad._is_speaking, "样本音频前 6 秒内未检测到语音，无法测试开口瞬间行为")

            # 阶段二：开口后继续说 1.2 秒（缓冲越过 1 秒门槛，但距开口不到保底间隔）→ 不应触发
            end = min(offset + 12 * sr // 10, end_limit)
            with recognizer._lock:
                recognizer._feed_samples(audio[offset:end])
            offset = end
            self.assertIsNone(
                recognizer._active_transcribe_future,
                "开口不应被开口的陈旧计时误判为'连续说话很久'而立即保底触发",
            )
            self.assertFalse(
                any(not event.is_final for event in callback.events),
                "开口前的长静音不应计入保底时间",
            )

            # 阶段三：保底机制本身仍要有效——把锚点回拨 6 秒（> 5 秒保底间隔）后继续说话 → 应触发增量更新
            while offset < end_limit and not any(not event.is_final for event in callback.events):
                with recognizer._lock:
                    recognizer._last_partial_time = time.monotonic() - 6.0
                    end = min(offset + sr // 10, end_limit)
                    recognizer._feed_samples(audio[offset:end])
                offset = end
            self.assertTrue(
                any(not event.is_final for event in callback.events),
                "锚点超过保底间隔后应触发保底增量更新",
            )
        finally:
            recognizer.stop()

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
