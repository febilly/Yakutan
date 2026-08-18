from __future__ import annotations

import time
import unittest
import wave
from pathlib import Path

import numpy as np

import config
from local_inference.model_manager import prepare_engine
from local_inference.vad_processor import VADProcessor
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


class _CountingStubEngine:
    """每次识别返回不同文本，便于区分"最终结果是复用还是重新识别"。"""

    def __init__(self) -> None:
        self.calls = 0

    def set_language(self, language: str) -> None:
        return None

    def unload(self) -> None:
        return None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        self.calls += 1
        return {"text": f"复用测试句{self.calls}", "language": "zh", "language_name": "zh"}


class LocalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SAMPLE_WAV.exists():
            raise unittest.SkipTest(f"缺少测试音频: {SAMPLE_WAV}")
        prepare_engine("sensevoice")

    def setUp(self) -> None:
        self._original_values = {
            "LOCAL_INFERENCE_ENGINE": config.LOCAL_INFERENCE_ENGINE,
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
        config.LOCAL_INFERENCE_ENGINE = "sensevoice"
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

    def _feed_until_open_segment(
        self,
        recognizer: "LocalSpeechRecognizer",
        audio: np.ndarray,
        sr: int,
        need_sec: float = 1.2,
    ) -> None:
        """小步喂入样本音频，直到落在一个"当前开口且尾部有声"的 VAD 段上：
        缓冲已累积 >= need_sec，且 VAD 仍在说话、尾部不是长静音。
        样本音频内部的自然间隙可能提前断句，这里跨间隙继续喂直到条件满足。"""
        head_end = min(6 * sr, len(audio))
        offset = 0
        step = sr // 5
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if offset < head_end:
                end = min(offset + step, head_end)
                for chunk in chunk_pcm16(audio[offset:end]):
                    recognizer.send_audio_frame(chunk)
                    time.sleep(0.005)
                offset = end
            with recognizer._lock:
                peek = recognizer._vad.peek_buffer()
                tail_voiced = (
                    recognizer._vad._is_speaking
                    and recognizer._vad._silence_counter <= 2
                )
            if peek is not None and peek[1] >= need_sec and tail_voiced:
                # 排空积压：快速喂入会让队列堆积，而队列积压 >= 8 时
                # 增量触发会被"防陈旧更新"闸门跳过，受控静音阶段前必须先排空。
                time.sleep(0.3)
                return
            time.sleep(0.02)
        raise AssertionError("超时：未找到满足条件的开口语音段")

    def test_final_reuses_partial_when_no_new_voiced_audio(self) -> None:
        """句尾短停顿已触发过中间结果且之后无新语音时，终句应直接复用，
        不再对整句音频重复识别一次。"""
        audio, sample_rate = load_audio_file(SAMPLE_WAV)
        if sample_rate != 16000:
            self.skipTest("测试音频采样率不是 16kHz")

        sr = 16000
        callback = CollectingCallback()
        engine = _CountingStubEngine()
        recognizer = LocalSpeechRecognizer(callback=callback, sample_rate=16000, source_language="auto")
        recognizer._engine = engine
        recognizer._ensure_engine = lambda: engine  # type: ignore[assignment]
        recognizer.start()
        try:
            self._feed_until_open_segment(recognizer, audio, sr)
            calls_before = engine.calls

            # 400ms 短静音（小于 800ms 断句阈值）→ 触发一次增量更新
            for chunk in chunk_pcm16(np.zeros(int(0.4 * sr), dtype=np.float32)):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.01)

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not any(
                not event.is_final for event in callback.events
            ):
                time.sleep(0.05)
            partial = next(
                (e for e in callback.events if not e.is_final), None,
            )
            self.assertIsNotNone(partial, "短停顿未触发增量更新")
            events_after_partial = len(callback.events)

            # 再追加 1.2s 纯静音（累计越过 800ms 断句阈值，且不再出现任何语音）
            # → 终句应复用中间结果
            for chunk in chunk_pcm16(np.zeros(int(1.2 * sr), dtype=np.float32)):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.01)

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not any(
                event.is_final for event in callback.events[events_after_partial:]
            ):
                time.sleep(0.05)
        finally:
            recognizer.stop()

        self.assertFalse(callback.errors)
        final = next(
            event for event in callback.events[events_after_partial:] if event.is_final
        )
        self.assertEqual(final.text, partial.text, "无声续时终句应复用中间结果文本")
        self.assertEqual(
            engine.calls,
            calls_before + 1,
            "复用场景下不应再对整句音频重复识别",
        )

    def test_final_retranscribes_when_new_voiced_after_partial(self) -> None:
        """中间结果之后又出现新语音时，整句音频与快照不等价，终句必须重新识别。"""
        audio, sample_rate = load_audio_file(SAMPLE_WAV)
        if sample_rate != 16000:
            self.skipTest("测试音频采样率不是 16kHz")

        sr = 16000
        callback = CollectingCallback()
        engine = _CountingStubEngine()
        recognizer = LocalSpeechRecognizer(callback=callback, sample_rate=16000, source_language="auto")
        recognizer._engine = engine
        recognizer._ensure_engine = lambda: engine  # type: ignore[assignment]
        recognizer.start()
        try:
            self._feed_until_open_segment(recognizer, audio, sr)

            for chunk in chunk_pcm16(np.zeros(int(0.4 * sr), dtype=np.float32)):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.01)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not any(
                not event.is_final for event in callback.events
            ):
                time.sleep(0.05)
            partial = next(
                (e for e in callback.events if not e.is_final), None,
            )
            self.assertIsNotNone(partial, "短停顿未触发增量更新")
            events_after_partial = len(callback.events)
            calls_before_more = engine.calls

            # 继续说话（新语音使快照失效）+ 尾静音断句 → 终句必须重新识别
            more_offset = min(2 * sr, max(0, len(audio) - 2 * sr))
            more = audio[more_offset : more_offset + 2 * sr]
            for chunk in chunk_pcm16(more):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.01)
            for chunk in chunk_pcm16(np.zeros(int(1.2 * sr), dtype=np.float32)):
                recognizer.send_audio_frame(chunk)
                time.sleep(0.01)

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not any(
                event.is_final for event in callback.events[events_after_partial:]
            ):
                time.sleep(0.05)
        finally:
            recognizer.stop()

        self.assertFalse(callback.errors)
        final = next(
            event for event in callback.events[events_after_partial:] if event.is_final
        )
        self.assertGreaterEqual(
            engine.calls,
            calls_before_more + 1,
            "有新语音时终句应重新识别，而不是复用中间结果",
        )
        self.assertNotEqual(
            final.text, partial.text, "终句文本应来自重新识别而非复用上次中间结果"
        )

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
