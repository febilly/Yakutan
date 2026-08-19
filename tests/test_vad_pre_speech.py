"""起声预缓冲测试：门控 GatePreBuffer 与 VADProcessor 段首拼接。"""

from __future__ import annotations

import numpy as np

from audio_capture import GatePreBuffer
from local_inference.vad_processor import VADProcessor

CHUNK = 512
CHUNK_DURATION = CHUNK / 16000


class TestGatePreBuffer:
    def test_flush_returns_frames_in_order_and_clears(self) -> None:
        buf = GatePreBuffer(cap_bytes=1024 * 1024)
        f1, f2, f3 = b"aa", b"bb", b"cc"
        buf.push(f1)
        buf.push(f2)
        buf.push(f3)
        assert len(buf) == 3
        assert buf.flush() == [f1, f2, f3]
        assert len(buf) == 0
        assert buf.flush() == []

    def test_cap_trims_oldest_frames(self) -> None:
        # 上限 10 字节：保留最近帧，最旧帧被丢弃
        buf = GatePreBuffer(cap_bytes=10)
        for i in range(6):
            buf.push(b"x" * 4)
        frames = buf.flush()
        assert sum(len(f) for f in frames) <= 10 + 4
        assert len(frames) == 2
        assert all(f == b"xxxx" for f in frames)

    def test_single_frame_larger_than_cap_is_kept(self) -> None:
        buf = GatePreBuffer(cap_bytes=4)
        buf.push(b"y" * 100)
        frames = buf.flush()
        assert frames == [b"y" * 100]

    def test_zero_cap_disables_buffering(self) -> None:
        buf = GatePreBuffer(cap_bytes=0)
        buf.push(b"zz")
        assert len(buf) == 0
        assert buf.flush() == []

    def test_clear(self) -> None:
        buf = GatePreBuffer(cap_bytes=1024)
        buf.push(b"aa")
        buf.clear()
        assert len(buf) == 0
        assert buf.flush() == []


def _make_energy_vad(pre_speech_duration: float = 0.5) -> VADProcessor:
    return VADProcessor(
        sample_rate=16000,
        threshold=0.50,
        min_speech_duration=0.2,
        chunk_duration=CHUNK_DURATION,
        pre_speech_duration=pre_speech_duration,
    )


class TestVadProcessorPreSpeech:
    def test_pre_speech_chunks_for_half_second(self) -> None:
        vad = _make_energy_vad(pre_speech_duration=0.5)
        assert vad._pre_speech_chunks == 16  # ceil(0.5 / 0.032)

    def test_default_pre_speech_duration_is_half_second(self) -> None:
        vad = VADProcessor(
            sample_rate=16000,
            threshold=0.50,
            min_speech_duration=1.0,
            chunk_duration=CHUNK_DURATION,
        )
        assert vad._pre_speech_chunks == 16

    def test_segment_starts_with_pre_speech_audio(self) -> None:
        vad = _make_energy_vad(pre_speech_duration=0.5)
        vad.update_settings({"vad_mode": "energy", "silence_duration": 0.1})

        noise = np.full(CHUNK, 0.001, dtype=np.float32)  # 低于阈值：视为静音
        tone = np.full(CHUNK, 0.1, dtype=np.float32)     # 高于阈值：语音
        silence = np.zeros(CHUNK, dtype=np.float32)

        segment = None
        # 20 个低幅值块：预缓冲最终只保留最近 16 个
        for _ in range(20):
            segment = vad.process_chunk(noise)
            assert segment is None
        # 语音开始：第一个 tone 块触发 SILENCE→SPEECH，预缓冲应拼到段首
        for _ in range(40):
            segment = vad.process_chunk(tone)
            if segment is not None:
                break
        # 尾部静音触发断句
        if segment is None:
            for _ in range(5):
                segment = vad.process_chunk(silence)
                if segment is not None:
                    break

        assert segment is not None
        # 段首 16 块为预缓冲（低幅值噪声），其后为语音
        assert len(segment) >= (16 + 40) * CHUNK
        head = segment[: 16 * CHUNK]
        assert np.allclose(head, 0.001)
        assert float(np.mean(np.abs(segment[16 * CHUNK : 17 * CHUNK]))) > 0.05

    def test_zero_pre_speech_disables_prepending(self) -> None:
        vad = _make_energy_vad(pre_speech_duration=0.0)
        vad.update_settings({"vad_mode": "energy", "silence_duration": 0.1})

        noise = np.full(CHUNK, 0.001, dtype=np.float32)
        tone = np.full(CHUNK, 0.1, dtype=np.float32)
        silence = np.zeros(CHUNK, dtype=np.float32)

        segment = None
        for _ in range(20):
            segment = vad.process_chunk(noise)
        for _ in range(40):
            segment = vad.process_chunk(tone)
            if segment is not None:
                break
        if segment is None:
            for _ in range(5):
                segment = vad.process_chunk(silence)
                if segment is not None:
                    break

        assert segment is not None
        # 无预缓冲：段首直接是语音，而不是低幅值噪声
        assert float(np.mean(np.abs(segment[:CHUNK]))) > 0.05
