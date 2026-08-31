"""Regression tests for online ASR finalization after local VAD closes the gate."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import audio_capture
import numpy as np


def test_dashscope_recognition_backends_receive_full_server_vad_window():
    with (
        patch.object(audio_capture.config, 'LOCAL_VAD_SILENCE_DURATION', 0.8),
        patch.object(audio_capture.config, 'ONLINE_VAD_END_BURST_MS', 200),
    ):
        assert audio_capture._vad_gate_tail_seconds('qwen_audio3') == 1.0
        assert audio_capture._vad_gate_tail_seconds('dashscope') == 1.0


def test_other_online_backends_keep_margin_only_tail():
    with (
        patch.object(audio_capture.config, 'LOCAL_VAD_SILENCE_DURATION', 0.8),
        patch.object(audio_capture.config, 'ONLINE_VAD_END_BURST_MS', 200),
    ):
        assert audio_capture._vad_gate_tail_seconds('soniox') == 0.2


def test_silence_burst_is_split_into_api_safe_unpaced_frames():
    burst = audio_capture._make_silence_burst(1.0, 16000)
    frames = list(audio_capture._iter_silence_burst_frames(burst))

    assert sum(map(len, frames)) == 32000
    assert len(frames) == 2
    assert all(0 < len(frame) <= 16 * 1024 for frame in frames)
    assert all(len(frame) % 2 == 0 for frame in frames)
    assert all(not any(frame) for frame in frames)


def test_sender_expands_tail_into_consecutive_frames():
    state = SimpleNamespace(recognition_active=True, audio_send_generation=7)
    burst = audio_capture._make_silence_burst(1.0, 16000)

    with patch.object(
        audio_capture,
        'send_audio_frame_async',
        new_callable=AsyncMock,
    ) as send:
        asyncio.run(audio_capture._send_queue_payload(state, object(), 7, burst))

    frames = [call.args[2] for call in send.await_args_list]
    assert len(frames) == 2
    assert sum(map(len, frames)) == 32000


def test_recognition_control_uses_serialized_asr_executor():
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='test-vad-control')
    state = SimpleNamespace(
        asr_send_executor=executor,
        ensure_asr_send_executor=MagicMock(),
    )
    recognizer = MagicMock()

    try:
        assert asyncio.run(
            audio_capture._run_recognizer_control_async(state, recognizer, 'pause')
        )
    finally:
        executor.shutdown(wait=True)

    state.ensure_asr_send_executor.assert_called_once()
    recognizer.pause.assert_called_once()


def test_recognition_control_reports_failure_for_fallback():
    executor = ThreadPoolExecutor(max_workers=1)
    state = SimpleNamespace(
        asr_send_executor=executor,
        ensure_asr_send_executor=MagicMock(),
    )
    recognizer = MagicMock()
    recognizer.pause.side_effect = RuntimeError('stop failed')

    try:
        assert not asyncio.run(
            audio_capture._run_recognizer_control_async(state, recognizer, 'pause')
        )
    finally:
        executor.shutdown(wait=True)


def test_dashscope_vad_falling_edge_stops_then_rising_edge_restarts_session():
    class FakeVad:
        def __init__(self):
            self._states = iter((True, False, True))
            self.is_speaking = False
            self.last_confidence = 0.9

        def process_chunk(self, _chunk):
            self.is_speaking = next(self._states)

        def reset(self):
            self.is_speaking = False

    class RecordingRecognizer:
        def __init__(self):
            self.events = []

        def send_audio_frame(self, data):
            self.events.append(('send', data))

        def pause(self):
            self.events.append(('pause', None))

        def resume(self):
            self.events.append(('resume', None))

    executor = ThreadPoolExecutor(max_workers=1)
    state = SimpleNamespace(
        recognition_active=True,
        vad_enabled=True,
        vad_processor=FakeVad(),
        current_asr_backend='qwen_audio3',
        audio_send_generation=3,
        asr_send_executor=executor,
        ensure_asr_send_executor=lambda: None,
        stop_event=asyncio.Event(),
        _vad_pending_samples=np.array([], dtype=np.float32),
        _vad_was_speaking=False,
        _vad_drop_count=0,
    )
    recognizer = RecordingRecognizer()
    frame = b'\x01\x00' * 512
    reads = 0

    async def fake_read(_state):
        nonlocal reads
        reads += 1
        if reads <= 3:
            return frame
        await asyncio.sleep(0.02)
        return None

    try:
        with (
            patch.object(audio_capture, 'read_audio_data', side_effect=fake_read),
            patch.object(audio_capture.config, 'VAD_PRE_SPEECH_DURATION', 0.0),
        ):
            asyncio.run(audio_capture.audio_capture_task(state, recognizer))
    finally:
        executor.shutdown(wait=True)

    event_names = [name for name, _payload in recognizer.events]
    assert event_names == ['send', 'pause', 'resume', 'send']


def test_dashscope_empty_session_is_not_paused_by_stale_vad_state():
    class FakeVad:
        is_speaking = False
        last_confidence = 0.0

        def process_chunk(self, _chunk):
            pass

    class RecordingRecognizer:
        def __init__(self):
            self.pause_calls = 0

        def send_audio_frame(self, _data):
            raise AssertionError('silence must remain gated')

        def pause(self):
            self.pause_calls += 1

    executor = ThreadPoolExecutor(max_workers=1)
    state = SimpleNamespace(
        recognition_active=True,
        vad_enabled=True,
        vad_processor=FakeVad(),
        current_asr_backend='dashscope',
        audio_send_generation=1,
        asr_send_executor=executor,
        ensure_asr_send_executor=lambda: None,
        stop_event=asyncio.Event(),
        _vad_pending_samples=np.array([], dtype=np.float32),
        _vad_was_speaking=True,
        _vad_drop_count=0,
    )
    recognizer = RecordingRecognizer()

    async def fake_read(_state):
        if not getattr(fake_read, 'done', False):
            fake_read.done = True
            return b'\x00\x00' * 512
        return None

    try:
        with patch.object(audio_capture, 'read_audio_data', side_effect=fake_read):
            asyncio.run(audio_capture.audio_capture_task(state, recognizer))
    finally:
        executor.shutdown(wait=True)

    assert recognizer.pause_calls == 0


def test_tail_uses_clamped_runtime_server_window_plus_margin():
    with (
        patch.object(audio_capture.config, 'LOCAL_VAD_SILENCE_DURATION', 9.0),
        patch.object(audio_capture.config, 'ONLINE_VAD_END_BURST_MS', 200),
    ):
        assert audio_capture._vad_gate_tail_seconds('qwen_audio3') == 6.2
