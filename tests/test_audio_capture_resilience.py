"""采集任务故障路径回归测试。

覆盖两个 P1 缺陷：
- verbose 诊断路径引用未定义变量，异常处理路径二次出错冲垮整个采集任务；
- VAD 判停等待发送队列排空时无超时，且 sender worker 一遇异常即死，
  两者都会让采集主循环永久卡死或静默退出。
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import audio_capture

FRAME = b'\x01\x00' * 512


def _make_state(**overrides):
    executor = ThreadPoolExecutor(max_workers=1)
    state = SimpleNamespace(
        recognition_active=True,
        vad_enabled=True,
        current_asr_backend='dashscope',
        audio_send_generation=1,
        asr_send_executor=executor,
        ensure_asr_send_executor=lambda: None,
        stop_event=asyncio.Event(),
        _vad_pending_samples=np.array([], dtype=np.float32),
        _vad_was_speaking=False,
        _vad_drop_count=0,
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state, executor


def _reader(frames):
    """按顺序返回 frames，耗尽后留给发送侧一个排空的间隙再结束。"""
    pending = list(frames)

    async def fake_read(_state):
        if pending:
            return pending.pop(0)
        await asyncio.sleep(0.02)
        return None

    return fake_read


class _SpeechVad:
    def __init__(self, states=None):
        self._states = iter(states) if states else None
        self.is_speaking = True
        self.last_confidence = 0.9

    def process_chunk(self, _chunk):
        if self._states is not None:
            self.is_speaking = next(self._states)

    def reset(self):
        self.is_speaking = False


class _RecordingRecognizer:
    def __init__(self, fail_on=None):
        self.events = []
        self.fail_on = fail_on or ()
        self.calls = 0

    def send_audio_frame(self, data):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError('send boom')
        self.events.append(data)

    def pause(self):
        self.events.append('pause')

    def resume(self):
        self.events.append('resume')


def test_verbose_diagnostic_does_not_kill_capture_task(capsys):
    state, executor = _make_state(vad_processor=_SpeechVad(), _vad_was_speaking=True)
    recognizer = _RecordingRecognizer()
    frames = [FRAME] * 5

    try:
        with (
            patch.object(audio_capture, 'read_audio_data', side_effect=_reader(frames)),
            patch.object(audio_capture.config, 'ENABLE_VAD_GATING_VERBOSE', True),
            patch.object(audio_capture.config, 'VAD_PRE_SPEECH_DURATION', 0.0),
        ):
            asyncio.run(audio_capture.audio_capture_task(state, recognizer))
    finally:
        executor.shutdown(wait=True)

    out = capsys.readouterr().out
    assert 'Audio capture error' not in out
    assert '[VAD] diag:' in out
    assert recognizer.events == frames


def test_vad_processing_exception_is_contained(capsys):
    class ExplodingVad(_SpeechVad):
        def process_chunk(self, _chunk):
            raise RuntimeError('vad boom')

    state, executor = _make_state(vad_processor=ExplodingVad(), _vad_was_speaking=True)
    recognizer = _RecordingRecognizer()

    try:
        with (
            patch.object(audio_capture, 'read_audio_data', side_effect=_reader([FRAME] * 3)),
            patch.object(audio_capture.config, 'ENABLE_VAD_GATING_VERBOSE', True),
            patch.object(audio_capture.config, 'VAD_PRE_SPEECH_DURATION', 0.0),
        ):
            asyncio.run(audio_capture.audio_capture_task(state, recognizer))
    finally:
        executor.shutdown(wait=True)

    out = capsys.readouterr().out
    assert 'Audio capture error' not in out
    assert '[VAD] ⚠ 处理异常（静默）' in out


def test_sender_worker_survives_failing_frame():
    """worker 内部异常必须只跳过该帧，不能让 worker 死亡。

    注意不能靠 send_audio_frame 抛异常来构造场景——它自身已吞掉发送异常，
    这里直接让帧处理函数首帧失败，才能覆盖 worker 的保护逻辑。
    """
    state, executor = _make_state(vad_processor=_SpeechVad(), _vad_was_speaking=True)
    recognizer = _RecordingRecognizer()
    frames = [FRAME] * 4
    original = audio_capture._send_queue_payload
    calls = {'count': 0}

    async def flaky_payload(state_arg, recognizer_arg, generation, payload):
        calls['count'] += 1
        if calls['count'] == 1:
            raise RuntimeError('payload boom')
        await original(state_arg, recognizer_arg, generation, payload)

    try:
        with (
            patch.object(audio_capture, 'read_audio_data', side_effect=_reader(frames)),
            patch.object(audio_capture, '_send_queue_payload', flaky_payload),
            patch.object(audio_capture.config, 'VAD_PRE_SPEECH_DURATION', 0.0),
        ):
            # worker 若被首帧异常杀死，后续帧不会送达，且收尾 await sender_task
            # 会把异常抛出来，这里会直接失败
            asyncio.run(
                asyncio.wait_for(
                    audio_capture.audio_capture_task(state, recognizer),
                    timeout=10,
                )
            )
    finally:
        executor.shutdown(wait=True)

    assert recognizer.events == frames[1:]


def test_join_timeout_drains_queue_instead_of_blocking(capsys):
    state, executor = _make_state(
        vad_processor=_SpeechVad(states=[True, True, False]),
        _vad_was_speaking=True,
    )
    recognizer = _RecordingRecognizer()

    async def never_completes(*_args, **_kwargs):
        await asyncio.Event().wait()

    try:
        with (
            patch.object(audio_capture, 'read_audio_data', side_effect=_reader([FRAME] * 3)),
            patch.object(audio_capture, 'send_audio_frame_async', never_completes),
            patch.object(audio_capture, 'SEND_QUEUE_JOIN_TIMEOUT_SECONDS', 0.05),
            patch.object(audio_capture.config, 'VAD_PRE_SPEECH_DURATION', 0.0),
        ):
            asyncio.run(
                asyncio.wait_for(
                    audio_capture.audio_capture_task(state, recognizer),
                    timeout=10,
                )
            )
    finally:
        executor.shutdown(wait=True)

    out = capsys.readouterr().out
    assert '未排空，丢弃余量' in out
    # 超时降级后仍继续结束会话，没有被永久卡住
    assert 'pause' in recognizer.events


def test_drain_send_queue_releases_join():
    async def scenario():
        queue = asyncio.Queue()
        for index in range(3):
            queue.put_nowait(index)

        assert audio_capture._drain_send_queue(queue) == 3
        assert audio_capture._drain_send_queue(queue) == 0
        await asyncio.wait_for(queue.join(), timeout=0.5)

    asyncio.run(scenario())


def test_send_failure_is_logged_not_swallowed(caplog):
    class FailingRecognizer:
        def send_audio_frame(self, _data):
            raise RuntimeError('dead connection')

    state, executor = _make_state()

    try:
        with (
            patch.object(audio_capture, '_last_send_fail_log_at', 0.0),
            caplog.at_level(logging.WARNING),
        ):
            asyncio.run(
                audio_capture.send_audio_frame_async(state, FailingRecognizer(), FRAME)
            )
    finally:
        executor.shutdown(wait=True)

    assert 'ASR 发送音频帧失败' in caplog.text
    assert 'dead connection' in caplog.text
