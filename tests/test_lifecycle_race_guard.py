"""识别器生命周期并发防护回归测试（P2-8 / P2-15）。

覆盖审查报告 docs/review-reports/audit-D-full-pipeline-findings.md：
- P2-8: main.stop_recognition_async 的 pause 段 shield 保护——延迟停止任务被
  取消静音撤销时，仍等待 in-flight pause（含网络 RTT）真正完成后再传播取消；
  取消静音路径先等待 in-flight 停止落地再触发 start/resume（带超时兜底）。
- P2-15: DashscopeSpeechRecognizer 生命周期关键段（start/stop/pause/resume/
  send_audio_frame）加锁串行化；pause/stop 失败不再裸 suppress，至少留下
  logger.warning（服务端会话悬挂风险）。
- P2-15: QwenAudio3SpeechRecognizer.stop 对私有 _running 的防御式判断与
  stop 后置状态校验。

所有用例均为 mock/桩实现，不访问真实网络与音频设备。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from speech_recognizers.dashscope_speech_recognizer import DashscopeSpeechRecognizer
from speech_recognizers.qwen_audio3_speech_recognizer import QwenAudio3SpeechRecognizer

DASHSCOPE_LOGGER = 'speech_recognizers.dashscope_speech_recognizer'
QWEN3_LOGGER = 'speech_recognizers.qwen_audio3_speech_recognizer'


def _import_main():
    # 延迟导入 main，避免在模块加载阶段触发其重型顶层副作用。
    import main as main_module
    return main_module


@pytest.fixture
def main_module():
    return _import_main()


async def _wait_event_async(event: threading.Event, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not event.is_set():
        assert time.monotonic() < deadline, '等待事件超时'
        await asyncio.sleep(0.005)


# ═══════════════════════════════════════════════════════════════════════
# P2-8: stop_recognition_async 的 pause 段 shield 保护
# ═══════════════════════════════════════════════════════════════════════


class TestStopRecognitionShield:
    def test_cancelled_stop_waits_for_inflight_pause(self, main_module):
        """任务取消（取消静音撤销）后，CancelledError 传播前底层 pause 必须完成。"""
        order: list[str] = []
        started, release = threading.Event(), threading.Event()

        state = MagicMock()
        state.recognition_active = True
        state.vad_processor = MagicMock()
        state.executor = ThreadPoolExecutor(max_workers=2)

        def slow_pause():
            order.append('pause_start')
            started.set()
            release.wait(5)
            order.append('pause_end')

        state.recognition_instance.pause = slow_pause

        try:
            async def scenario():
                task = asyncio.create_task(main_module.stop_recognition_async(state))
                await _wait_event_async(started)
                task.cancel()
                # 取消已送达，但 in-flight pause 未完成前，stop 任务必须继续等待
                await asyncio.sleep(0.05)
                assert order == ['pause_start']
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                # 取消向上传播之前，底层 pause 已真正完成（shield 生效）
                assert order == ['pause_start', 'pause_end']
                assert state.recognition_active is False
                # 取消路径跳过 VAD 重置（识别将继续，保留 VAD 历史）
                state.vad_processor.reset.assert_not_called()

            asyncio.run(scenario())
        finally:
            release.set()
            state.executor.shutdown(wait=True)

    def test_normal_stop_still_resets_vad_and_bumps_generation(self, main_module):
        """未被取消的正常停止路径行为不变：VAD 重置 + 代次 bump。"""
        state = MagicMock()
        state.recognition_active = True
        state.vad_processor = MagicMock()
        state.recognition_instance.pause = MagicMock()
        state.executor = ThreadPoolExecutor(max_workers=1)

        try:
            asyncio.run(main_module.stop_recognition_async(state))
        finally:
            state.executor.shutdown(wait=True)

        state.recognition_instance.pause.assert_called_once_with()
        state.vad_processor.reset.assert_called_once()
        state.bump_audio_send_generation.assert_called_once()
        assert state.recognition_active is False

    def test_pause_failure_is_tolerated_with_warning(self, main_module, caplog):
        """pause 抛异常时不再静默：记录 warning 且不阻断停止流程。"""
        state = MagicMock()
        state.recognition_active = True
        state.vad_processor = MagicMock()
        state.recognition_instance.pause = MagicMock(side_effect=RuntimeError('end_session timeout'))
        state.executor = ThreadPoolExecutor(max_workers=1)

        try:
            with caplog.at_level(logging.WARNING, logger='main'):
                asyncio.run(main_module.stop_recognition_async(state))
        finally:
            state.executor.shutdown(wait=True)

        state.vad_processor.reset.assert_called_once()
        state.bump_audio_send_generation.assert_called_once()
        assert any('暂停识别失败' in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════
# P2-8: 取消静音路径先等待 in-flight 停止落地再 start
# ═══════════════════════════════════════════════════════════════════════


def _make_mute_race_state(order, started, release):
    """构造模拟快速闭麦→开麦竞态的 state：pause 含模拟网络 RTT。"""
    state = MagicMock()
    state.last_mute_engaged_time = None
    state.current_asr_backend = 'dashscope'
    state.mute_delay_task = None
    state.recognition_active = True
    state.recognition_started = False

    def slow_pause():
        order.append('pause_start')
        started.set()
        release.wait(5)
        order.append('pause_end')

    state.recognition_instance.pause = slow_pause
    state.recognition_instance.start = lambda: order.append('start')
    state.executor = ThreadPoolExecutor(max_workers=2)
    return state


class TestUnmuteWaitsForInflightStop:
    @patch('main.is_effective_mic_control_enabled', return_value=True)
    @patch('main.osc_manager')
    @patch('main.config')
    def test_unmute_waits_until_pause_lands_before_start(
        self, mock_config, mock_osc, mock_mic, main_module,
    ):
        """快速闭麦→开麦：开麦必须等 in-flight pause 真正完成后才 start。"""
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = False
        mock_config.MUTE_DELAY_SECONDS = 0.05

        order: list[str] = []
        started, release = threading.Event(), threading.Event()
        state = _make_mute_race_state(order, started, release)

        try:
            async def scenario():
                await main_module.handle_mute_change(state, is_muted=True)
                await _wait_event_async(started)
                # 延迟停止已进入 in-flight pause（recognition_active 已置 False）
                assert state.recognition_active is False

                unmute = asyncio.create_task(
                    main_module.handle_mute_change(state, is_muted=False)
                )
                await asyncio.sleep(0.05)
                # unmute 正在等待 in-flight 停止落地，尚未 start
                assert 'start' not in order

                release.set()
                await unmute
                # pause 真正完成后才触发 start，无并发交错
                assert order == ['pause_start', 'pause_end', 'start']

            asyncio.run(scenario())
        finally:
            release.set()
            state.executor.shutdown(wait=True)

    @patch('main.is_effective_mic_control_enabled', return_value=True)
    @patch('main.osc_manager')
    @patch('main.config')
    def test_unmute_proceeds_after_wait_timeout(
        self, mock_config, mock_osc, mock_mic, main_module, monkeypatch,
    ):
        """pause 长时间悬挂（模拟网络故障）时，开麦不被永久阻塞。"""
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = False
        mock_config.MUTE_DELAY_SECONDS = 0.05
        monkeypatch.setattr(main_module, 'INFLIGHT_PAUSE_WAIT_SECONDS', 0.15)

        order: list[str] = []
        started, release = threading.Event(), threading.Event()
        state = _make_mute_race_state(order, started, release)

        try:
            async def scenario():
                await main_module.handle_mute_change(state, is_muted=True)
                await _wait_event_async(started)

                unmute = asyncio.create_task(
                    main_module.handle_mute_change(state, is_muted=False)
                )
                # 最多 ~0.15s 后 unmute 必须返回（超时兜底）
                await asyncio.wait_for(unmute, timeout=3)
                # 未等 pause 完成（悬挂模拟），开麦继续
                assert order == ['pause_start', 'start']

                release.set()
                # 让 in-flight 延迟停止任务在事件循环存活期间落地，
                # 避免事件循环关闭后才完成产生噪音
                await asyncio.wait_for(
                    asyncio.shield(state.mute_delay_task), timeout=3
                )

            asyncio.run(scenario())
        finally:
            release.set()
            state.executor.shutdown(wait=True)

    @patch('main.is_effective_mic_control_enabled', return_value=True)
    @patch('main.osc_manager')
    @patch('main.config')
    def test_unmute_before_delay_fires_skips_stop_and_start(
        self, mock_config, mock_osc, mock_mic, main_module,
    ):
        """延迟窗口内取消静音：延迟停止尚未触发，无需 stop/start。"""
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = False
        mock_config.MUTE_DELAY_SECONDS = 5.0

        state = MagicMock()
        state.last_mute_engaged_time = None
        state.current_asr_backend = 'dashscope'
        state.mute_delay_task = None
        state.recognition_active = True
        state.recognition_started = False
        state.recognition_instance = MagicMock()
        state.executor = ThreadPoolExecutor(max_workers=2)

        try:
            async def scenario():
                await main_module.handle_mute_change(state, is_muted=True)
                assert state.mute_delay_task is not None
                await main_module.handle_mute_change(state, is_muted=False)
                # 延迟任务被取消并等待其退出，识别从未停止 → 不触发 start
                state.recognition_instance.pause.assert_not_called()
                state.recognition_instance.start.assert_not_called()
                assert state.recognition_active is True
                assert state.mute_delay_task.done()

            asyncio.run(scenario())
        finally:
            state.executor.shutdown(wait=True)


# ═══════════════════════════════════════════════════════════════════════
# P2-15: DashscopeSpeechRecognizer 生命周期锁与失败告警
# ═══════════════════════════════════════════════════════════════════════


class _FakeRecognition:
    """dashscope Recognition 最小桩：可控的私有 _running 与 stop 行为。"""

    def __init__(self, *args, **kwargs):
        self._running = True
        self.stop_error: Exception | None = None
        self.stop_keeps_running = False
        self.stop_calls = 0
        self.start_calls = 0
        self.sent_frames: list[bytes] = []

    def drop_running_attr(self) -> None:
        del self._running

    def start(self, **kwargs):
        self.start_calls += 1
        self._running = True

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        if not self.stop_keeps_running:
            self._running = False

    def send_audio_frame(self, data: bytes) -> None:
        self.sent_frames.append(data)


def _make_dashscope(recognition_impl):
    with patch(
        'speech_recognizers.dashscope_speech_recognizer.Recognition',
        return_value=recognition_impl,
    ):
        return DashscopeSpeechRecognizer(MagicMock())


def _make_qwen3(recognition_impl):
    with patch(
        'speech_recognizers.dashscope_speech_recognizer.Recognition',
        return_value=recognition_impl,
    ):
        return QwenAudio3SpeechRecognizer(MagicMock())


class TestDashscopeLifecycleLock:
    def test_pause_serializes_against_start_and_send(self):
        """pause 关键段持锁期间，并发的 start/send 必须被阻塞串行化。"""
        rec = MagicMock()
        order: list[str] = []
        inside, release = threading.Event(), threading.Event()

        def send_frame(data):
            # pause 的静音帧为 3200 字节全零，测试自发帧仅为 2 字节
            if len(data) > 2:
                order.append('silence_sent')

        def stop():
            order.append('stop_enter')
            inside.set()
            release.wait(5)
            order.append('stop_exit')

        rec.send_audio_frame.side_effect = send_frame
        rec.stop.side_effect = stop

        recognizer = _make_dashscope(rec)

        def do_pause():
            recognizer.pause()

        def do_start():
            recognizer.start()
            order.append('start_done')

        def do_send():
            recognizer.send_audio_frame(b'\x00\x01')
            order.append('send_done')

        t_pause = threading.Thread(target=do_pause)
        t_pause.start()
        assert inside.wait(5), 'pause 未进入 stop 关键段'
        t_start = threading.Thread(target=do_start)
        t_send = threading.Thread(target=do_send)
        t_start.start()
        t_send.start()

        time.sleep(0.15)
        # 锁保护：pause 完成前 start/send 均不得触碰 SDK
        assert 'start_done' not in order
        assert 'send_done' not in order

        release.set()
        t_pause.join(5)
        t_start.join(5)
        t_send.join(5)
        assert not any(t.is_alive() for t in (t_pause, t_start, t_send))
        assert order[:3] == ['silence_sent', 'stop_enter', 'stop_exit']
        assert sorted(order[3:]) == ['send_done', 'start_done']

    def test_resume_serialized_against_pause(self):
        """resume（复用 start）与 pause 在锁上串行，不并发操作 SDK。"""
        rec = MagicMock()
        order: list[str] = []
        inside, release = threading.Event(), threading.Event()

        def stop():
            order.append('stop_enter')
            inside.set()
            release.wait(5)
            order.append('stop_exit')

        rec.stop.side_effect = stop

        recognizer = _make_dashscope(rec)

        t_pause = threading.Thread(target=recognizer.pause)
        t_pause.start()
        assert inside.wait(5)

        t_resume = threading.Thread(target=recognizer.resume)
        t_resume.start()
        time.sleep(0.1)
        assert 'stop_exit' not in order

        release.set()
        t_pause.join(5)
        t_resume.join(5)
        assert not t_pause.is_alive() and not t_resume.is_alive()
        assert order == ['stop_enter', 'stop_exit']

    def test_quick_mute_unmute_concurrent_no_sdk_overlap(self):
        """快速闭麦→开麦并发压力：多线程并发 pause/start/send，SDK 方法不得交错。"""
        rec = MagicMock()
        monitor = threading.Lock()
        active = {'count': 0}
        violations = {'count': 0}

        def guarded(*args, **kwargs):
            with monitor:
                active['count'] += 1
                if active['count'] > 1:
                    violations['count'] += 1
            try:
                time.sleep(0.001)
            finally:
                with monitor:
                    active['count'] -= 1

        rec.start.side_effect = guarded
        rec.stop.side_effect = guarded
        rec.send_audio_frame.side_effect = guarded

        recognizer = _make_dashscope(rec)

        threads = []
        for i in range(6):
            if i % 2 == 0:
                threads.append(threading.Thread(target=recognizer.pause))
            else:
                threads.append(threading.Thread(target=recognizer.start))

        def sender():
            for _ in range(50):
                recognizer.send_audio_frame(b'\x00\x00')

        for _ in range(3):
            threads.append(threading.Thread(target=sender))

        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
        assert all(not t.is_alive() for t in threads)
        assert violations['count'] == 0, 'SDK 关键段出现并发交错'


class TestDashscopeFailureLogging:
    def test_pause_failures_warn_and_do_not_raise(self, caplog):
        """pause 中静音帧发送/stop 失败不再裸 suppress：留下告警且不抛出。"""
        rec = _FakeRecognition()

        def raise_on_send(data):
            raise RuntimeError('send broken')

        rec.send_audio_frame = raise_on_send
        rec.stop_error = RuntimeError('server hang')

        recognizer = _make_dashscope(rec)
        with caplog.at_level(logging.WARNING, logger=DASHSCOPE_LOGGER):
            recognizer.pause()  # 不应抛出

        messages = [r.message for r in caplog.records]
        assert any('发送静音帧失败' in m for m in messages)
        assert any('暂停时停止会话失败' in m for m in messages)
        assert any('server hang' in m for m in messages)

    def test_stop_failure_warns_and_reraises(self, caplog):
        """stop 失败：告警留痕（服务端会话悬挂风险）并继续向上抛。"""
        rec = _FakeRecognition()
        rec.stop_error = RuntimeError('finish-task failed')

        recognizer = _make_dashscope(rec)
        with caplog.at_level(logging.WARNING, logger=DASHSCOPE_LOGGER):
            with pytest.raises(RuntimeError, match='finish-task failed'):
                recognizer.stop()

        assert any('停止识别会话失败' in r.message for r in caplog.records)

    def test_pause_sends_silence_then_stops(self):
        rec = _FakeRecognition()
        recognizer = _make_dashscope(rec)
        recognizer.pause()
        assert len(rec.sent_frames) == 1
        # 16kHz/单声道/100ms/int16 静音帧
        assert rec.sent_frames[0] == b'\x00' * 3200
        assert rec.stop_calls == 1


# ═══════════════════════════════════════════════════════════════════════
# P2-15: QwenAudio3SpeechRecognizer.stop 防御式 _running + 后置校验
# ═══════════════════════════════════════════════════════════════════════


class TestQwenAudio3StopDefensive:
    def test_stop_runs_when_running(self):
        rec = _FakeRecognition()
        recognizer = _make_qwen3(rec)
        recognizer.stop()
        assert rec.stop_calls == 1
        assert rec._running is False

    def test_stop_skipped_when_running_attr_missing(self, caplog):
        """SDK 私有属性 _running 缺失时防御式跳过，不抛异常、不误调 stop。"""
        rec = _FakeRecognition()
        rec.drop_running_attr()
        recognizer = _make_qwen3(rec)

        with caplog.at_level(logging.WARNING, logger=QWEN3_LOGGER):
            recognizer.stop()  # 不应抛出

        assert rec.stop_calls == 0
        assert not any('仍为运行状态' in r.message for r in caplog.records)

    def test_stop_warns_when_session_still_running_afterwards(self, caplog):
        """stop 正常返回但 _running 仍为真：后置校验必须留下告警。"""
        rec = _FakeRecognition()
        rec.stop_keeps_running = True
        recognizer = _make_qwen3(rec)

        with caplog.at_level(logging.WARNING, logger=QWEN3_LOGGER):
            recognizer.stop()

        assert rec.stop_calls == 1
        assert any('仍为运行状态' in r.message for r in caplog.records)

    def test_stop_no_warning_when_session_cleanly_stopped(self, caplog):
        rec = _FakeRecognition()
        recognizer = _make_qwen3(rec)
        with caplog.at_level(logging.WARNING, logger=QWEN3_LOGGER):
            recognizer.stop()
        assert not any('仍为运行状态' in r.message for r in caplog.records)

    def test_stop_failure_warns_and_reraises(self, caplog):
        rec = _FakeRecognition()
        rec.stop_error = RuntimeError('ws closed')
        recognizer = _make_qwen3(rec)
        with caplog.at_level(logging.WARNING, logger=DASHSCOPE_LOGGER):
            with pytest.raises(RuntimeError, match='ws closed'):
                recognizer.stop()
        assert any('停止识别会话失败' in r.message for r in caplog.records)

    def test_stop_serialized_against_inflight_pause(self):
        """qwen_audio3 stop 复用基类锁：in-flight pause 期间 stop 被串行化。"""
        rec = MagicMock()
        order: list[str] = []
        inside, release = threading.Event(), threading.Event()

        def stop():
            order.append('stop_enter')
            inside.set()
            release.wait(5)
            order.append('stop_exit')

        rec.stop.side_effect = stop
        recognizer = _make_qwen3(rec)

        t_pause = threading.Thread(target=recognizer.pause)
        t_pause.start()
        assert inside.wait(5)

        def do_stop():
            recognizer.stop()
            order.append('qwen_stop_done')

        t_stop = threading.Thread(target=do_stop)
        t_stop.start()
        time.sleep(0.1)
        assert 'qwen_stop_done' not in order

        release.set()
        t_pause.join(5)
        t_stop.join(5)
        assert not t_pause.is_alive() and not t_stop.is_alive()
        # pause 的 stop 与 qwen3.stop 触发的 SDK stop 两次串行执行
        assert order == [
            'stop_enter', 'stop_exit',   # pause 持锁期间
            'stop_enter', 'stop_exit',   # qwen3.stop 在锁释放后
            'qwen_stop_done',
        ]

    @patch('speech_recognizers.qwen_audio3_speech_recognizer.build_asr_context_text', return_value='')
    @patch('speech_recognizers.qwen_audio3_speech_recognizer.refresh_system_proxy_env')
    def test_start_serialized_against_inflight_pause(self, _proxy, _ctx):
        """qwen_audio3 start 同样持锁：不与 in-flight pause 并发操作 SDK。"""
        rec = MagicMock()
        order: list[str] = []
        inside, release = threading.Event(), threading.Event()

        def stop():
            order.append('stop_enter')
            inside.set()
            release.wait(5)
            order.append('stop_exit')

        rec.stop.side_effect = stop
        rec.start.side_effect = lambda **kwargs: order.append('sdk_start')
        recognizer = _make_qwen3(rec)

        t_pause = threading.Thread(target=recognizer.pause)
        t_pause.start()
        assert inside.wait(5)

        def do_start():
            recognizer.start()
            order.append('start_done')

        t_start = threading.Thread(target=do_start)
        t_start.start()
        time.sleep(0.1)
        assert 'sdk_start' not in order

        release.set()
        t_pause.join(5)
        t_start.join(5)
        assert not t_pause.is_alive() and not t_start.is_alive()
        assert order == ['stop_enter', 'stop_exit', 'sdk_start', 'start_done']
