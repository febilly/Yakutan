"""生命周期与回调可观测性测试（P2-9 / P2-13 / P3-26）。

覆盖审查报告 docs/review-reports/audit-D-full-pipeline-findings.md：
- P2-13: main() 初始化段纳入 try/finally —— 启动中途失败（无麦克风、后端
  不可用等）时同样执行清理（本地模型卸载 / OSC 停止 / mute 回调清除 /
  executor 关闭），且正常路径语义不变。
- P2-9: 静音回调经 run_coroutine_threadsafe 投递后，future 异常不再被
  吞掉 —— done callback 记录 logger.warning 并上报状态接口。
- P3-26: create_task(ipc_client.start()) 保存引用，异常经 done callback
  记录并上报，不再只以 "never retrieved" 形式丢失。

所有用例均为 mock/桩实现，不访问真实网络与音频设备。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _import_main():
    # 延迟导入 main，避免在模块加载阶段触发其重型顶层副作用。
    import main as main_module
    return main_module


@pytest.fixture
def main_module():
    return _import_main()


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, '等待条件超时'
        await asyncio.sleep(interval)


def _make_mock_config(**overrides):
    cfg = MagicMock()
    cfg.VAD_ENABLED = False
    cfg.ENABLE_TRANSLATION = False
    cfg.ENABLE_HOT_WORDS = False
    cfg.IPC_ENABLED = False
    cfg.SAMPLE_RATE = 16000
    cfg.FORMAT_PCM = 8
    cfg.SOURCE_LANGUAGE = 'ja'
    cfg.ENABLE_VAD = False
    cfg.VAD_THRESHOLD = 0.5
    cfg.KEEPALIVE_INTERVAL = 0
    cfg.ENABLE_MIC_CONTROL = False
    cfg.PREFERRED_ASR_BACKEND = 'qwen'
    cfg.VALID_ASR_BACKENDS = ['qwen']
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class _MainHarness:
    """把 main() 的全部外部依赖打桩，驱动其完整（或失败）生命周期。"""

    def __init__(self, main_module, *, ipc_enabled: bool = False,
                 ipc_start_exception: Optional[Exception] = None,
                 init_audio_stream_exception: Optional[Exception] = None):
        self.main = main_module
        self.ipc_enabled = ipc_enabled
        self.ipc_start_exception = ipc_start_exception
        self.init_audio_stream_exception = init_audio_stream_exception

        self.lifecycle: list = []
        self.captured_state: dict = {}
        self.ipc_client: Optional[MagicMock] = None
        self.cleanup_calls: dict = {}

        self.mock_config = _make_mock_config(IPC_ENABLED=ipc_enabled)

        self.release_local_engines = MagicMock(side_effect=self._track('release_local_engines'))
        self.reinit_translator = MagicMock(side_effect=self._track('reinitialize_translator'))
        self.clear_contexts = MagicMock(side_effect=self._track('clear_translation_contexts'))
        self.prewarm = MagicMock(return_value=0)
        self.osc_start_server = AsyncMock(side_effect=self._track_async('osc.start_server'))
        self.osc_stop_server = AsyncMock(side_effect=self._track_async('osc.stop_server'))
        self.osc_clear_mute = MagicMock(side_effect=self._track('osc.clear_mute_callback'))
        self.osc_reset = MagicMock(side_effect=self._track('osc.reset_runtime_state'))
        self.close_audio_stream = AsyncMock(side_effect=self._track_async('close_audio_stream'))
        self.init_audio_stream = AsyncMock(
            side_effect=self._make_init_audio_stream_side_effect(),
        )
        self.create_recognizer = MagicMock(return_value=MagicMock(name='recognizer'))
        self.select_backend = MagicMock(return_value='qwen')

    # ---- 记账 ----

    def _track(self, name):
        def _call(*args, **kwargs):
            self.cleanup_calls[name] = self.cleanup_calls.get(name, 0) + 1
        return _call

    def _track_async(self, name):
        async def _call(*args, **kwargs):
            self.cleanup_calls[name] = self.cleanup_calls.get(name, 0) + 1
        return _call

    def _make_init_audio_stream_side_effect(self):
        harness = self

        async def _call(state):
            harness.cleanup_calls['init_audio_stream'] = (
                harness.cleanup_calls.get('init_audio_stream', 0) + 1
            )
            if harness.init_audio_stream_exception is not None:
                raise harness.init_audio_stream_exception

        return _call

    # ---- IPC 桩 ----

    def _make_ipc_client(self):
        client = MagicMock(name='ipc_client')
        if self.ipc_start_exception is not None:
            client.start = AsyncMock(side_effect=self.ipc_start_exception)
        else:
            started = asyncio.Event()

            async def _ok_start():
                await started.wait()
            client.start = AsyncMock(side_effect=_ok_start)
        client.stop = AsyncMock(side_effect=self._track_async('ipc.stop'))
        return client

    # ---- 打桩上下文 ----

    @staticmethod
    async def _fake_capture_task(state, instance):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass

    def _make_osc_manager(self):
        osc = MagicMock(name='osc_manager')
        osc.start_server = self.osc_start_server
        osc.stop_server = self.osc_stop_server
        osc.clear_mute_callback = self.osc_clear_mute
        osc.reset_runtime_state = self.osc_reset
        osc.set_mute_callback = MagicMock()
        osc.clear_ipc_client = MagicMock()
        osc.probe_initial_mute_state = AsyncMock()
        return osc

    def _patch_context(self):
        m = self.main
        harness = self

        real_set_state = m.set_state

        def _capture_set_state(state):
            harness.captured_state['state'] = state
            real_set_state(state)

        class _FakeIPCClient:
            def __init__(self, translator=None):
                harness.ipc_client = harness._make_ipc_client()

            def __getattr__(self, name):
                return getattr(harness.ipc_client, name)

        return patch.multiple(
            m,
            config=self.mock_config,
            set_state=_capture_set_state,
            select_backend=self.select_backend,
            _create_language_detector=MagicMock(return_value=MagicMock(name='detector')),
            create_recognizer=self.create_recognizer,
            init_audio_stream=self.init_audio_stream,
            close_audio_stream=self.close_audio_stream,
            audio_capture_task=self._fake_capture_task,
            refresh_system_proxy_env=MagicMock(return_value={}),
            print_proxy_info=MagicMock(),
            init_dashscope_api_key=MagicMock(),
            release_local_engines=self.release_local_engines,
            reinitialize_translator=self.reinit_translator,
            prewarm_local_engines=self.prewarm,
            clear_translation_contexts=self.clear_contexts,
            is_effective_mic_control_enabled=MagicMock(return_value=False),
            osc_manager=self._make_osc_manager(),
            IPCClient=_FakeIPCClient,
        )

    def emit(self, lifecycle, recognition_active=None):
        self.lifecycle.append((lifecycle, recognition_active))

    # ---- 场景 ----

    def run_until_init_failure(self):
        """运行 main()，预期初始化中途抛异常并完成清理后传播。"""

        async def _scenario():
            # noqa 触发初始化失败的完整路径（含 finally 清理）
            return await self.main.main(lifecycle_callback=self.emit)

        return _scenario

    def run_full_lifecycle(self):
        """运行 main() 至 running，随后置位 stop_event 等待正常停机。"""

        async def _scenario():
            task = asyncio.create_task(
                self.main.main(lifecycle_callback=self.emit),
            )
            await _wait_until(
                lambda: self.lifecycle and self.lifecycle[-1][0] == 'running',
            )
            self.captured_state['state'].stop_event.set()
            await task

        return _scenario


# ═══════════════════════════════════════════════════════════════════════
# P2-13: 初始化段纳入 try/finally —— 启动失败执行清理
# ═══════════════════════════════════════════════════════════════════════


class TestInitFailureCleanup:
    def test_init_failure_runs_full_cleanup(self, main_module):
        """init_audio_stream 抛异常时：清理路径完整执行且异常向上传播。"""
        harness = _MainHarness(
            main_module,
            init_audio_stream_exception=RuntimeError('no microphone'),
        )
        with harness._patch_context():
            with pytest.raises(RuntimeError, match='no microphone'):
                asyncio.run(harness.run_until_init_failure()())

        calls = harness.cleanup_calls
        # 本地模型卸载 + 翻译上下文清理（本地模型泄漏是 P2-13 的核心损害）
        assert calls.get('release_local_engines', 0) >= 1
        assert calls.get('clear_translation_contexts', 0) >= 1
        # OSC：mute 回调清除 + 接收服务停止
        assert calls.get('osc.clear_mute_callback', 0) == 1
        assert calls.get('osc.reset_runtime_state', 0) >= 1
        assert calls.get('osc.stop_server', 0) == 1
        # 识别实例 stop + 音频流关闭
        assert harness.create_recognizer.return_value.stop.called
        assert calls.get('close_audio_stream', 0) == 1
        # 生命周期回调仍完整收尾
        lifecycles = [item[0] for item in harness.lifecycle]
        assert lifecycles[0] == 'starting'
        assert lifecycles[-1] == 'stopped'
        assert 'stopping' in lifecycles

        # executor 已关闭（资源不再悬挂）
        state = harness.captured_state['state']
        assert state.executor._shutdown
        assert state.audio_executor._shutdown
        assert state.asr_send_executor._shutdown

    def test_early_init_failure_still_cleans(self, main_module):
        """更早的初始化步骤（翻译器构建）失败时同样执行清理。"""
        harness = _MainHarness(main_module)
        # ENABLE_TRANSLATION=True 才会走 reinitialize_translator
        harness.mock_config.ENABLE_TRANSLATION = True
        harness.reinit_translator = MagicMock(
            side_effect=RuntimeError('translator init boom'),
        )

        with harness._patch_context():
            with pytest.raises(RuntimeError, match='translator init boom'):
                asyncio.run(harness.run_until_init_failure()())

        assert harness.cleanup_calls.get('release_local_engines', 0) >= 1
        assert harness.cleanup_calls.get('osc.clear_mute_callback', 0) == 1
        assert harness.cleanup_calls.get('osc.stop_server', 0) == 1
        assert harness.cleanup_calls.get('close_audio_stream', 0) == 1

    def test_ipc_cleanup_on_init_failure(self, main_module):
        """IPC 已创建后初始化失败：ipc_client.stop 被调用。"""
        harness = _MainHarness(
            main_module,
            ipc_enabled=True,
            init_audio_stream_exception=RuntimeError('device lost'),
        )
        with harness._patch_context():
            with pytest.raises(RuntimeError, match='device lost'):
                asyncio.run(harness.run_until_init_failure()())

        assert harness.cleanup_calls.get('ipc.stop', 0) == 1
        assert harness.cleanup_calls.get('osc.stop_server', 0) == 1

    def test_normal_path_semantics_unchanged(self, main_module):
        """正常启动→停机：生命周期序列与清理调用与改造前语义一致。"""
        harness = _MainHarness(main_module)
        with harness._patch_context():
            asyncio.run(harness.run_full_lifecycle()())

        lifecycles = [item[0] for item in harness.lifecycle]
        assert lifecycles[0] == 'starting'
        assert lifecycles.count('running') == 1
        assert lifecycles[-1] == 'stopped'
        # 停机段：stop body 与 finally 各 emit 一次 stopping（既有行为）
        assert lifecycles.count('stopping') == 2
        # 清理路径照常执行
        assert harness.cleanup_calls.get('release_local_engines', 0) == 1
        assert harness.cleanup_calls.get('osc.clear_mute_callback', 0) == 1
        assert harness.cleanup_calls.get('osc.stop_server', 0) == 1
        assert harness.cleanup_calls.get('close_audio_stream', 0) == 1
        assert harness.create_recognizer.return_value.stop.called
        state = harness.captured_state['state']
        assert state.executor._shutdown
        # OSC 接收服务正常启动过
        assert harness.cleanup_calls.get('osc.start_server', 0) == 1
        # mute 回调在初始化中设置过
        assert harness.cleanup_calls.get('init_audio_stream', 0) == 1


# ═══════════════════════════════════════════════════════════════════════
# P2-14（main 侧）: 配置热更新重建与识别回调共用同一把 state 级锁
# ═══════════════════════════════════════════════════════════════════════


class TestHotUpdateSharesSecondaryLock:
    def test_reinitialize_translator_compat_waits_for_lock(self, main_module):
        """锁被持有时，配置热更新路径不应进入次翻译器重建。"""
        from recognition_handler import get_secondary_translator_lock

        from types import SimpleNamespace

        # 注：不能用 MagicMock —— 其自动属性会让惰性锁创建失效
        state = SimpleNamespace(translator=MagicMock(), main_loop=None)
        lock = get_secondary_translator_lock(state)
        entered: list = []

        def slow_update(s, cfg):
            entered.append('update')
            time.sleep(0.05)

        with patch.object(main_module, 'get_state', return_value=state), \
             patch.object(main_module, 'config') as mock_config, \
             patch.object(main_module, 'config_from_module', MagicMock(return_value='CFG')), \
             patch.object(main_module, '_is_primary_translator_config_changed', return_value=False), \
             patch.object(main_module, 'update_secondary_translator', slow_update), \
             patch.object(main_module, '_refresh_ipc_translator_reference'), \
             patch.object(main_module, 'release_local_engines'):
            mock_config.ENABLE_TRANSLATION = True
            lock.acquire()
            thread = threading.Thread(
                target=main_module.reinitialize_translator_compat,
            )
            thread.start()
            time.sleep(0.2)
            blocked = not entered
            lock.release()
            thread.join(5)

        assert blocked, '锁被持有时配置热更新路径不应进入重建'
        assert entered == ['update']


# ═══════════════════════════════════════════════════════════════════════
# P2-9: 静音回调 future 异常可观测
# ═══════════════════════════════════════════════════════════════════════


class TestMuteCallbackFutureObservability:
    def _run_scenario(self, main_module, coroutine_factory):
        """在事件循环内安装 _make_mute_callback，从旁路线程投递一次事件。"""
        posted = threading.Event()

        async def scenario():
            loop = asyncio.get_running_loop()
            state = MagicMock(name='state')
            state.main_loop = loop
            callback = main_module._make_mute_callback(state)

            def _post():
                callback(True)
                posted.set()

            with patch.object(main_module, 'handle_mute_change', coroutine_factory), \
                 patch.object(main_module, 'report_recognition_error') as report:
                thread = threading.Thread(target=_post)
                thread.start()
                await asyncio.sleep(0.3)
                thread.join(timeout=2)
                return report

        return asyncio.run(scenario()), posted

    def test_mute_coroutine_exception_logged_and_reported(self, main_module, caplog):
        """回调协程抛异常：logger.warning + 状态接口上报，不再静默。"""
        async def failing_mute(state, is_muted):
            raise RuntimeError('resume boom')

        with caplog.at_level(logging.WARNING, logger='main'):
            report, posted = self._run_scenario(main_module, failing_mute)

        assert posted.is_set()
        assert report.call_count == 1
        args, _kwargs = report.call_args
        assert 'resume boom' in str(args[1])
        assert args[2] == 'mute_callback'
        assert any(
            '处理静音状态变化失败' in record.getMessage()
            for record in caplog.records
        )

    def test_mute_coroutine_success_not_reported(self, main_module, caplog):
        """回调协程正常完成：不产生告警、不上报。"""
        async def ok_mute(state, is_muted):
            return None

        with caplog.at_level(logging.WARNING, logger='main'):
            report, posted = self._run_scenario(main_module, ok_mute)

        assert posted.is_set()
        report.assert_not_called()
        assert not any(
            '处理静音状态变化失败' in record.getMessage()
            for record in caplog.records
        )

    def test_dead_loop_warns_and_skips(self, main_module, caplog):
        """主事件循环不可用：直接告警且不再投递。"""
        state = MagicMock(name='state')
        state.main_loop = None
        callback = main_module._make_mute_callback(state)
        with caplog.at_level(logging.WARNING, logger='main'):
            callback(True)
        assert any('主事件循环不可用' in r.getMessage() for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════
# P3-26: IPC 启动任务保存引用 + done callback 记录异常
# ═══════════════════════════════════════════════════════════════════════


class TestIpcStartTaskObservability:
    def test_ipc_start_failure_logged_and_reported(self, main_module, caplog):
        """ipc_client.start() 异常退出：记录错误日志并上报状态接口。"""
        harness = _MainHarness(
            main_module,
            ipc_enabled=True,
            ipc_start_exception=ValueError('ipc handshake failed'),
        )

        def _scenario():
            return harness.run_full_lifecycle()()

        # IPC 异常会经真实 report_recognition_error 写入 AppState，无副作用
        with harness._patch_context(), \
             caplog.at_level(logging.ERROR, logger='main'):
            asyncio.run(_scenario())

        assert any(
            'IPC 客户端异常退出' in record.getMessage()
            for record in caplog.records
        )
        # IPC 异常不阻断主服务启动
        assert [item[0] for item in harness.lifecycle][-1] == 'stopped'
        # 停机时 ipc_client.stop 仍被调用
        assert harness.cleanup_calls.get('ipc.stop', 0) == 1
