"""识别回调可观测性与并发防护测试（P2-14 / P3-19 / P3-20 / P3-23）。

覆盖审查报告 docs/review-reports/audit-D-full-pipeline-findings.md：
- P2-14: on_result（终句，WS 回调线程）与 _translate_partial_task（中间
  结果，事件循环）无锁并发调用 ensure_secondary_translator —— 修复为
  state 级锁（get_secondary_translator_lock）串行化重建。
- P3-19: 流式翻译任务 except: pass —— 至少留下 logger 告警与堆栈。
- P3-20: on_error 只打日志 —— 致命错误上报到状态接口（AppState 动态
  属性），/api/status 暴露只读字段。
- P3-23: on_result 终句分支的 _finalized_seq / _final_output_version
  自增未持锁 —— 统一移入 _translate_ordering_lock。

所有用例均为 mock/桩实现，不访问真实网络与音频设备。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import recognition_handler as rh
from recognition_handler import (
    VRChatRecognitionCallback,
    get_secondary_translator_lock,
    report_recognition_error,
)
from speech_recognizers.base_speech_recognizer import RecognitionEvent


class _PlainState:
    """不带 MagicMock 自动属性的极简 state 桩。"""

    def __init__(self):
        self.language_detector = MagicMock()
        self.language_detector.detect.return_value = {'language': 'ja'}
        self.subtitles_state = {
            'original': '', 'translated': '', 'reverse_translated': '', 'ongoing': False,
        }
        self.current_asr_backend = 'qwen'
        self.secondary_translator = None
        self.secondary_target_language = None
        self.secondary_deepl_fallback_translator = None
        self.main_loop = None
        self.executor = MagicMock()  # submit 不真执行翻译

    def update_subtitles(self, original, translated, ongoing, reverse_translated=''):
        self.subtitles_state.update({
            'original': original,
            'translated': translated,
            'reverse_translated': reverse_translated,
            'ongoing': ongoing,
        })


# ═══════════════════════════════════════════════════════════════════════
# P2-14: 次翻译器重建的 state 级锁
# ═══════════════════════════════════════════════════════════════════════


class TestSecondaryTranslatorLock:
    def test_lock_is_stable_per_state(self):
        state_a, state_b = _PlainState(), _PlainState()
        assert get_secondary_translator_lock(state_a) is get_secondary_translator_lock(state_a)
        assert get_secondary_translator_lock(state_a) is not get_secondary_translator_lock(state_b)

    def test_lock_serializes_concurrent_rebuild(self):
        """两个线程同时经锁进入重建临界区：任一时刻并发度必须为 1。"""
        state = _PlainState()
        lock = get_secondary_translator_lock(state)
        concurrency = {'now': 0, 'max': 0}

        def rebuild():
            with lock:
                concurrency['now'] += 1
                concurrency['max'] = max(concurrency['max'], concurrency['now'])
                time.sleep(0.05)
                concurrency['now'] -= 1

        threads = [threading.Thread(target=rebuild) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert concurrency['max'] == 1

    def test_partial_task_rebuild_waits_for_lock(self):
        """_translate_partial_task 的 ensure_secondary_translator 需先获取锁。

        partial 任务运行在事件循环上，锁上的阻塞会停住循环线程——这正是
        被测行为（必须等锁）；由定时器线程在观察窗口后释放锁，用时间戳
        证明重建发生在释放之后而非无锁直入。
        """
        state = _PlainState()
        callback = VRChatRecognitionCallback(state)
        lock = get_secondary_translator_lock(state)
        ensure_calls: list = []
        release_time: dict = {}

        def fake_ensure(s, target, cfg):
            ensure_calls.append((target, time.monotonic()))

        def _release():
            release_time['t'] = time.monotonic()
            lock.release()

        async def scenario():
            # 从另一线程持有锁（若在事件循环线程持锁会自锁死锁）
            holder = threading.Thread(target=lock.acquire, daemon=True)
            holder.start()
            holder.join(5)
            release_timer = threading.Timer(0.3, _release)
            with patch.object(rh, 'ensure_secondary_translator', fake_ensure), \
                 patch.object(rh, 'config_from_module', MagicMock(return_value='CFG')), \
                 patch.object(rh, 'config') as mock_config:
                mock_config.TARGET_LANGUAGE = 'ja'
                mock_config.SECONDARY_TARGET_LANGUAGE = None
                mock_config.SMART_TARGET_PRIMARY_ENABLED = False
                mock_config.SMART_TARGET_SECONDARY_ENABLED = False
                mock_config.SOURCE_LANGUAGE = 'auto'
                release_timer.start()
                task = asyncio.create_task(
                    callback._translate_partial_task('こんにちは。', 1, 0, 0, 1, 0),
                )
                # 任务启动后阻塞在锁上（循环线程破停住），定时器释放后恢复
                await asyncio.wait_for(task, timeout=5)

        asyncio.run(scenario())
        assert release_time.get('t') is not None
        assert ensure_calls, '释放锁后 partial 路径应完成重建调用'
        target, entered_at = ensure_calls[0]
        assert target is None  # secondary 目标为 None → ensure(None)
        assert entered_at >= release_time['t'], '锁被持有时 partial 路径不应进入重建'

    def test_on_result_final_rebuild_waits_for_lock(self):
        """on_result 终句路径的 ensure_secondary_translator 需先获取锁。"""
        state = _PlainState()
        callback = VRChatRecognitionCallback(state)
        lock = get_secondary_translator_lock(state)
        ensure_calls: list = []

        def fake_ensure(s, target, cfg):
            ensure_calls.append(target)

        lock.acquire()
        with patch.object(rh, 'ensure_secondary_translator', fake_ensure), \
             patch.object(rh, 'config_from_module', MagicMock(return_value='CFG')), \
             patch.object(rh, 'config') as mock_config:
            mock_config.ENABLE_TRANSLATION = True
            mock_config.TARGET_LANGUAGE = 'ja'
            mock_config.SECONDARY_TARGET_LANGUAGE = 'ko'
            mock_config.SMART_TARGET_PRIMARY_ENABLED = False
            mock_config.SMART_TARGET_SECONDARY_ENABLED = False
            mock_config.SOURCE_LANGUAGE = 'auto'
            mock_config.ENABLE_REVERSE_TRANSLATION = False
            mock_config.TRANSLATION_API_TYPE = 'qwen_mt'
            mock_config.STREAMING_FINAL_DEEPL_MAX_UPDATES = 0
            mock_config.SHOW_ORIGINAL_AND_LANG_TAG = True

            thread = threading.Thread(
                target=lambda: callback.on_result(
                    RecognitionEvent(text='こんにちは', is_final=True),
                ),
            )
            thread.start()
            time.sleep(0.3)
            blocked = not ensure_calls
            lock.release()
            thread.join(5)

        assert blocked, '锁被持有时终句路径不应进入重建'
        assert ensure_calls == ['ko']


# ═══════════════════════════════════════════════════════════════════════
# P3-19: 流式翻译任务异常可观测
# ═══════════════════════════════════════════════════════════════════════


class TestPartialTranslationFailureLogging:
    def test_exception_logged_with_traceback(self, caplog):
        """_translate_partial_task 内部异常：留下 WARNING 日志与堆栈。"""
        state = _PlainState()
        state.language_detector.detect.side_effect = RuntimeError('detect boom')
        callback = VRChatRecognitionCallback(state)

        with patch.object(rh, 'config') as mock_config:
            mock_config.TRANSLATE_PARTIAL_RESULTS = True
            with caplog.at_level(logging.WARNING, logger='recognition_handler'):
                asyncio.run(
                    callback._translate_partial_task('こんにちは。', 1, 0, 0, 1, 0),
                )

        records = [
            r for r in caplog.records
            if 'Partial translation task failed' in r.getMessage()
        ]
        assert records, '流式翻译失败必须留下日志'
        assert records[0].exc_info is not None, '必须携带异常堆栈'
        # finally 分支仍正常回收 inflight 计数
        assert callback._partial_inflight == 0
        assert callback.translating_partial is False


# ═══════════════════════════════════════════════════════════════════════
# P3-20: on_error 上报状态接口
# ═══════════════════════════════════════════════════════════════════════


class TestOnErrorStatusReporting:
    def test_on_error_writes_state_fields(self):
        state = _PlainState()
        callback = VRChatRecognitionCallback(state)
        before = time.time() * 1000.0

        callback.on_error(RuntimeError('quota exceeded'))

        assert state.last_recognition_error == 'RuntimeError: quota exceeded'
        assert state.last_recognition_error_source == 'recognizer'
        assert state.last_recognition_error_at_ms >= before

    def test_report_recognition_error_none_state_noop(self):
        # state 为 None 时静默跳过，绝不抛异常
        report_recognition_error(None, RuntimeError('x'), 'recognizer')

    def test_report_recognition_error_never_raises(self):
        class _BrokenState:
            def __setattr__(self, name, value):
                raise AttributeError('read-only')
        report_recognition_error(_BrokenState(), RuntimeError('x'), 'recognizer')

    def test_api_status_exposes_asr_error_fields(self):
        """ui/app.py /api/status 只读暴露最近一次识别错误。"""
        from app_state import AppState, set_state
        from ui.app import app as flask_app

        state = AppState()
        report_recognition_error(state, RuntimeError('auth failed'), 'recognizer')
        set_state(state)
        try:
            client = flask_app.test_client()
            data = client.get('/api/status').get_json()
            assert data['asr_error'] == 'RuntimeError: auth failed'
            assert data['asr_error_source'] == 'recognizer'
            assert data['asr_error_at_ms'] > 0
        finally:
            set_state(None)

    def test_api_status_without_error_fields_are_none(self):
        from app_state import AppState, set_state
        from ui.app import app as flask_app

        state = AppState()
        set_state(state)
        try:
            client = flask_app.test_client()
            data = client.get('/api/status').get_json()
            assert data['asr_error'] is None
            assert data['asr_error_source'] is None
            assert data['asr_error_at_ms'] is None
        finally:
            set_state(None)


# ═══════════════════════════════════════════════════════════════════════
# P3-23: 终句计数自增移入 _translate_ordering_lock
# ═══════════════════════════════════════════════════════════════════════


class TestFinalizedSeqLockedIncrement:
    def test_final_counters_wait_for_ordering_lock(self):
        """持锁期间终句计数不得前进；释放后恰好自增一次。"""
        state = _PlainState()
        callback = VRChatRecognitionCallback(state)
        lock = callback._translate_ordering_lock
        acquired = threading.Event()
        release = threading.Event()

        def hold_lock():
            with lock:
                acquired.set()
                release.wait(5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert acquired.wait(5)

        try:
            with patch.object(rh, 'config') as mock_config:
                mock_config.ENABLE_TRANSLATION = False
                thread = threading.Thread(
                    target=lambda: callback.on_result(
                        RecognitionEvent(text='こんにちは', is_final=True),
                    ),
                )
                thread.start()
                time.sleep(0.3)
                # 持锁期间：终句分支的计数自增必须也在等锁
                assert callback._finalized_seq == 0
                assert callback._final_output_version == 0
        finally:
            release.set()
            holder.join(5)
            thread.join(5)

        assert callback._finalized_seq == 1
        assert callback._final_output_version == 1

    def test_final_counters_incremented_exactly_once_per_final(self):
        state = _PlainState()
        callback = VRChatRecognitionCallback(state)
        with patch.object(rh, 'config') as mock_config:
            mock_config.ENABLE_TRANSLATION = False
            callback.on_result(RecognitionEvent(text='one', is_final=True))
            callback.on_result(RecognitionEvent(text='two', is_final=True))
        assert callback._finalized_seq == 2
        assert callback._final_output_version == 2
