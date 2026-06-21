"""Tests for the double-mute-to-clear chatbox feature.

Covers:
- recognition_handler.discard_pending_outputs / resume_outputs / on_result discard window
- main.handle_mute_change double-mute detection logic
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from recognition_handler import VRChatRecognitionCallback
from speech_recognizers.base_speech_recognizer import RecognitionEvent


# ═══════════════════════════════════════════════════════════════════════
# recognition_handler: discard / resume / on_result discard window
# ═══════════════════════════════════════════════════════════════════════


class TestDiscardPendingOutputs:
    @pytest.fixture
    def cb(self):
        return VRChatRecognitionCallback(MagicMock())

    @patch("recognition_handler.config")
    def test_discard_sets_flag_and_deadline(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        before = time.monotonic()
        cb.discard_pending_outputs()
        after = time.monotonic()
        assert cb._discard_results is True
        assert before + 2.0 <= cb._discard_deadline <= after + 2.0

    @patch("recognition_handler.config")
    def test_discard_zero_window_does_not_block_next_result(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 0.0
        cb.discard_pending_outputs()
        assert cb._discard_results is True
        # window=0 时下一次 on_result 应立即解除丢弃（不再阻塞）
        cb.state.update_subtitles.reset_mock()
        cb.on_result(RecognitionEvent(text="", is_final=True, raw={}))
        assert cb._discard_results is False

    @patch("recognition_handler.config")
    def test_discard_increments_session_generation(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        old_gen = cb._get_session_generation()
        cb.discard_pending_outputs()
        assert cb._get_session_generation() == old_gen + 1

    @patch("recognition_handler.config")
    def test_discard_clears_subtitles(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        cb.discard_pending_outputs()
        cb.state.update_subtitles.assert_called_once_with("", "", False, "")

    @patch("recognition_handler.config")
    def test_discard_resets_typing_ongoing(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        cb._last_osc_typing_ongoing = True
        cb.discard_pending_outputs()
        assert cb._last_osc_typing_ongoing is False

    @patch("recognition_handler.config")
    def test_discard_resets_partial_translation_state(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        cb.last_partial_translation = "some text"
        cb.last_partial_translation_secondary = "other"
        cb._latest_partial_request_id = 42
        cb._prefer_deepl_on_next_final = True
        cb.discard_pending_outputs()
        assert cb.last_partial_translation is None
        assert cb.last_partial_translation_secondary is None
        assert cb._latest_partial_request_id == 0
        assert cb._prefer_deepl_on_next_final is False

    @patch("recognition_handler.config")
    def test_discard_does_not_raise_when_update_subtitles_fails(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        cb.state.update_subtitles.side_effect = RuntimeError("boom")
        # 不应抛出
        cb.discard_pending_outputs()
        assert cb._discard_results is True

    def test_resume_clears_flag(self, cb):
        cb._discard_results = True
        cb._discard_deadline = time.monotonic() + 999
        cb.resume_outputs()
        assert cb._discard_results is False


class TestOnResultDiscardWindow:
    @pytest.fixture
    def cb(self):
        return VRChatRecognitionCallback(MagicMock())

    @patch("recognition_handler.config")
    def test_on_result_dropped_within_discard_window(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        cb.discard_pending_outputs()
        cb.state.update_subtitles.reset_mock()
        cb.on_result(RecognitionEvent(text="hello", is_final=True, raw={}))
        cb.state.update_subtitles.assert_not_called()
        assert cb._discard_results is True

    @patch("recognition_handler.config")
    def test_on_result_partial_dropped_within_discard_window(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        cb.discard_pending_outputs()
        cb.state.update_subtitles.reset_mock()
        cb.on_result(RecognitionEvent(text="hello", is_final=False, raw={}))
        cb.state.update_subtitles.assert_not_called()
        assert cb._discard_results is True

    @patch("recognition_handler.config")
    def test_on_result_auto_clears_after_window_expires(self, mock_config, cb):
        mock_config.DOUBLE_MUTE_CLEAR_DISCARD_WINDOW_SECONDS = 2.0
        cb.discard_pending_outputs()
        # 把 deadline 设为过去，模拟窗口过期；用空 text 让 on_result 解除后早退
        cb._discard_deadline = time.monotonic() - 1.0
        cb.on_result(RecognitionEvent(text="", is_final=True, raw={}))
        assert cb._discard_results is False

    def test_on_session_started_clears_discard(self, cb):
        cb._discard_results = True
        cb.on_session_started()
        assert cb._discard_results is False

    def test_on_session_started_increments_generation_after_discard(self, cb):
        gen_before = cb._get_session_generation()
        cb._discard_results = True
        cb.on_session_started()
        # on_session_started 自身也会递增 generation，加上 discard 已递增过一次
        assert cb._get_session_generation() > gen_before
        assert cb._discard_results is False


# ═══════════════════════════════════════════════════════════════════════
# main.handle_mute_change: double-mute detection
# ═══════════════════════════════════════════════════════════════════════


# 延迟导入 main，避免在模块加载阶段触发其重型顶层副作用。
def _import_main():
    import main as main_module
    return main_module


@pytest.fixture
def main_module():
    return _import_main()


@pytest.fixture
def fake_state():
    state = MagicMock()
    state.last_mute_engaged_time = None
    state.recognition_callback = MagicMock()
    state.current_asr_backend = "qwen"
    state.recognition_instance = None
    return state


class TestHandleMuteChangeDoubleMute:
    @patch("main.is_effective_mic_control_enabled", return_value=False)
    @patch("main.osc_manager")
    @patch("main.config")
    def test_first_mute_records_time_no_clear(
        self, mock_config, mock_osc, mock_mic, main_module, fake_state,
    ):
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = True
        mock_config.DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
        mock_osc.clear_chatbox = AsyncMock()

        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))

        mock_osc.clear_chatbox.assert_not_called()
        fake_state.recognition_callback.discard_pending_outputs.assert_not_called()
        assert fake_state.last_mute_engaged_time is not None

    @patch("main.is_effective_mic_control_enabled", return_value=False)
    @patch("main.osc_manager")
    @patch("main.config")
    def test_second_mute_within_window_triggers_clear(
        self, mock_config, mock_osc, mock_mic, main_module, fake_state,
    ):
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = True
        mock_config.DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
        mock_osc.clear_chatbox = AsyncMock()

        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))
        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))

        mock_osc.clear_chatbox.assert_awaited_once()
        fake_state.recognition_callback.discard_pending_outputs.assert_called_once()
        # 触发后重置，避免连续误触发
        assert fake_state.last_mute_engaged_time is None

    @patch("main.is_effective_mic_control_enabled", return_value=False)
    @patch("main.osc_manager")
    @patch("main.config")
    def test_second_mute_outside_window_no_clear(
        self, mock_config, mock_osc, mock_mic, main_module, fake_state,
    ):
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = True
        mock_config.DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
        mock_osc.clear_chatbox = AsyncMock()

        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))
        # 手动把上次静音时间设到很久以前，模拟窗口已过期
        fake_state.last_mute_engaged_time = time.monotonic() - 100.0
        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))

        mock_osc.clear_chatbox.assert_not_called()
        fake_state.recognition_callback.discard_pending_outputs.assert_not_called()
        # 第二次更新了记录时间（不再是过去那个值）
        assert fake_state.last_mute_engaged_time != time.monotonic() - 100.0

    @patch("main.is_effective_mic_control_enabled", return_value=False)
    @patch("main.osc_manager")
    @patch("main.config")
    def test_disabled_flag_never_triggers_clear(
        self, mock_config, mock_osc, mock_mic, main_module, fake_state,
    ):
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = False
        mock_config.DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
        mock_osc.clear_chatbox = AsyncMock()

        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))
        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))

        mock_osc.clear_chatbox.assert_not_called()
        fake_state.recognition_callback.discard_pending_outputs.assert_not_called()
        # 开关关闭时从未记录时间
        assert fake_state.last_mute_engaged_time is None

    @patch("main.is_effective_mic_control_enabled", return_value=False)
    @patch("main.osc_manager")
    @patch("main.config")
    def test_unmute_does_not_trigger_clear(
        self, mock_config, mock_osc, mock_mic, main_module, fake_state,
    ):
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = True
        mock_config.DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
        mock_osc.clear_chatbox = AsyncMock()

        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=False))
        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=False))

        mock_osc.clear_chatbox.assert_not_called()
        assert fake_state.last_mute_engaged_time is None

    @patch("main.is_effective_mic_control_enabled", return_value=False)
    @patch("main.osc_manager")
    @patch("main.config")
    def test_third_mute_after_clear_records_again(
        self, mock_config, mock_osc, mock_mic, main_module, fake_state,
    ):
        """连续三次静音：第二次触发清空并重置，第三次应重新记录而非再次触发。"""
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = True
        mock_config.DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
        mock_osc.clear_chatbox = AsyncMock()

        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))
        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))
        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))

        # 只触发一次清空
        mock_osc.clear_chatbox.assert_awaited_once()
        fake_state.recognition_callback.discard_pending_outputs.assert_called_once()
        # 第三次重新记录了时间
        assert fake_state.last_mute_engaged_time is not None

    @patch("main.is_effective_mic_control_enabled", return_value=False)
    @patch("main.osc_manager")
    @patch("main.config")
    def test_clear_works_without_recognition_callback(
        self, mock_config, mock_osc, mock_mic, main_module, fake_state,
    ):
        """recognition_callback 为 None 时仍能清空聊天框，不抛异常。"""
        mock_config.ENABLE_DOUBLE_MUTE_CLEAR = True
        mock_config.DOUBLE_MUTE_CLEAR_WINDOW_SECONDS = 0.8
        mock_osc.clear_chatbox = AsyncMock()
        fake_state.recognition_callback = None

        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))
        asyncio.run(main_module.handle_mute_change(fake_state, is_muted=True))

        mock_osc.clear_chatbox.assert_awaited_once()
        assert fake_state.last_mute_engaged_time is None
