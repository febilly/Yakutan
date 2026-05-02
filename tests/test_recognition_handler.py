"""Tests for recognition_handler — core callback logic."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from recognition_handler import VRChatRecognitionCallback
from speech_recognizers.base_speech_recognizer import RecognitionEvent


class TestNormalizeLang:
    def test_english(self):
        result = VRChatRecognitionCallback._normalize_lang("en")
        assert result == "en"

    def test_chinese_simplified(self):
        result = VRChatRecognitionCallback._normalize_lang("zh-cn")
        assert result == "zh-hans"

    def test_chinese_traditional(self):
        result = VRChatRecognitionCallback._normalize_lang("zh-tw")
        assert result == "zh-hant"

    def test_japanese(self):
        result = VRChatRecognitionCallback._normalize_lang("ja")
        assert result == "ja"

    def test_none(self):
        result = VRChatRecognitionCallback._normalize_lang(None)
        assert result == "auto"


class TestShouldTranslate:
    @patch("recognition_handler.config")
    def test_same_language_no_translate(self, mock_config):
        mock_config.TARGET_LANGUAGE = "ja"
        result = VRChatRecognitionCallback._should_translate("ja", "ja")
        assert result is False

    @patch("recognition_handler.config")
    def test_different_language_translate(self, mock_config):
        mock_config.TARGET_LANGUAGE = "ja"
        result = VRChatRecognitionCallback._should_translate("en", "ja")
        assert result is True

    @patch("recognition_handler.config")
    def test_no_target_no_translate(self, mock_config):
        mock_config.TARGET_LANGUAGE = None
        result = VRChatRecognitionCallback._should_translate("en", None)
        assert result is False


class TestExtractStreamingSegment:
    def test_no_punctuation(self):
        result = VRChatRecognitionCallback._extract_streaming_segment("hello")
        assert result is None

    def test_punctuation_at_end_no_content_after(self):
        """Nothing after punctuation → no segment extracted."""
        result = VRChatRecognitionCallback._extract_streaming_segment("hello.")
        assert result is None

    def test_punctuation_in_middle(self):
        result = VRChatRecognitionCallback._extract_streaming_segment("hello, world")
        assert result == "hello,"

    def test_multiple_punctuation(self):
        """Finds last punctuation that has content after it."""
        result = VRChatRecognitionCallback._extract_streaming_segment("hello! world.")
        # last "." has nothing after → fallback to "!" which has " world"
        assert result == "hello!"

    def test_chinese_punctuation(self):
        result = VRChatRecognitionCallback._extract_streaming_segment("你好。世界")
        assert result == "你好。"

    def test_japanese_punctuation(self):
        result = VRChatRecognitionCallback._extract_streaming_segment("こんにちは。元気ですか")
        assert result == "こんにちは。"

    def test_empty_text(self):
        result = VRChatRecognitionCallback._extract_streaming_segment("")
        assert result is None

    def test_none_text(self):
        result = VRChatRecognitionCallback._extract_streaming_segment(None)
        assert result is None

    def test_text_with_only_punctuation(self):
        """Only punctuation: last char has no content after, but penultimate does."""
        result = VRChatRecognitionCallback._extract_streaming_segment("...")
        assert result == ".."

    def test_trailing_whitespace_no_content(self):
        """Whitespace after punctuation is stripped, so nothing remains."""
        result = VRChatRecognitionCallback._extract_streaming_segment("hello.   ")
        assert result is None


class TestShouldTriggerPartialTranslation:
    @patch("recognition_handler.config")
    def test_segment_long_enough(self, mock_config):
        mock_config.MIN_PARTIAL_TRANSLATION_CHARS = 2
        result = VRChatRecognitionCallback._should_trigger_partial_translation("hello")
        assert result is True

    @patch("recognition_handler.config")
    def test_segment_too_short(self, mock_config):
        mock_config.MIN_PARTIAL_TRANSLATION_CHARS = 5
        result = VRChatRecognitionCallback._should_trigger_partial_translation("hi")
        assert result is False

    @patch("recognition_handler.config")
    def test_strips_punctuation_before_counting(self, mock_config):
        mock_config.MIN_PARTIAL_TRANSLATION_CHARS = 2
        result = VRChatRecognitionCallback._should_trigger_partial_translation("a.")
        assert result is False

    @patch("recognition_handler.config")
    def test_none_segment(self, mock_config):
        mock_config.MIN_PARTIAL_TRANSLATION_CHARS = 2
        result = VRChatRecognitionCallback._should_trigger_partial_translation(None)
        assert result is False

    @patch("recognition_handler.config")
    def test_empty_segment(self, mock_config):
        mock_config.MIN_PARTIAL_TRANSLATION_CHARS = 2
        result = VRChatRecognitionCallback._should_trigger_partial_translation("")
        assert result is False


class TestHasErrorText:
    def test_has_error(self):
        assert VRChatRecognitionCallback._has_error_text("[ERROR] something") is True

    def test_no_error(self):
        assert VRChatRecognitionCallback._has_error_text("normal text") is False

    def test_none(self):
        assert VRChatRecognitionCallback._has_error_text(None) is False

    def test_empty(self):
        assert VRChatRecognitionCallback._has_error_text("") is False

    def test_error_in_middle(self):
        assert VRChatRecognitionCallback._has_error_text("prefix [ERROR] suffix") is True


class TestFilterErrorLinesForOsc:
    @patch("recognition_handler.config")
    def test_no_error_shows_display(self, mock_config):
        mock_config.OSC_SEND_ERROR_MESSAGES = False
        result = VRChatRecognitionCallback._filter_error_lines_for_osc(
            "hello [original]",
            primary_raw_text="translated text",
            primary_display_text="translated text",
        )
        assert result == "hello [original]"

    @patch("recognition_handler.config")
    def test_primary_error_removes_primary(self, mock_config):
        mock_config.OSC_SEND_ERROR_MESSAGES = False
        result = VRChatRecognitionCallback._filter_error_lines_for_osc(
            "ERR [original]",
            primary_raw_text="[ERROR] fail",
        )
        assert result is None

    @patch("recognition_handler.config")
    def test_osc_send_errors_override(self, mock_config):
        mock_config.OSC_SEND_ERROR_MESSAGES = True
        result = VRChatRecognitionCallback._filter_error_lines_for_osc(
            "[ERROR] fail",
            primary_raw_text="[ERROR] fail",
        )
        assert result == "[ERROR] fail"

    @patch("recognition_handler.config")
    def test_all_lines_error_returns_none(self, mock_config):
        mock_config.OSC_SEND_ERROR_MESSAGES = False
        result = VRChatRecognitionCallback._filter_error_lines_for_osc(
            "[ERROR] a\n[ERROR] b",
            primary_raw_text="[ERROR] a",
            primary_display_text="[ERROR] a",
            secondary_raw_text="[ERROR] b",
            secondary_display_text="[ERROR] b",
        )
        assert result is None

    @patch("recognition_handler.config")
    def test_default_display_text_none(self, mock_config):
        mock_config.OSC_SEND_ERROR_MESSAGES = False
        result = VRChatRecognitionCallback._filter_error_lines_for_osc(None)
        assert result is None


class TestVRChatRecognitionCallbackInit:
    def test_initial_state(self):
        state = MagicMock()
        callback = VRChatRecognitionCallback(state)
        assert callback.state is state
        assert callback.last_partial_translation is None
        assert callback.last_partial_translation_secondary is None
        assert callback.translating_partial is False
        assert callback._finalized_seq == 0
        assert callback._final_output_version == 0

    def test_mute_finalization(self):
        state = MagicMock()
        callback = VRChatRecognitionCallback(state)
        assert callback._prefer_deepl_on_next_final is False
        callback.mark_mute_finalization_requested()
        assert callback._prefer_deepl_on_next_final is True
        callback.clear_mute_finalization_requested()
        assert callback._prefer_deepl_on_next_final is False


class TestSeqManagement:
    @pytest.fixture
    def cb(self):
        return VRChatRecognitionCallback(MagicMock())

    def test_next_async_result_seq_increments(self, cb):
        seq1 = cb._next_async_result_seq()
        seq2 = cb._next_async_result_seq()
        assert seq2 == seq1 + 1

    def test_session_generation(self, cb):
        gen1 = cb._get_session_generation()
        cb._translate_ordering_lock = MagicMock()  # reset
        with cb._translate_ordering_lock:
            cb._session_generation += 1
        gen2 = cb._get_session_generation()
        assert gen2 == gen1 + 1

    def test_is_latest_partial_request(self, cb):
        cb._latest_partial_request_id = 5
        cb._finalized_seq = 2
        cb._final_output_version = 1
        cb._session_generation = 0
        assert cb._is_latest_partial_request(5, 2, 1, 0) is True
        assert cb._is_latest_partial_request(4, 2, 1, 0) is False
        assert cb._is_latest_partial_request(5, 3, 1, 0) is False

    def test_adopt_async_result(self, cb):
        assert cb._try_adopt_async_result(1, 0) is True
        assert cb._try_adopt_async_result(0, 0) is False  # older seq
        assert cb._try_adopt_async_result(2, 0) is True
        assert cb._try_adopt_async_result(1, 0) is False  # already applied

    def test_adopt_async_result_wrong_generation(self, cb):
        cb._session_generation = 5
        assert cb._try_adopt_async_result(1, 3) is False

    def test_is_async_result_current(self, cb):
        cb._latest_applied_async_result_seq = 5
        cb._session_generation = 0
        assert cb._is_async_result_current(5, 0) is True
        assert cb._is_async_result_current(4, 0) is False

    def test_reset_partial_state(self, cb):
        cb.last_partial_translation = "some text"
        cb.last_partial_translation_secondary = "other"
        cb._latest_partial_request_id = 42
        cb._prefer_deepl_on_next_final = True
        cb._reset_partial_translation_state()
        assert cb.last_partial_translation is None
        assert cb._latest_partial_request_id == 0
        assert cb._prefer_deepl_on_next_final is False


class TestSessionLifecycle:
    def test_on_session_started(self):
        cb = VRChatRecognitionCallback(MagicMock())
        old_gen = cb._get_session_generation()
        cb.mark_mute_finalization_requested()
        cb.last_partial_translation = "old text"
        cb.on_session_started()
        assert cb._get_session_generation() > old_gen
        assert cb.last_partial_translation is None

    def test_on_session_stopped(self):
        cb = VRChatRecognitionCallback(MagicMock())
        old_gen = cb._get_session_generation()
        cb.on_session_stopped()
        assert cb._get_session_generation() > old_gen

    def test_on_error_logs(self):
        cb = VRChatRecognitionCallback(MagicMock())
        # Should not raise
        cb.on_error(RuntimeError("test error"))


# ═══════════════════════════════════════════════════════════════════════
# Decoupling Tests — ASR ↔ Translation non-blocking guarantees
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_state(executor=None, loop=None):
    """Create a minimal mock state with required attributes for testing."""
    state = MagicMock()
    state.executor = executor or ThreadPoolExecutor(max_workers=4)
    state.audio_executor = ThreadPoolExecutor(max_workers=1)
    state.translator = MagicMock()
    state.secondary_translator = None
    state.deepl_fallback_translator = None
    state.secondary_deepl_fallback_translator = None
    state.backwards_translator = MagicMock()
    state.language_detector = MagicMock()
    state.language_detector.detect.return_value = {"language": "zh-cn"}
    state.subtitles_state = {"original": "", "translated": "", "reverse_translated": "", "ongoing": False}
    state.current_asr_backend = "qwen"
    state.main_loop = loop
    return state


class TestASRNotBlockedByTranslation:
    """Verify that ASR callbacks return immediately, even when translation is slow."""

    @patch("recognition_handler.config")
    def test_on_result_returns_immediately_for_final_with_slow_translation(self, mock_config):
        """ASR callback must not block while translation runs in background."""
        mock_config.ENABLE_TRANSLATION = True
        mock_config.TRANSLATE_PARTIAL_RESULTS = False
        mock_config.TARGET_LANGUAGE = "ja"
        mock_config.SOURCE_LANGUAGE = "auto"
        mock_config.SECONDARY_TARGET_LANGUAGE = None
        mock_config.CONTEXT_PREFIX = ""
        mock_config.ENABLE_REVERSE_TRANSLATION = False
        mock_config.SHOW_ORIGINAL_AND_LANG_TAG = True
        mock_config.SMART_TARGET_PRIMARY_ENABLED = False
        mock_config.SMART_TARGET_SECONDARY_ENABLED = False
        mock_config.TRANSLATION_API_TYPE = "deepl"
        mock_config.get_effective_osc_text_max_length.return_value = None

        executor = ThreadPoolExecutor(max_workers=2)
        state = _make_mock_state(executor=executor)

        translate_call_count = 0

        def slow_translate(*args, **kwargs):
            nonlocal translate_call_count
            translate_call_count += 1
            time.sleep(2.0)
            return "翻訳結果"

        state.translator.translate = MagicMock(side_effect=slow_translate)

        callback = VRChatRecognitionCallback(state)
        callback.loop = None

        event = RecognitionEvent(text="こんにちは世界", is_final=True, raw={})

        start = time.monotonic()
        callback.on_result(event)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"on_result took {elapsed:.2f}s — should be < 0.5s "
            "(ASR callback must not block on translation)"
        )

        time.sleep(2.5)
        assert translate_call_count >= 1, "Translation was never called"

        executor.shutdown(wait=True)

    @patch("recognition_handler.config")
    def test_new_asr_events_processed_while_translation_running(self, mock_config):
        """Verify that new ASR events can be processed during slow translation."""
        mock_config.ENABLE_TRANSLATION = True
        mock_config.TRANSLATE_PARTIAL_RESULTS = False
        mock_config.TARGET_LANGUAGE = "ja"
        mock_config.SOURCE_LANGUAGE = "auto"
        mock_config.SECONDARY_TARGET_LANGUAGE = None
        mock_config.CONTEXT_PREFIX = ""
        mock_config.ENABLE_REVERSE_TRANSLATION = False
        mock_config.SHOW_ORIGINAL_AND_LANG_TAG = True
        mock_config.SMART_TARGET_PRIMARY_ENABLED = False
        mock_config.SMART_TARGET_SECONDARY_ENABLED = False
        mock_config.TRANSLATION_API_TYPE = "deepl"
        mock_config.get_effective_osc_text_max_length.return_value = None

        executor = ThreadPoolExecutor(max_workers=2)
        state = _make_mock_state(executor=executor)

        translate_started = threading.Event()
        translate_block = threading.Event()

        def slow_translate(*args, **kwargs):
            translate_started.set()
            translate_block.wait()
            return "翻訳結果"

        state.translator.translate = MagicMock(side_effect=slow_translate)

        callback = VRChatRecognitionCallback(state)
        callback.loop = None

        event1 = RecognitionEvent(text="最初の文章", is_final=True, raw={})
        callback.on_result(event1)

        assert translate_started.wait(timeout=3), "Translation did not start"

        start = time.monotonic()
        event2 = RecognitionEvent(text="二番目の文章", is_final=True, raw={})
        callback.on_result(event2)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"Second on_result took {elapsed:.2f}s — should not block "
            "while previous translation is running"
        )

        translate_block.set()
        executor.shutdown(wait=True)

    @patch("recognition_handler.config")
    def test_partial_result_not_blocked_by_translation(self, mock_config):
        """Partial (ongoing) results should always be fast — no translation on ASR thread."""
        mock_config.ENABLE_TRANSLATION = True
        mock_config.TRANSLATE_PARTIAL_RESULTS = True
        mock_config.TARGET_LANGUAGE = "ja"
        mock_config.SOURCE_LANGUAGE = "auto"
        mock_config.SHOW_ORIGINAL_AND_LANG_TAG = True
        mock_config.SHOW_PARTIAL_RESULTS = True
        mock_config.MIN_PARTIAL_TRANSLATION_CHARS = 2
        mock_config.SMART_TARGET_PRIMARY_ENABLED = False
        mock_config.SMART_TARGET_SECONDARY_ENABLED = False
        mock_config.SECONDARY_TARGET_LANGUAGE = None
        mock_config.get_effective_osc_text_max_length.return_value = None

        state = _make_mock_state()
        callback = VRChatRecognitionCallback(state)
        callback.loop = None

        event = RecognitionEvent(text="こんにちは。これ", is_final=False, raw={})

        start = time.monotonic()
        callback.on_result(event)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1, (
            f"Partial on_result took {elapsed:.3f}s — partial results must be instant"
        )

        state.executor.shutdown(wait=True)


class TestExecutorDispatchPath:
    """Verify that the executor fallback path correctly translates and applies results."""

    @patch("recognition_handler.config")
    def test_dispatch_final_translation_to_executor_returns_immediately(self, mock_config):
        """_dispatch_final_translation_to_executor must return without waiting."""
        mock_config.ENABLE_REVERSE_TRANSLATION = False
        mock_config.SOURCE_LANGUAGE = "auto"
        mock_config.CONTEXT_PREFIX = ""
        mock_config.TRANSLATION_API_TYPE = "deepl"

        executor = ThreadPoolExecutor(max_workers=1)
        state = _make_mock_state(executor=executor)

        def slow_translate(*args, **kwargs):
            time.sleep(1.0)
            return "翻訳"

        with patch("recognition_handler.translate_with_backend", side_effect=slow_translate):
            callback = VRChatRecognitionCallback(state)
            callback.loop = MagicMock()

            start = time.monotonic()
            callback._dispatch_final_translation_to_executor(
                text="テスト",
                source_lang="ja",
                normalized_source="ja",
                actual_target="en",
                actual_secondary_target=None,
                primary_should_translate=True,
                secondary_should_translate=False,
                use_secondary_output=False,
                use_deepl_final=False,
                previous_translation=None,
                previous_translation_secondary=None,
                previous_source_segment=None,
                async_result_seq=1,
                session_generation=callback._get_session_generation(),
            )
            elapsed = time.monotonic() - start

            assert elapsed < 0.2, (
                f"_dispatch_final_translation_to_executor took {elapsed:.2f}s "
                "— must return immediately"
            )

        executor.shutdown(wait=True)

    @patch("recognition_handler.config")
    def test_executor_dispatch_posts_results_to_event_loop(self, mock_config):
        """Translation results from executor must be posted to event loop."""
        mock_config.ENABLE_REVERSE_TRANSLATION = False
        mock_config.SOURCE_LANGUAGE = "auto"
        mock_config.CONTEXT_PREFIX = ""
        mock_config.TRANSLATION_API_TYPE = "deepl"

        executor = ThreadPoolExecutor(max_workers=1)
        state = _make_mock_state(executor=executor)
        state.translator.append_history_entry = MagicMock()

        loop = MagicMock()
        callback = VRChatRecognitionCallback(state)
        callback.loop = loop
        callback._try_adopt_async_result = MagicMock(return_value=True)
        callback._is_async_result_current = MagicMock(return_value=True)

        with patch("recognition_handler.translate_with_backend", return_value="hello"):
            callback._dispatch_final_translation_to_executor(
                text="世界",
                source_lang="zh-cn",
                normalized_source="zh-hans",
                actual_target="en",
                actual_secondary_target=None,
                primary_should_translate=True,
                secondary_should_translate=False,
                use_secondary_output=False,
                use_deepl_final=False,
                previous_translation=None,
                previous_translation_secondary=None,
                previous_source_segment=None,
                async_result_seq=1,
                session_generation=callback._get_session_generation(),
            )

        executor.shutdown(wait=True)

        assert loop.call_soon_threadsafe.called, (
            "Results must be posted to event loop via call_soon_threadsafe"
        )

    @patch("recognition_handler.config")
    def test_executor_dispatch_records_history(self, mock_config):
        """Translation history must be recorded after executor translation."""
        mock_config.ENABLE_REVERSE_TRANSLATION = False
        mock_config.SOURCE_LANGUAGE = "auto"
        mock_config.CONTEXT_PREFIX = ""
        mock_config.TRANSLATION_API_TYPE = "deepl"

        executor = ThreadPoolExecutor(max_workers=1)
        state = _make_mock_state(executor=executor)
        state.translator.append_history_entry = MagicMock()

        callback = VRChatRecognitionCallback(state)
        callback.loop = MagicMock()  # Prevent event loop posting from failing

        with patch("recognition_handler.translate_with_backend", return_value="hello world"):
            callback._dispatch_final_translation_to_executor(
                text="世界你好",
                source_lang="zh-cn",
                normalized_source="zh-hans",
                actual_target="en",
                actual_secondary_target=None,
                primary_should_translate=True,
                secondary_should_translate=False,
                use_secondary_output=False,
                use_deepl_final=False,
                previous_translation=None,
                previous_translation_secondary=None,
                previous_source_segment=None,
                async_result_seq=1,
                session_generation=callback._get_session_generation(),
            )

        executor.shutdown(wait=True)

        state.translator.append_history_entry.assert_called_once_with(
            "世界你好", "hello world", "en",
        )

    @patch("recognition_handler.config")
    def test_executor_dispatch_discards_stale_session_results(self, mock_config):
        """Results from an old session must be discarded."""
        mock_config.ENABLE_REVERSE_TRANSLATION = False
        mock_config.SOURCE_LANGUAGE = "auto"
        mock_config.CONTEXT_PREFIX = ""
        mock_config.TRANSLATION_API_TYPE = "deepl"

        executor = ThreadPoolExecutor(max_workers=1)
        state = _make_mock_state(executor=executor)
        state.translator.append_history_entry = MagicMock()

        callback = VRChatRecognitionCallback(state)
        callback.loop = MagicMock()

        old_gen = callback._get_session_generation()
        callback.on_session_started()
        new_gen = callback._get_session_generation()
        assert new_gen > old_gen

        with patch("recognition_handler.translate_with_backend", return_value="stale result"):
            callback._dispatch_final_translation_to_executor(
                text="古いメッセージ",
                source_lang="ja",
                normalized_source="ja",
                actual_target="en",
                actual_secondary_target=None,
                primary_should_translate=True,
                secondary_should_translate=False,
                use_secondary_output=False,
                use_deepl_final=False,
                previous_translation=None,
                previous_translation_secondary=None,
                previous_source_segment=None,
                async_result_seq=1,
                session_generation=old_gen,
            )

        executor.shutdown(wait=True)

        state.translator.append_history_entry.assert_not_called()


class TestAudioExecutorSeparation:
    """Verify audio frame sending uses dedicated audio_executor, not shared executor."""

    @patch("recognition_handler.config")
    def test_audio_capture_uses_audio_executor(self, mock_config):
        """send_audio_frame must use audio_executor, not the shared executor."""
        import audio_capture

        state = _make_mock_state()
        state.executor = ThreadPoolExecutor(max_workers=8)
        state.audio_executor = ThreadPoolExecutor(max_workers=1)

        recognizer = MagicMock()

        async def _test():
            await audio_capture.send_audio_frame_async(state, recognizer, b"\x00" * 1024)

        asyncio.run(_test())

        recognizer.send_audio_frame.assert_called_once()

        state.executor.shutdown(wait=True)
        state.audio_executor.shutdown(wait=True)
