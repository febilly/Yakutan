"""Tests for pipeline orchestration functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from streaming_translation import (
    DEFAULT_API_TYPE,
    TRANSLATION_API_CLASS_REGISTRY,
    TranslationConfig,
    ensure_secondary_translator,
    is_streaming_deepl_hybrid_mode,
    is_streaming_translation_mode,
    reinitialize_translator,
    reverse_translation,
    translate_with_backend,
    update_secondary_translator,
)
from streaming_translation.pipeline import (
    _get_api_class,
    _normalize_optional_language_code,
)


# ── Helpers ───────────────────────────────────────────────────────────

class MockState:
    def __init__(self):
        self.translation_api = None
        self.translator = None
        self.secondary_translation_api = None
        self.secondary_translator = None
        self.backwards_translation_api = None
        self.backwards_translator = None
        self.deepl_fallback_translation_api = None
        self.deepl_fallback_translator = None
        self.secondary_deepl_fallback_translation_api = None
        self.secondary_deepl_fallback_translator = None
        self.translation_api_type = None
        self.target_language = None
        self.secondary_target_language = None


# ── Normalize helpers ─────────────────────────────────────────────────

class TestNormalizeOptionalLanguageCode:
    def test_none(self):
        assert _normalize_optional_language_code(None) is None

    def test_empty_string(self):
        assert _normalize_optional_language_code("") is None

    def test_whitespace(self):
        assert _normalize_optional_language_code("  ") is None

    def test_valid_code(self):
        assert _normalize_optional_language_code("ja") == "ja"
        assert _normalize_optional_language_code(" zh-CN ") == "zh-CN"
        assert _normalize_optional_language_code("en") == "en"


# ── API Registry ──────────────────────────────────────────────────────

class TestAPIRegistry:
    def test_has_all_backends(self):
        expected = {
            "google_web", "google_dictionary", "openrouter",
            "openrouter_streaming", "openrouter_streaming_deepl_hybrid",
            "deepl", "qwen_mt",
        }
        assert set(TRANSLATION_API_CLASS_REGISTRY.keys()) == expected

    def test_get_api_class_exists(self):
        from streaming_translation.api.deepl import DeepLAPI
        assert _get_api_class("deepl") is DeepLAPI

    def test_get_api_class_fallback_to_default(self):
        from streaming_translation.api.qwen_mt import QwenMTAPI
        assert _get_api_class("nonexistent") is QwenMTAPI

    def test_default_api_type(self):
        assert DEFAULT_API_TYPE == "qwen_mt"


# ── Streaming mode helpers ────────────────────────────────────────────

class TestStreamingModeHelpers:
    def test_is_streaming_true(self):
        assert is_streaming_translation_mode("openrouter_streaming") is True
        assert is_streaming_translation_mode("openrouter_streaming_deepl_hybrid") is True

    def test_is_streaming_false(self):
        assert is_streaming_translation_mode("deepl") is False
        assert is_streaming_translation_mode("qwen_mt") is False
        assert is_streaming_translation_mode("google_web") is False

    def test_is_hybrid_true(self):
        assert is_streaming_deepl_hybrid_mode("openrouter_streaming_deepl_hybrid") is True

    def test_is_hybrid_false(self):
        assert is_streaming_deepl_hybrid_mode("deepl") is False


# ── reinitialize_translator ───────────────────────────────────────────

class TestReinitializeTranslator:
    def test_sets_primary_translator(self):
        state = MockState()
        cfg = TranslationConfig(
            target_language="de",
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
        )
        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            reinitialize_translator(state, cfg)

        assert state.translator is not None
        assert state.translation_api_type == "qwen_mt"
        assert state.target_language == "de"

    def test_secondary_translator_when_configured(self):
        state = MockState()
        cfg = TranslationConfig(
            target_language="en",
            secondary_target_language="ja",
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
        )
        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            reinitialize_translator(state, cfg)

        assert state.secondary_translator is not None
        assert state.secondary_target_language == "ja"

    def test_no_secondary_when_none(self):
        state = MockState()
        cfg = TranslationConfig(
            target_language="en",
            secondary_target_language=None,
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
        )
        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            reinitialize_translator(state, cfg)

        assert state.secondary_translator is None

    def test_backwards_translator_always_set(self):
        state = MockState()
        cfg = TranslationConfig(
            target_language="en",
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
            proxy_url=None,
        )
        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai, \
             patch("streaming_translation.api.google_dictionary.aiohttp.ClientSession") as mock_session:
            mock_openai.return_value = MagicMock()
            mock_session.return_value = MagicMock()
            reinitialize_translator(state, cfg)
        assert state.backwards_translator is not None
        assert state.backwards_translation_api is not None

    def test_streaming_mode_sets_partial(self):
        state = MockState()
        cfg = TranslationConfig(
            target_language="en",
            translation_api_type="openrouter_streaming",
            llm_base_url="https://test.ai/v1",
            llm_model="test",
            llm_api_key="test-key",
            proxy_url=None,
        )
        with patch("streaming_translation.api.openrouter.OpenAI"):
            reinitialize_translator(state, cfg)
        assert cfg.translate_partial_results is True


# ── update_secondary_translator ───────────────────────────────────────

class TestUpdateSecondaryTranslator:
    def test_adds_secondary(self):
        state = MockState()
        state.secondary_target_language = None
        state.secondary_translator = None
        cfg = TranslationConfig(
            secondary_target_language="fr",
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
        )
        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            update_secondary_translator(state, cfg)
        assert state.secondary_target_language == "fr"
        assert state.secondary_translator is not None

    def test_removes_secondary(self):
        state = MockState()
        state.secondary_target_language = "fr"
        state.secondary_translator = MagicMock()
        cfg = TranslationConfig(secondary_target_language=None)
        update_secondary_translator(state, cfg)
        assert state.secondary_target_language is None
        assert state.secondary_translator is None

    def test_no_change_skips(self):
        state = MockState()
        state.secondary_target_language = "fr"
        state.secondary_translator = MagicMock()
        cfg = TranslationConfig(secondary_target_language="fr")
        update_secondary_translator(state, cfg)
        # Should not recreate
        assert state.secondary_translator is not None


# ── ensure_secondary_translator ───────────────────────────────────────

class TestEnsureSecondaryTranslator:
    def test_returns_false_when_no_target(self):
        state = MockState()
        cfg = TranslationConfig()
        assert ensure_secondary_translator(state, None, cfg) is False

    def test_returns_true_when_already_set(self):
        state = MockState()
        state.secondary_target_language = "ja"
        state.secondary_translator = MagicMock()
        cfg = TranslationConfig()
        assert ensure_secondary_translator(state, "ja", cfg) is True

    def test_creates_and_returns_true(self):
        state = MockState()
        cfg = TranslationConfig(
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
        )
        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            result = ensure_secondary_translator(state, "de", cfg)
        assert result is True
        assert state.secondary_translator is not None
        assert state.secondary_target_language == "de"


# ── translate_with_backend ────────────────────────────────────────────

class TestTranslateWithBackend:
    def test_translate_called(self):
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "Hallo"
        result = translate_with_backend(
            mock_translator, None,
            "Hello", "de",
            source_language="en", context_prefix="Chat:",
        )
        assert result == "Hallo"
        mock_translator.translate.assert_called_once()

    def test_prefer_deepl_uses_deepl_when_available(self):
        mock_translator = MagicMock()
        mock_deepl = MagicMock()
        mock_deepl.translate.return_value = "Hallo"
        result = translate_with_backend(
            mock_translator, mock_deepl,
            "Hello", "de",
            prefer_deepl=True,
            source_language="en",
        )
        assert result == "Hallo"
        mock_deepl.translate.assert_called_once()
        mock_translator.translate.assert_not_called()

    def test_prefer_deepl_fallback_when_deepl_fails(self):
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "Hallo"
        mock_deepl = MagicMock()
        mock_deepl.translate.return_value = "[ERROR] DeepL failed"
        result = translate_with_backend(
            mock_translator, mock_deepl,
            "Hello", "de",
            prefer_deepl=True,
            source_language="en",
        )
        assert result == "Hallo"
        mock_translator.translate.assert_called_once()

    def test_passes_previous_translation(self):
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "Hello World"
        translate_with_backend(
            mock_translator, None,
            "Hola Mundo", "en",
            previous_translation="Hola",
            source_language="es",
        )
        _, kwargs = mock_translator.translate.call_args
        assert kwargs.get("previous_translation") == "Hola"

    def test_record_history_default_true(self):
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "test"
        translate_with_backend(mock_translator, None, "test", "en", source_language="en")
        _, kwargs = mock_translator.translate.call_args
        assert kwargs.get("record_history") is True

    def test_record_history_false(self):
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "test"
        translate_with_backend(mock_translator, None, "test", "en",
                              source_language="en", record_history=False)
        _, kwargs = mock_translator.translate.call_args
        assert kwargs.get("record_history") is False

    def test_detected_source_language_passed(self):
        mock_translator = MagicMock()
        mock_translator.translate.return_value = "test"
        translate_with_backend(mock_translator, None, "test", "en",
                              source_language="auto",
                              detected_source_language="ja")
        _, kwargs = mock_translator.translate.call_args
        assert kwargs.get("detected_source_language") == "ja"


# ── reverse_translation ───────────────────────────────────────────────

class TestReverseTranslation:
    def test_success(self):
        mock_bt = MagicMock()
        mock_bt.translate.return_value = "Hello"
        result = reverse_translation(mock_bt, "Hola", "es", "en")
        assert result == "Hello"

    def test_exception_returns_none(self):
        mock_bt = MagicMock()
        mock_bt.translate.side_effect = RuntimeError("fail")
        result = reverse_translation(mock_bt, "Hola", "es", "en")
        assert result is None
