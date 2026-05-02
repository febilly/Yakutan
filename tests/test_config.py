"""Tests for TranslationConfig and config_from_module."""

from __future__ import annotations

from streaming_translation import TranslationConfig, config_from_module


class TestTranslationConfig:
    """Default values, field types, and edge cases."""

    def test_default_values(self):
        cfg = TranslationConfig()
        assert cfg.source_language == "auto"
        assert cfg.target_language == "ja"
        assert cfg.secondary_target_language is None
        assert cfg.fallback_language == "en"
        assert cfg.translation_api_type == "qwen_mt"
        assert cfg.translate_partial_results is False
        assert cfg.translation_context_size == 6
        assert cfg.translation_context_aware is True
        assert cfg.smart_target_primary_enabled is False
        assert cfg.smart_target_secondary_enabled is False
        assert cfg.smart_target_strategy == "most_common"
        assert cfg.proxy_url is None
        assert cfg.deepl_api_key is None
        assert cfg.dashscope_api_key is None
        assert cfg.llm_api_key is None

    def test_custom_values(self):
        cfg = TranslationConfig(
            target_language="de",
            source_language="en",
            translation_api_type="deepl",
            proxy_url="http://localhost:7890",
            deepl_api_key="test-key",
            smart_target_primary_enabled=True,
            translation_context_size=10,
        )
        assert cfg.target_language == "de"
        assert cfg.source_language == "en"
        assert cfg.translation_api_type == "deepl"
        assert cfg.proxy_url == "http://localhost:7890"
        assert cfg.deepl_api_key == "test-key"
        assert cfg.smart_target_primary_enabled is True
        assert cfg.translation_context_size == 10

    def test_secondary_target_none(self):
        cfg = TranslationConfig(secondary_target_language=None)
        assert cfg.secondary_target_language is None

    def test_secondary_target_set(self):
        cfg = TranslationConfig(secondary_target_language="en")
        assert cfg.secondary_target_language == "en"

    def test_fallback_custom(self):
        cfg = TranslationConfig(fallback_language="zh-CN")
        assert cfg.fallback_language == "zh-CN"


class TestConfigFromModule:
    """config_from_module builds TranslationConfig from module-like objects."""

    def test_empty_module(self):
        class Empty:
            pass
        cfg = config_from_module(Empty)
        assert cfg.target_language == "ja"  # default
        assert cfg.translation_api_type == "qwen_mt"
        assert cfg.source_language == "auto"

    def test_partial_module(self):
        class PartialModule:
            TARGET_LANGUAGE = "fr"
            TRANSLATION_API_TYPE = "deepl"
        cfg = config_from_module(PartialModule)
        assert cfg.target_language == "fr"
        assert cfg.translation_api_type == "deepl"
        assert cfg.source_language == "auto"  # default

    def test_full_module(self):
        class FullModule:
            SOURCE_LANGUAGE = "en"
            TARGET_LANGUAGE = "ja"
            SECONDARY_TARGET_LANGUAGE = "ko"
            FALLBACK_LANGUAGE = "zh-CN"
            TRANSLATION_API_TYPE = "openrouter"
            TRANSLATE_PARTIAL_RESULTS = True
            CONTEXT_PREFIX = "VRChat conversation:"
            TRANSLATION_CONTEXT_SIZE = 10
            TRANSLATION_CONTEXT_AWARE = True
            SMART_TARGET_PRIMARY_ENABLED = True
            SMART_TARGET_SECONDARY_ENABLED = False
            SMART_TARGET_LANGUAGE_STRATEGY = "latest"
            SMART_TARGET_LANGUAGE_WINDOW_SIZE = 20
            SMART_TARGET_LANGUAGE_EXCLUDE_SELF_LANGUAGE = False
            SMART_TARGET_LANGUAGE_FALLBACK = "en"
            SMART_TARGET_LANGUAGE_MIN_SAMPLES = 5
            LLM_BASE_URL = "https://api.example.com/v1"
            LLM_MODEL = "test-model"
            LLM_TRANSLATION_TEMPERATURE = 0.5
            LLM_TRANSLATION_TIMEOUT = 60
            LLM_TRANSLATION_FORMALITY = "high"
            LLM_TRANSLATION_STYLE = "standard"
            OPENAI_COMPAT_EXTRA_BODY_JSON = '{"custom": true}'
            LLM_PARALLEL_FASTEST_MODE = "final_only"
            USE_INTERNATIONAL_ENDPOINT = True

        cfg = config_from_module(FullModule)
        assert cfg.source_language == "en"
        assert cfg.target_language == "ja"
        assert cfg.secondary_target_language == "ko"
        assert cfg.fallback_language == "zh-CN"
        assert cfg.translation_api_type == "openrouter"
        assert cfg.translate_partial_results is True
        assert cfg.context_prefix == "VRChat conversation:"
        assert cfg.translation_context_size == 10
        assert cfg.translation_context_aware is True
        assert cfg.smart_target_primary_enabled is True
        assert cfg.smart_target_strategy == "latest"
        assert cfg.smart_target_window_size == 20
        assert cfg.smart_target_exclude_self is False
        assert cfg.smart_target_fallback == "en"
        assert cfg.smart_target_min_samples == 5
        assert cfg.llm_base_url == "https://api.example.com/v1"
        assert cfg.llm_model == "test-model"
        assert cfg.llm_temperature == 0.5
        assert cfg.llm_timeout == 60
        assert cfg.llm_formality == "high"
        assert cfg.llm_style == "standard"
        assert cfg.llm_extra_body_json == '{"custom": true}'
        assert cfg.llm_parallel_fastest_mode == "final_only"
        assert cfg.use_international_endpoint is True
