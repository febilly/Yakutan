"""
Integration tests — make real API calls using credentials from ``.env``.

Marked with ``pytest.mark.integration``; skipped automatically when the
relevant API key is not present in the environment.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from streaming_translation import (
    ContextAwareTranslator,
    DeepLAPI,
    QwenMTAPI,
    TranslationConfig,
)
from streaming_translation.api.openrouter import OpenRouterAPI

# ── Helpers ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv_if_needed():
    """Load .env manually so tests see env vars even without dotenv auto-load."""
    if not DOTENV_PATH.exists():
        return
    for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


load_dotenv_if_needed()

has_deepl = bool(os.environ.get("DEEPL_API_KEY"))
has_dashscope = bool(os.environ.get("DASHSCOPE_API_KEY"))
has_llm = bool(os.environ.get("LLM_API_KEY")) and bool(os.environ.get("LLM_BASE_URL"))

integration = pytest.mark.skipif(
    not (has_deepl or has_dashscope or has_llm),
    reason="No API keys found in .env — skipping integration tests",
)
needs_deepl = pytest.mark.skipif(not has_deepl, reason="DEEPL_API_KEY not set")
needs_dashscope = pytest.mark.skipif(not has_dashscope, reason="DASHSCOPE_API_KEY not set")
needs_llm = pytest.mark.skipif(
    not has_llm,
    reason="LLM_API_KEY and LLM_BASE_URL not both set",
)

TEST_SENTENCES = [
    ("Hello, how are you?", "en"),
    ("Good morning, nice to meet you.", "en"),
    ("The weather is beautiful today.", "en"),
]


# ── DeepL API ─────────────────────────────────────────────────────────

@pytest.mark.integration
@needs_deepl
class TestDeepLIntegration:
    def test_translate_to_japanese(self):
        api = DeepLAPI(proxy_url=None)
        result = api.translate("Hello, how are you?", target_language="JA")
        assert result and not result.startswith("[ERROR]")
        assert "元気" in result or "こんにちは" in result

    def test_translate_to_chinese(self):
        api = DeepLAPI(proxy_url=None)
        result = api.translate("Good morning", target_language="zh-CN")
        assert result and not result.startswith("[ERROR]")
        assert "早上" in result or "早安" in result or "上午" in result

    def test_translate_to_german(self):
        api = DeepLAPI(proxy_url=None)
        result = api.translate("How are you?", target_language="DE")
        assert result and not result.startswith("[ERROR]")
        assert "geht" in result.lower() or "wie" in result.lower()

    def test_translate_multiple_sentences(self):
        api = DeepLAPI(proxy_url=None)
        for text, src_lang in TEST_SENTENCES:
            result = api.translate(text, source_language=src_lang, target_language="JA")
            assert result and not result.startswith("[ERROR]"), f"Failed on: {text}"

    def test_with_context(self):
        api = DeepLAPI(proxy_url=None)
        result = api.translate(
            "See you later!",
            context="We just finished a meeting.",
            target_language="JA",
        )
        assert result and not result.startswith("[ERROR]")

    def test_context_aware_translator_wrapper(self):
        api = DeepLAPI(proxy_url=None)
        translator = ContextAwareTranslator(
            api, target_language="DE", max_context_size=3,
        )
        r1 = translator.translate("Hello, how are you?")
        assert r1 and not r1.startswith("[ERROR]")
        r2 = translator.translate("I am fine, thank you.")
        assert r2 and not r2.startswith("[ERROR]")
        assert len(translator.get_contexts()) == 2

    def test_language_code_variants(self):
        api = DeepLAPI(proxy_url=None)
        for code in ("JA", "ja", "DE", "de", "ZH", "zh"):
            result = api.translate("Hello", target_language=code)
            assert result and not result.startswith("[ERROR]")


# ── Qwen-MT (DashScope) API ───────────────────────────────────────────

@pytest.mark.integration
@needs_dashscope
class TestQwenMTIntegration:
    def test_translate_to_japanese(self):
        api = QwenMTAPI(proxy_url=None)
        result = api.translate("Hello, how are you?", target_language="ja")
        assert result and not result.startswith("[ERROR]")
        assert "こんにちは" in result or "元気" in result

    def test_translate_to_english(self):
        api = QwenMTAPI(proxy_url=None)
        result = api.translate("你好，今天天气真好", target_language="en")
        assert result and not result.startswith("[ERROR]")
        assert "weather" in result.lower()

    def test_translate_multiple_sentences(self):
        api = QwenMTAPI(proxy_url=None)
        for text, src_lang in TEST_SENTENCES:
            result = api.translate(text, source_language=src_lang, target_language="zh")
            assert result and not result.startswith("[ERROR]"), f"Failed on: {text}"

    def test_with_context_pairs(self):
        api = QwenMTAPI(proxy_url=None)
        result = api.translate(
            "I like it too.",
            context_pairs=[{"source": "Do you like pizza?", "target": "你喜欢披萨吗？"}],
            target_language="zh",
        )
        assert result and not result.startswith("[ERROR]")

    def test_context_aware_wrapper(self):
        api = QwenMTAPI(proxy_url=None)
        translator = ContextAwareTranslator(
            api, target_language="ja", max_context_size=3,
        )
        r1 = translator.translate("Hello")
        assert r1 and not r1.startswith("[ERROR]")
        r2 = translator.translate("How are you?")
        assert r2 and not r2.startswith("[ERROR]")
        assert len(translator.get_contexts()) == 2


# ── OpenRouter (LLM) API ──────────────────────────────────────────────

@pytest.mark.integration
@needs_llm
class TestOpenRouterIntegration:
    def _make_api(self):
        extra_body = os.environ.get("OPENAI_COMPAT_EXTRA_BODY_JSON", "")
        return OpenRouterAPI(
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ.get("LLM_MODEL", "qwen3.5-plus"),
            api_key=os.environ["LLM_API_KEY"],
            timeout=120,
            extra_body_json=extra_body,
            proxy_url=None,
        )

    def test_translate_to_japanese(self):
        api = self._make_api()
        result = api.translate("Hello, how are you?", target_language="ja")
        assert result and not result.startswith("[ERROR]")

    def test_translate_to_chinese(self):
        api = self._make_api()
        result = api.translate("Good morning", target_language="zh-CN")
        assert result and not result.startswith("[ERROR]")

    def test_context_aware_translator(self):
        api = self._make_api()
        translator = ContextAwareTranslator(
            api, target_language="ja", max_context_size=3,
        )
        r1 = translator.translate("Hello, how are you?")
        assert r1 and not r1.startswith("[ERROR]")
        r2 = translator.translate("I am fine, thank you.")
        assert r2 and not r2.startswith("[ERROR]")
        assert len(translator.get_contexts()) == 2


# ─── Smoke test — at least one backend works ─────────────────────

@pytest.mark.integration
@integration
class TestTranslationPipelineSmoke:
    """Verify that at least one real translation backend works end-to-end."""

    def test_reinitialize_translator(self):
        from streaming_translation import reinitialize_translator

        class MockState:
            pass

        state = MockState()

        if has_deepl:
            cfg = TranslationConfig(
                target_language="JA",
                translation_api_type="deepl",
            )
        elif has_dashscope:
            cfg = TranslationConfig(
                target_language="ja",
                translation_api_type="qwen_mt",
            )
        else:
            pytest.skip("No API key available for smoke test")

        reinitialize_translator(state, cfg)

        assert state.translator is not None
        # Try a real translation
        result = state.translator.translate("Hello world")
        assert result and not result.startswith("[ERROR]")

    def test_reverse_translation(self):
        from streaming_translation import reverse_translation

        if not has_deepl:
            pytest.skip("DeepL needed for reverse translation test")

        api = DeepLAPI(proxy_url=None)
        bt = ContextAwareTranslator(
            api, target_language="en", context_aware=False,
        )
        result = reverse_translation(bt, "Hola mundo", "es", "en")
        assert result is not None
        assert not result.startswith("[ERROR]")
