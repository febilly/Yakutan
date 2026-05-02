"""Tests for simultaneous dual-target-language translation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
if DOTENV_PATH.exists():
    for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"").strip()
        if key and key not in os.environ:
            os.environ[key] = value

has_deepl = bool(os.environ.get("DEEPL_API_KEY"))


class TestDualTargetPipeline:
    """Verify both translators are created and can translate independently."""

    def test_reinitialize_with_dual_target(self):
        from streaming_translation import (
            TranslationConfig,
            config_from_module,
            reinitialize_translator,
        )

        class MockState:
            pass

        state = MockState()
        cfg = TranslationConfig(
            target_language="ja",
            secondary_target_language="ko",
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
        )

        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            reinitialize_translator(state, cfg)

        assert state.translator is not None, "Primary translator not created"
        assert state.secondary_translator is not None, "Secondary translator not created"
        assert state.target_language == "ja"
        assert state.secondary_target_language == "ko"

    def test_both_translators_translate_independently(self):
        from streaming_translation import (
            BaseTranslationAPI,
            ContextAwareTranslator,
            TranslationConfig,
        )

        class MockDualAPI(BaseTranslationAPI):
            SUPPORTS_CONTEXT = True
            def __init__(self, tag=""):
                self.tag = tag
                self.calls = []
            def translate(self, text, **kw):
                tgt = kw.get("target_language", "?")
                self.calls.append((text, tgt))
                return f"[{tgt}] {text}"

        primary_api = MockDualAPI()
        secondary_api = MockDualAPI()

        primary = ContextAwareTranslator(primary_api, target_language="ja")
        secondary = ContextAwareTranslator(secondary_api, target_language="ko")

        r1 = primary.translate("Hello", source_language="en")
        r2 = secondary.translate("Hello", source_language="en")

        assert r1 == "[ja] Hello"
        assert r2 == "[ko] Hello"
        assert len(primary_api.calls) == 1
        assert len(secondary_api.calls) == 1

    def test_dual_translators_maintain_separate_contexts(self):
        from streaming_translation import (
            BaseTranslationAPI,
            ContextAwareTranslator,
        )

        class MockAPI(BaseTranslationAPI):
            SUPPORTS_CONTEXT = True
            def translate(self, text, **kw):
                return kw.get("target_language", "?") + ": " + text

        api1 = MockAPI()
        api2 = MockAPI()

        t1 = ContextAwareTranslator(api1, target_language="ja", max_context_size=3)
        t2 = ContextAwareTranslator(api2, target_language="ko", max_context_size=3)

        for i in range(3):
            t1.translate(f"msg {i}")
            t2.translate(f"MSG {i}")

        ctx1 = t1.get_contexts()
        ctx2 = t2.get_contexts()

        assert len(ctx1) == 3
        assert len(ctx2) == 3
        assert ctx1[0]["source"].startswith("msg")
        assert ctx2[0]["source"].startswith("MSG")

    def test_dual_translate_with_backend(self):
        from streaming_translation import (
            BaseTranslationAPI,
            ContextAwareTranslator,
            translate_with_backend,
        )

        class MockAPI(BaseTranslationAPI):
            SUPPORTS_CONTEXT = True
            def translate(self, text, **kw):
                tgt = kw.get("target_language", "?")
                return f"[{tgt}] {text}"

        p = ContextAwareTranslator(MockAPI(), target_language="ja")
        s = ContextAwareTranslator(MockAPI(), target_language="ko")

        r1 = translate_with_backend(p, None, "hello", "ja",
                                    source_language="en", context_prefix="")
        r2 = translate_with_backend(s, None, "hello", "ko",
                                    source_language="en", context_prefix="")

        assert r1 == "[ja] hello"
        assert r2 == "[ko] hello"

    def test_secondary_ensure_recreates_on_language_change(self):
        from streaming_translation import (
            TranslationConfig,
            config_from_module,
            ensure_secondary_translator,
        )

        class MockState:
            def __init__(self):
                self.secondary_translator = None
                self.secondary_translation_api = None
                self.secondary_deepl_fallback_translator = None
                self.secondary_deepl_fallback_translation_api = None
                self.secondary_target_language = None

        cfg = TranslationConfig(
            translation_api_type="qwen_mt",
            dashscope_api_key="test-key",
        )
        state = MockState()

        with patch("streaming_translation.api.qwen_mt.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            ok1 = ensure_secondary_translator(state, "ja", cfg)
            assert ok1 is True
            assert state.secondary_target_language == "ja"

            ok2 = ensure_secondary_translator(state, "de", cfg)
            assert ok2 is True
            assert state.secondary_target_language == "de"

            ok3 = ensure_secondary_translator(state, "de", cfg)
            assert ok3 is True

    def test_secondary_removed_when_target_none(self):
        from streaming_translation import (
            TranslationConfig,
            ensure_secondary_translator,
        )

        class MockState:
            def __init__(self):
                self.secondary_translator = MagicMock()
                self.secondary_translation_api = MagicMock()
                self.secondary_deepl_fallback_translator = MagicMock()
                self.secondary_deepl_fallback_translation_api = MagicMock()
                self.secondary_target_language = "ja"

        cfg = TranslationConfig()
        state = MockState()
        result = ensure_secondary_translator(state, None, cfg)
        assert result is False
        assert state.secondary_translator is None
        assert state.secondary_target_language is None


class TestTranslateWithBackendCallPattern:
    def test_no_duplicate_arg_error(self):
        from streaming_translation import (
            BaseTranslationAPI,
            ContextAwareTranslator,
            translate_with_backend,
        )

        class Mock(BaseTranslationAPI):
            SUPPORTS_CONTEXT = True
            def translate(self, text, **kw):
                sl = kw.get("source_language", "NOT_SET")
                detected = kw.get("detected_source_language", "NOT_SET")
                return f"sl={sl} detected={detected} text={text}"

        t = ContextAwareTranslator(Mock(), target_language="ja")
        result = translate_with_backend(
            t, None, "hello", "ja", None, False, None, "en",
            source_language="auto", context_prefix="VRChat:",
            record_history=False,
        )
        assert not result.startswith("[ERROR]")
        assert "sl=auto" in result
        assert "detected=en" in result


@pytest.mark.integration
@pytest.mark.skipif(not has_deepl, reason="DEEPL_API_KEY not set")
class TestDualTargetIntegration:
    """Real API dual-target translation tests using DeepL."""

    def test_dual_target_real_translation(self):
        from streaming_translation import (
            ContextAwareTranslator,
            DeepLAPI,
            translate_with_backend,
        )

        primary_api = DeepLAPI(proxy_url=None)
        secondary_api = DeepLAPI(proxy_url=None)

        primary = ContextAwareTranslator(primary_api, target_language="JA")
        secondary = ContextAwareTranslator(secondary_api, target_language="DE")

        text = "Hello, how are you?"
        r1 = translate_with_backend(
            primary, None, text, "JA",
            source_language="en", context_prefix="",
        )
        r2 = translate_with_backend(
            secondary, None, text, "DE",
            source_language="en", context_prefix="",
        )

        assert r1 and not r1.startswith("[ERROR]"), f"Primary failed: {r1}"
        assert r2 and not r2.startswith("[ERROR]"), f"Secondary failed: {r2}"
        assert r1 != r2, "Translations should differ"
        assert any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in r1), \
            f"Not Japanese: {r1}"
        assert all(ord(c) < 0x2e80 for c in r2), f"Not German: {r2}"

    def test_dual_target_context_aware(self):
        from streaming_translation import (
            ContextAwareTranslator,
            DeepLAPI,
        )

        primary_api = DeepLAPI(proxy_url=None)
        secondary_api = DeepLAPI(proxy_url=None)

        primary = ContextAwareTranslator(primary_api, target_language="JA",
                                        max_context_size=3)
        secondary = ContextAwareTranslator(secondary_api, target_language="DE",
                                          max_context_size=3)

        sentences = [
            ("Hello!", "こんにちは！", "Hallo!"),
            ("How are you?", "お元気ですか？", "Wie geht es dir?"),
        ]

        for src, exp_ja_hint, exp_de_hint in sentences:
            r1 = primary.translate(src)
            r2 = secondary.translate(src)
            assert r1 and not r1.startswith("[ERROR]"), f"JA failed for: {src}"
            assert r2 and not r2.startswith("[ERROR]"), f"DE failed for: {src}"

        assert len(primary.get_contexts()) == len(sentences)
        assert len(secondary.get_contexts()) == len(sentences)

    def test_individual_contexts_independent(self):
        from streaming_translation import ContextAwareTranslator, DeepLAPI

        api = DeepLAPI(proxy_url=None)
        t1 = ContextAwareTranslator(api, target_language="JA", max_context_size=3)
        t2 = ContextAwareTranslator(api, target_language="FR", max_context_size=3)

        # Vary the number of translations
        t1.translate("Hello")
        t1.translate("World")
        t2.translate("Hello")

        assert len(t1.get_contexts()) == 2
        assert len(t2.get_contexts()) == 1

    def test_dual_with_secondary_update(self):
        from streaming_translation import DeepLAPI, ContextAwareTranslator

        api = DeepLAPI(proxy_url=None)
        t = ContextAwareTranslator(api, target_language="JA")
        result = t.translate("Good morning")
        assert result and not result.startswith("[ERROR]")
