"""Tests for ContextAwareTranslator and SmartTargetLanguageSelector."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from streaming_translation import (
    BaseTranslationAPI,
    ContextAwareTranslator,
    SmartTargetLanguageSelector,
    TranslationConfig,
    TranslationHistoryEntry,
)


# ── Helpers ───────────────────────────────────────────────────────────

class MockNativeAPI(BaseTranslationAPI):
    SUPPORTS_CONTEXT = True
    def __init__(self):
        self.calls = []
    def translate(self, text, source_language="auto", target_language="zh-CN",
                  context=None, context_pairs=None, **kwargs):
        self.calls.append(dict(text=text, target=target_language,
                               context=context, pairs=context_pairs, kwargs=kwargs))
        return f"[{target_language}] {text}"


class MockNonNativeAPI(BaseTranslationAPI):
    SUPPORTS_CONTEXT = False
    def __init__(self):
        self.calls = []
    def translate(self, text, source_language="auto", target_language="zh-CN", **kwargs):
        self.calls.append(dict(text=text, target=target_language, kwargs=kwargs))
        return f"TR-{target_language}: {text}"


# ── ContextAwareTranslator ────────────────────────────────────────────

class TestContextAwareTranslatorInit:
    def test_default_construction(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        assert t.translation_api is api
        assert t.max_context_size == 6
        assert t.target_language == "zh-CN"
        assert t.context_aware is True

    def test_custom_params(self):
        api = MockNonNativeAPI()
        t = ContextAwareTranslator(api, api_name="custom", max_context_size=3,
                                   target_language="en", context_aware=False)
        assert t.api_name == "custom"
        assert t.max_context_size == 3
        assert t.target_language == "en"
        assert t.context_aware is False


class TestContextAwareTranslatorTranslate:
    def test_empty_text_returns_empty(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        assert t.translate("") == ""
        assert t.translate("   ") == ""
        assert len(api.calls) == 0

    def test_native_context_first_call(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, target_language="ja")
        result = t.translate("Hello", context_prefix="VRChat:")
        assert result == "[ja] Hello"
        assert len(api.calls) == 1
        call = api.calls[0]
        assert call["target"] == "ja"
        assert "VRChat:" in call["context"]
        assert call["pairs"] is None

    def test_native_context_builds_history(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, target_language="ja")
        t.translate("First", context_prefix="Test:", record_history=True)
        t.translate("Second", context_prefix="Test:")
        assert len(api.calls) == 2
        second = api.calls[1]
        assert len(second["pairs"]) == 1
        assert second["pairs"][0]["source"] == "[Me] First"
        assert second["pairs"][0]["target"] == "[ja] First"
        assert "Second" not in second["context"]

    def test_non_native_context(self):
        api = MockNonNativeAPI()
        t = ContextAwareTranslator(api, target_language="de")
        result = t.translate("Bonjour le monde")
        assert "Bonjour le monde" in result
        assert len(api.calls) == 1

    def test_native_context_aware_disabled(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, target_language="ja", context_aware=False)
        result = t.translate("Hello", context_prefix="VRChat:")
        assert result == "[ja] Hello"
        call = api.calls[0]
        assert call["context"] is None
        assert call["pairs"] is None

    def test_record_history_false(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, target_language="ja")
        t.translate("First", record_history=False)
        t.translate("Second", record_history=True)
        assert len(t.get_contexts()) == 1
        assert t.get_contexts()[0]["source"] == "Second"

    def test_partial_result_not_recorded(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, target_language="ja")
        t.translate("Partial", is_partial=True, record_history=True)
        assert len(t.get_contexts()) == 0

    def test_error_not_recorded(self):
        class ErrorAPI(BaseTranslationAPI):
            SUPPORTS_CONTEXT = True
            def translate(self, **kw):
                return "[ERROR] something went wrong"
        api = ErrorAPI()
        t = ContextAwareTranslator(api)
        result = t.translate("Test")
        assert result.startswith("[ERROR]")
        assert len(t.get_contexts()) == 0

    def test_api_exception_handled(self):
        class CrashAPI(BaseTranslationAPI):
            SUPPORTS_CONTEXT = True
            def translate(self, **kw):
                raise RuntimeError("API crashed")
        api = CrashAPI()
        t = ContextAwareTranslator(api)
        result = t.translate("Test")
        assert result.startswith("[ERROR]")

    def test_non_native_error_not_recorded(self):
        api = MockNonNativeAPI()
        t = ContextAwareTranslator(api, target_language="de")
        t.translate("Good")
        api.translate = lambda **kw: "[ERROR] fail"
        t.translate("Bad")
        assert len(t.get_contexts()) == 1


class TestContextAwareTranslatorHistory:
    def test_add_external_speech(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        t.add_external_speech("Someone said something")
        entries = t.get_contexts()
        assert len(entries) == 1
        assert entries[0]["source"] == "Someone said something"
        assert entries[0]["speaker"] == "others"

    def test_add_external_speech_empty(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        t.add_external_speech("")
        assert len(t.get_contexts()) == 0
        t.add_external_speech("  ")
        assert len(t.get_contexts()) == 0

    def test_append_history_entry(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        t.append_history_entry("source", "target", "fr")
        entries = t.get_contexts()
        assert len(entries) == 1
        assert entries[0]["source"] == "source"
        assert entries[0]["translated"] == "target"
        assert entries[0]["language"] == "fr"

    def test_append_history_entry_skips_empty(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        t.append_history_entry("", "target")
        t.append_history_entry("src", "")
        t.append_history_entry("src", "[ERROR] fail")
        assert len(t.get_contexts()) == 0

    def test_append_history_entry_skips_none(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        t.append_history_entry(None, "target")
        t.append_history_entry("src", None)
        t.append_history_entry("src", "[ERROR] boom")
        assert len(t.get_contexts()) == 0

    def test_context_window_limit(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, max_context_size=3)
        for i in range(5):
            t.translate(f"Message {i}")
        assert len(t.get_contexts()) == 3
        assert t.get_contexts()[0]["source"] == "Message 4"
        assert t.get_contexts()[-1]["source"] == "Message 2"

    def test_clear_contexts(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        t.translate("Hello")
        t.translate("World")
        assert len(t.get_contexts()) == 2
        t.clear_contexts()
        assert len(t.get_contexts()) == 0

    def test_display_contexts_reversed_order(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api)
        t.translate("First")
        t.translate("Second")
        t.translate("Third")
        display = t.display_contexts
        assert display[0].source_text == "Third"
        assert display[1].source_text == "Second"
        assert display[2].source_text == "First"


class TestContextAwareTranslatorTranslateWithContext:
    def test_returns_translation_and_info(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, target_language="ja")
        translated, info = t.translate_with_context("Hello")
        assert translated == "[ja] Hello"
        assert info["contexts_count"] == 1


class TestContextAwareTranslatorSetters:
    def test_set_context_aware(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, context_aware=True)
        t.set_context_aware(False)
        assert t.context_aware is False
        t.set_context_aware(True)
        assert t.context_aware is True

    def test_set_target_language(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, target_language="en")
        t.set_target_language("fr")
        assert t.target_language == "fr"

    def test_repr(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, api_name="TestAPI")
        r = repr(t)
        assert "TestAPI" in r
        assert "context_aware=True" in r


class TestContextAwareTranslatorThreadSafety:
    def test_concurrent_translate(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, max_context_size=100)
        errors = []

        def do_translate(i):
            try:
                t.translate(f"Message {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_translate, args=(i,)) for i in range(20)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0
        assert len(t.get_contexts()) == 20

    def test_concurrent_clear_and_translate(self):
        api = MockNativeAPI()
        t = ContextAwareTranslator(api, max_context_size=100)
        errors = []

        def writer():
            for i in range(50):
                t.translate(f"Msg {i}")

        def clearer():
            for _ in range(10):
                t.clear_contexts()

        threads = [threading.Thread(target=writer), threading.Thread(target=clearer)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        # no crash = pass
        assert True


# ── SmartTargetLanguageSelector ───────────────────────────────────────

class TestSmartTargetLanguageSelector:
    def test_default_disabled_returns_empty(self):
        cfg = TranslationConfig()
        sel = SmartTargetLanguageSelector(cfg)
        assert sel.select_target_language() == []

    def test_primary_enabled_no_history_returns_fallback(self):
        cfg = TranslationConfig(smart_target_primary_enabled=True)
        sel = SmartTargetLanguageSelector(cfg)
        result = sel.select_target_language()
        assert result == ["en"]

    def test_most_common_strategy(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_strategy="most_common",
        )
        sel = SmartTargetLanguageSelector(cfg)
        for lang in ["ja", "ja", "en", "ja", "ko"]:
            sel.record_language(lang)
        result = sel.select_target_language()
        assert result[0] == "ja"

    def test_latest_strategy(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_strategy="latest",
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("ja")
        sel.record_language("ko")
        sel.record_language("en")
        result = sel.select_target_language()
        assert result[0] == "en"

    def test_weighted_strategy(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_strategy="weighted",
            smart_target_window_size=5,
            smart_target_count=1,
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("ja")
        sel.record_language("ja")
        sel.record_language("en")
        sel.record_language("en")
        sel.record_language("ko")
        result = sel.select_target_language()
        # "en" has highest cumulative weight (appears twice later)
        assert result[0] == "en"

    def test_exclude_self_language(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_exclude_self=True,
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("ja")
        sel.record_language("en")
        sel.record_language("ja")
        result = sel.select_target_language(self_language="ja")
        assert result[0] == "en"

    def test_exclude_self_removes_all_self(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_exclude_self=True,
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("ja")
        sel.record_language("JA")
        result = sel.select_target_language(self_language="ja")
        assert result == ["en"]  # fallback

    def test_all_history_excluded_returns_fallback(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_exclude_self=True,
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("en")
        sel.record_language("en")
        result = sel.select_target_language(self_language="en")
        assert result == ["en"]

    def test_secondary_language_output(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_secondary_enabled=True,
            smart_target_count=2,
        )
        sel = SmartTargetLanguageSelector(cfg)
        for lang in ["ja", "ja", "en", "en"]:
            sel.record_language(lang)
        result = sel.select_target_language()
        assert len(result) == 2
        # both have count 2; tiebreaker is -last_index: en(3) < ja(1) ⇒ en first
        assert result[0] == "en"
        assert result[1] == "ja"

    def test_manual_secondary(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_secondary_enabled=True,
            smart_target_count=2,
            smart_target_manual_secondary="ko",
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("ja")
        result = sel.select_target_language()
        assert result == ["ja", "ko"]

    def test_clear_history(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_count=1,
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("ja")
        assert len(sel.select_target_language()) == 1
        sel.clear_history()
        assert sel.select_target_language() == ["en"]

    def test_window_size_respected(self):
        cfg = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_window_size=2,
        )
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("ja")
        sel.record_language("ko")
        sel.record_language("en")
        result = sel.select_target_language()
        assert result[0] == "en"  # only ["ko", "en"] in window

    def test_empty_language_not_recorded(self):
        cfg = TranslationConfig(smart_target_primary_enabled=True)
        sel = SmartTargetLanguageSelector(cfg)
        sel.record_language("")
        sel.record_language(None)
        assert len(sel.select_target_language()) == 1
        assert sel.select_target_language() == ["en"]

    def test_reload_config_updates_settings(self):
        cfg1 = TranslationConfig(
            smart_target_primary_enabled=False,
            smart_target_count=1,
        )
        cfg2 = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_strategy="latest",
            smart_target_count=1,
        )
        sel = SmartTargetLanguageSelector(cfg1)
        assert sel.select_target_language() == []

        sel.reload_config(cfg2)
        assert sel.select_target_language() == ["en"]

        sel.record_language("ja")
        sel.record_language("ko")
        result = sel.select_target_language()
        assert result[0] == "ko"

    def test_reload_config_preserves_history(self):
        cfg1 = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_count=1,
        )
        cfg2 = TranslationConfig(
            smart_target_primary_enabled=True,
            smart_target_strategy="latest",
            smart_target_count=1,
            smart_target_window_size=100,
        )
        sel = SmartTargetLanguageSelector(cfg1)
        sel.record_language("ja")
        sel.record_language("ko")

        sel.reload_config(cfg2)
        assert len(sel._history) == 2
        assert sel._history.maxlen == 100
