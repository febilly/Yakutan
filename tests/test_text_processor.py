"""Tests for text_processor — language codes, formatting, furigana/pinyin."""

from __future__ import annotations

from unittest.mock import patch

from text_processor import (
    ARABIC_OSC_LINE_MAX_CHARS,
    ARABIC_PDI,
    ARABIC_RLI,
    apply_arabic_reshaper_if_needed,
    build_dual_output_display,
    build_streaming_output_line,
    get_display_translation_text,
    get_display_text,
    has_secondary_translation_target,
    language_code_for_osc_tag,
    limit_dual_output_text,
    normalize_lang_code,
    normalize_optional_language_code,
    resolve_output_target_language,
    sanitize_text_fancy_style,
    add_furigana_if_needed,
    add_pinyin_if_needed,
)


class _FakeArabicReshaper:
    @staticmethod
    def reshape(text):
        return f"reshaped:{text}"


class _IdentityArabicReshaper:
    @staticmethod
    def reshape(text):
        return text


ARABIC_HELLO = "\u0645\u0631\u062d\u0628\u0627"


class TestNormalizeOptionalLanguageCode:
    def test_none(self):
        assert normalize_optional_language_code(None) is None

    def test_empty(self):
        assert normalize_optional_language_code("") is None

    def test_whitespace(self):
        assert normalize_optional_language_code("  ") is None

    def test_valid(self):
        assert normalize_optional_language_code("ja") == "ja"
        assert normalize_optional_language_code(" zh-CN ") == "zh-CN"


class TestNormalizeLangCode:
    def test_none_or_empty_returns_auto(self):
        assert normalize_lang_code(None) == "auto"
        assert normalize_lang_code("") == "auto"

    def test_auto_passthrough(self):
        assert normalize_lang_code("auto") == "auto"

    def test_english_variants(self):
        for code in ("en", "en-us", "en-gb", "en-au", "en-ca"):
            assert normalize_lang_code(code) == "en", f"failed for {code}"

    def test_traditional_chinese(self):
        for code in ("zh-tw", "zh-hant", "zh-hk", "zh-mo"):
            assert normalize_lang_code(code) == "zh-hant", f"failed for {code}"

    def test_simplified_chinese(self):
        for code in ("zh", "zh-cn", "zh-hans", "zh-sg", "cmn", "wuu", "yue"):
            assert normalize_lang_code(code) == "zh-hans", f"failed for {code}"

    def test_japanese(self):
        assert normalize_lang_code("ja") == "ja"
        assert normalize_lang_code("ja-jp") == "ja-jp"

    def test_korean(self):
        assert normalize_lang_code("ko") == "ko"

    def test_underscore_to_hyphen(self):
        assert normalize_lang_code("zh_CN") == "zh-hans"
        assert normalize_lang_code("zh_TW") == "zh-hant"


class TestLanguageCodeForOscTag:
    def test_none_returns_auto(self):
        assert language_code_for_osc_tag(None) == "auto"

    def test_auto(self):
        assert language_code_for_osc_tag("auto") == "auto"

    def test_chinese_all_variants_to_zh(self):
        for code in ("zh", "zh-cn", "zh-hans", "zh-tw", "zh-hant", "zh-hk", "cmn", "wuu", "yue"):
            assert language_code_for_osc_tag(code) == "zh", f"failed for {code}"

    def test_english(self):
        assert language_code_for_osc_tag("en") == "en"
        assert language_code_for_osc_tag("en-us") == "en"

    def test_japanese(self):
        assert language_code_for_osc_tag("ja") == "ja"
        assert language_code_for_osc_tag("ja-jp") == "ja"

    def test_strips_region(self):
        assert language_code_for_osc_tag("fr-ca") == "fr"
        assert language_code_for_osc_tag("de-de") == "de"

    def test_empty_string(self):
        assert language_code_for_osc_tag("") == "auto"

    def test_whitespace_string(self):
        assert language_code_for_osc_tag("  ") == "auto"


class TestResolveOutputTargetLanguage:
    @patch("text_processor.config")
    def test_no_target(self, mock_config):
        mock_config.FALLBACK_LANGUAGE = "en"
        assert resolve_output_target_language("en", None) is None

    @patch("text_processor.config")
    def test_source_differs_from_target(self, mock_config):
        mock_config.FALLBACK_LANGUAGE = "en"
        result = resolve_output_target_language("ja", "zh-CN")
        assert result == "zh-CN"

    @patch("text_processor.config")
    def test_source_same_as_target_uses_fallback(self, mock_config):
        mock_config.FALLBACK_LANGUAGE = "en"
        result = resolve_output_target_language("ja", "ja")
        assert result == "en"

    @patch("text_processor.config")
    def test_source_same_without_fallback_returns_target(self, mock_config):
        """Without fallback, same source/target returns the target unchanged."""
        mock_config.FALLBACK_LANGUAGE = None
        result = resolve_output_target_language("ja", "ja")
        assert result == "ja"


class TestHasSecondaryTranslationTarget:
    @patch("text_processor.config")
    def test_has_secondary(self, mock_config):
        mock_config.SECONDARY_TARGET_LANGUAGE = "en"
        assert has_secondary_translation_target() is True

    @patch("text_processor.config")
    def test_no_secondary(self, mock_config):
        mock_config.SECONDARY_TARGET_LANGUAGE = None
        assert has_secondary_translation_target() is False


class TestSanitizeTextFancyStyle:
    def test_valid_styles(self):
        assert sanitize_text_fancy_style("none") == "none"
        assert sanitize_text_fancy_style("smallCaps") == "smallCaps"
        assert sanitize_text_fancy_style("curly") == "curly"

    def test_invalid_style_defaults_to_none(self):
        assert sanitize_text_fancy_style("invalid") == "none"
        assert sanitize_text_fancy_style(None) == "none"


class TestBuildStreamingOutputLine:
    def test_appends_ellipsis(self):
        result = build_streaming_output_line("Hello")
        assert result == "Hello……"

    def test_preserves_existing_ellipsis(self):
        result = build_streaming_output_line("Hello……")
        assert result == "Hello……"

    def test_empty_text(self):
        result = build_streaming_output_line("")
        assert result == "……"

    def test_applies_post_processing(self):
        result = build_streaming_output_line("  Hello  ")
        assert "Hello" in result


class TestLimitDualOutputText:
    @patch("text_processor.config")
    def test_under_limit(self, mock_config):
        mock_config.OSC_TEXT_MAX_LENGTH = ""
        mock_config.OSC_TEXT_MAX_BYTES = ""
        result = limit_dual_output_text("short", max_chars=100)
        assert result == "short"

    @patch("text_processor.config")
    def test_over_limit(self, mock_config):
        mock_config.OSC_TEXT_MAX_LENGTH = ""
        mock_config.OSC_TEXT_MAX_BYTES = ""
        result = limit_dual_output_text("hello world", max_chars=5)
        assert result == "world"

    @patch("text_processor.config")
    def test_over_limit_drops_prefix_at_clause_punctuation(self, mock_config):
        mock_config.OSC_TEXT_MAX_LENGTH = ""
        mock_config.OSC_TEXT_MAX_BYTES = ""
        result = limit_dual_output_text(
            "\u524d\u534a\u6bb5\u8981\u4e22\u6389\uff0c\u540e\u534a\u6bb5\u8981\u4fdd\u7559",
            max_chars=8,
        )
        assert result == "\u540e\u534a\u6bb5\u8981\u4fdd\u7559"

    @patch("text_processor.config")
    def test_no_max_chars(self, mock_config):
        mock_config.OSC_TEXT_MAX_LENGTH = ""
        mock_config.OSC_TEXT_MAX_BYTES = ""
        mock_config.get_effective_osc_text_max_length.return_value = None
        result = limit_dual_output_text("hello", max_chars=None)
        assert result == "hello"


class TestBuildDualOutputDisplay:
    @patch("text_processor.config")
    def test_single_output(self, mock_config):
        mock_config.OSC_TEXT_MAX_LENGTH = ""
        mock_config.OSC_TEXT_MAX_BYTES = ""
        mock_config.get_effective_osc_text_max_length.return_value = None
        result = build_dual_output_display("hello", None)
        assert result == "hello"

    @patch("text_processor.config")
    def test_dual_output(self, mock_config):
        mock_config.OSC_TEXT_MAX_LENGTH = ""
        mock_config.OSC_TEXT_MAX_BYTES = ""
        mock_config.get_effective_osc_text_max_length.return_value = None
        result = build_dual_output_display("hello", "world")
        assert result == "hello\nworld"

    @patch("text_processor.config")
    def test_dual_output_trimmed(self, mock_config):
        mock_config.OSC_TEXT_MAX_LENGTH = ""
        mock_config.OSC_TEXT_MAX_BYTES = ""
        mock_config.get_effective_osc_text_max_length.return_value = 10
        result = build_dual_output_display("hello there", "world", "en", "en")
        assert len(result) <= 10


class TestGetDisplayText:
    @patch("text_processor.config")
    def test_no_language_no_change(self, mock_config):
        mock_config.ENABLE_JA_FURIGANA = False
        mock_config.ENABLE_ZH_PINYIN = False
        result = get_display_text("hello", language=None)
        assert result == "hello"

    @patch("text_processor.config")
    def test_with_language_no_annotations(self, mock_config):
        mock_config.ENABLE_JA_FURIGANA = False
        mock_config.ENABLE_ZH_PINYIN = False
        mock_config.REMOVE_TRAILING_PERIOD = False
        mock_config.TEXT_FANCY_STYLE = "none"
        result = get_display_text("hello", language="en")
        assert result == "hello"


class TestArabicReshaper:
    def test_display_text_keeps_unreshaped_arabic(self):
        result = get_display_translation_text(ARABIC_HELLO, "ar")
        assert result == ARABIC_HELLO

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _FakeArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: text)
    def test_applies_arabic_reshaper_and_direction_isolate(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        result = apply_arabic_reshaper_if_needed(ARABIC_HELLO, "ar")
        assert result == f"{ARABIC_RLI}reshaped:{ARABIC_HELLO}{ARABIC_PDI}"

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _FakeArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: text)
    def test_keeps_punctuation_inside_direction_isolate(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        result = apply_arabic_reshaper_if_needed(f"{ARABIC_HELLO}.", "ar")
        assert result == f"{ARABIC_RLI}reshaped:{ARABIC_HELLO}.{ARABIC_PDI}"

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _IdentityArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: f"display:{text}")
    def test_applies_bidi_per_line_for_mixed_text(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        result = apply_arabic_reshaper_if_needed(f"Hello {ARABIC_HELLO} 123", "ar")
        assert result == f"{ARABIC_RLI}display:Hello {ARABIC_HELLO} 123{ARABIC_PDI}"

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _IdentityArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: text)
    def test_wraps_arabic_text_by_words_before_processing(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        text = (
            "\u0647\u0627\u0645\u0634 \u0631\u0628\u062d\u0647 "
            "\u0645\u0631\u062a\u0641\u0639\u060c \u0623\u0648 "
            "\u0623\u0642\u0646\u0639\u0629 \u0648\u062c\u0647\u060c "
            "\u0623\u0648 \u0643\u0631\u064a\u0645 \u062d\u0644\u0627\u0642\u0629."
        )

        result = apply_arabic_reshaper_if_needed(text, "ar", max_chars=144)
        lines = result.split("\n")

        assert len(lines) > 1
        for line in lines:
            assert line.startswith(ARABIC_RLI)
            assert line.endswith(ARABIC_PDI)
            assert len(line[1:-1]) <= ARABIC_OSC_LINE_MAX_CHARS

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _IdentityArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: text)
    def test_arabic_wrapping_respects_total_budget_without_cutting_controls(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        text = " ".join([ARABIC_HELLO] * 60)

        result = apply_arabic_reshaper_if_needed(text, "ar", max_chars=144)

        assert len(result) <= 144
        for line in result.split("\n"):
            assert line.startswith(ARABIC_RLI)
            assert line.endswith(ARABIC_PDI)
            assert ARABIC_RLI not in line[1:-1]
            assert ARABIC_PDI not in line[1:-1]

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", None)
    def test_missing_reshaper_still_adds_direction_isolate(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        result = apply_arabic_reshaper_if_needed(ARABIC_HELLO, "ar")
        assert result == f"{ARABIC_RLI}{ARABIC_HELLO}{ARABIC_PDI}"

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _FakeArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: text)
    def test_does_not_process_already_isolated_text(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        already_processed = f"{ARABIC_RLI}{ARABIC_HELLO}{ARABIC_PDI}"
        result = apply_arabic_reshaper_if_needed(already_processed, "ar")
        assert result == already_processed

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _FakeArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: text)
    def test_skips_when_disabled(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = False
        result = apply_arabic_reshaper_if_needed(ARABIC_HELLO, "ar")
        assert result == ARABIC_HELLO

    @patch("text_processor.config")
    @patch("text_processor._arabic_reshaper", _FakeArabicReshaper)
    @patch("text_processor._bidi_get_display", lambda text: text)
    def test_skips_non_arabic_text(self, mock_config):
        mock_config.ENABLE_ARABIC_RESHAPER = True
        result = apply_arabic_reshaper_if_needed("hello", "en")
        assert result == "hello"


class TestGetDisplayTranslationText:
    @patch("text_processor.config")
    def test_basic(self, mock_config):
        mock_config.ENABLE_JA_FURIGANA = False
        mock_config.ENABLE_ZH_PINYIN = False
        mock_config.REMOVE_TRAILING_PERIOD = False
        mock_config.TEXT_FANCY_STYLE = "none"
        result = get_display_translation_text("こんにちは", "ja")
        assert result == "こんにちは"
