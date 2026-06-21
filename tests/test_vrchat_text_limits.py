"""Tests for the shared VRChat chatbox length limits and prefix-trimming rules.

这是「不超过上限、超长时丢弃最前面的旧文本」这条核心规则的直接单元测试，
其它模块（text_processor / osc_manager）都依赖这里的行为。
"""

from __future__ import annotations

from shared.vrchat_text_limits import (
    VRCHAT_OSC_TEXT_MAX_LENGTH,
    normalize_osc_text_max_length,
    trim_text_prefix_to_limit,
)


class TestNormalizeOscTextMaxLength:
    def test_default_is_vrchat_limit(self):
        assert VRCHAT_OSC_TEXT_MAX_LENGTH == 144

    def test_valid_int_passthrough(self):
        assert normalize_osc_text_max_length(100) == 100

    def test_numeric_string_is_parsed(self):
        assert normalize_osc_text_max_length("120") == 120

    def test_invalid_string_falls_back_to_default(self):
        assert normalize_osc_text_max_length("abc") == 144

    def test_none_falls_back_to_default(self):
        assert normalize_osc_text_max_length(None) == 144

    def test_custom_default_is_used(self):
        assert normalize_osc_text_max_length(None, default=80) == 80

    def test_zero_and_negative_clamped_to_one(self):
        assert normalize_osc_text_max_length(0) == 1
        assert normalize_osc_text_max_length(-5) == 1

    def test_float_is_truncated(self):
        assert normalize_osc_text_max_length(3.9) == 3


class TestTrimTextPrefixToLimit:
    def test_none_limit_returns_text_unchanged(self):
        assert trim_text_prefix_to_limit("hello", None) == "hello"

    def test_zero_or_negative_limit_returns_empty(self):
        assert trim_text_prefix_to_limit("hello", 0) == ""
        assert trim_text_prefix_to_limit("hello", -3) == ""

    def test_text_within_limit_unchanged(self):
        assert trim_text_prefix_to_limit("hello", 10) == "hello"
        assert trim_text_prefix_to_limit("hello", 5) == "hello"

    def test_drops_front_at_clause_punctuation(self):
        result = trim_text_prefix_to_limit("前半段要丢掉，后半段保留", 8)
        assert result == "后半段保留"
        assert len(result) <= 8

    def test_keeps_as_much_newest_text_as_fits(self):
        # 第一个能容下的边界即返回，保留尽可能多的最新文本。
        result = trim_text_prefix_to_limit("AAAA。BBBB。CC", 7)
        assert result == "BBBB。CC"

    def test_skips_boundary_whose_remainder_still_overflows(self):
        # 第一段剩余仍超长时，跳到下一个标点边界继续丢弃。
        result = trim_text_prefix_to_limit("AAAA。BBBB。CC", 6)
        assert result == "CC"

    def test_multi_char_ellipsis_marker_is_treated_as_unit(self):
        # "……" 作为一个边界单元被识别，丢弃其前缀后保留其后的文本。
        result = trim_text_prefix_to_limit("abc……defghij", 8)
        assert result == "defghij"
        assert len(result) <= 8

    def test_falls_back_to_whitespace_boundary(self):
        # 没有可用标点时，退化到在空白边界丢弃前缀。
        result = trim_text_prefix_to_limit("aaaa bbbb cccc", 9)
        assert result == "bbbb cccc"
        assert len(result) <= 9

    def test_falls_back_to_last_n_chars_when_no_boundary(self):
        # 既无标点也无空白时，兜底保留末尾 N 个字符（仍是丢前半段）。
        result = trim_text_prefix_to_limit("abcdefghij", 4)
        assert result == "ghij"
        assert len(result) <= 4

    def test_result_never_exceeds_limit_for_long_cjk_run(self):
        result = trim_text_prefix_to_limit("话" * 500, 144)
        assert len(result) == 144
        assert set(result) == {"话"}
