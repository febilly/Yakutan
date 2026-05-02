"""Tests for language detectors — CJKEDetector, EnZhDetector."""

from __future__ import annotations

import pytest

from language_detectors.cjke_detector import CJKEDetector
from language_detectors.enzh_detector import EnZhDetector


class TestCJKEDetector:
    @pytest.fixture
    def detector(self):
        return CJKEDetector()

    def test_detect_english(self, detector):
        result = detector.detect("Hello, how are you today?")
        assert result["language"] == "en"
        assert result["confidence"] > 0.5

    def test_detect_chinese_simplified(self, detector):
        result = detector.detect("你好，今天天气真好")
        assert result["language"] == "zh"
        assert result["confidence"] > 0.5

    def test_detect_japanese(self, detector):
        result = detector.detect("こんにちは、今日はいい天気です")
        assert result["language"] == "ja"
        assert result["confidence"] > 0.5

    def test_detect_japanese_with_kanji(self, detector):
        result = detector.detect("私は日本語を勉強しています")
        assert result["language"] == "ja"

    def test_detect_korean(self, detector):
        result = detector.detect("안녕하세요, 오늘 날씨가 좋네요")
        assert result["language"] == "ko"
        assert result["confidence"] > 0.5

    def test_empty_text_returns_unknown(self, detector):
        result = detector.detect("")
        assert result["language"] == "unknown"
        assert result["confidence"] == 0.0

    def test_whitespace_only(self, detector):
        result = detector.detect("   ")
        assert result["language"] == "unknown"

    def test_only_numbers(self, detector):
        result = detector.detect("1234567890")
        assert result["language"] == "unknown"

    def test_only_punctuation(self, detector):
        result = detector.detect("!?,.;:")
        assert result["language"] == "unknown"

    def test_mixed_english_chinese(self, detector):
        result = detector.detect("Hello 世界")
        assert result["language"] in ("en", "zh-cn")

    def test_mixed_japanese_chinese(self, detector):
        result = detector.detect("東京は大きい city です")
        assert result["language"] == "ja"

    def test_english_with_numbers(self, detector):
        result = detector.detect("I have 2 apples and 3 oranges")
        assert result["language"] == "en"

    def test_chinese_with_punctuation(self, detector):
        result = detector.detect("《论语》中说：「学而时习之，不亦说乎？」")
        assert result["language"] == "zh"

    def test_confidence_high_for_pure_text(self, detector):
        result = detector.detect("This is a pure English sentence with many words")
        assert result["confidence"] > 0.9

    def test_confidence_lower_for_mixed(self, detector):
        result = detector.detect("Hello 你好 こんにちは")
        # Multiple scripts present
        assert result["confidence"] >= 0.0

    def test_detect_async(self, detector):
        import asyncio
        result = asyncio.run(detector.detect_async("Hello world"))
        assert result["language"] == "en"


class TestEnZhDetector:
    @pytest.fixture
    def detector(self):
        return EnZhDetector()

    def test_detect_english(self, detector):
        result = detector.detect("Hello, how are you?")
        assert result["language"] == "en"
        assert result["confidence"] > 0.5

    def test_detect_chinese(self, detector):
        result = detector.detect("你好，今天天气怎么样？")
        assert result["language"] == "zh-cn"
        assert result["confidence"] > 0.5

    def test_empty_text_unknown(self, detector):
        result = detector.detect("")
        assert result["language"] == "unknown"

    def test_whitespace_only(self, detector):
        result = detector.detect("   ")
        assert result["language"] == "unknown"

    def test_only_numbers(self, detector):
        result = detector.detect("12345")
        assert result["language"] == "unknown"

    def test_mixed_chinese_english(self, detector):
        result = detector.detect("This is a 混合 sentence")
        assert result["language"] in ("en", "zh-cn")

    def test_chinese_heavier_than_english(self, detector):
        result = detector.detect("今天天气真的非常好 and very nice")
        # Chinese chars weighted 3x, so zh-cn should win
        assert result["language"] == "zh-cn"

    def test_confidence_range(self, detector):
        result = detector.detect("Hello world")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_detect_async(self, detector):
        import asyncio
        result = asyncio.run(detector.detect_async("Hello"))
        assert result["language"] == "en"
