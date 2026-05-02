"""Tests for hot_words_manager — file loading and processing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_vocabulary_service():
    """Mock dashscope VocabularyService to prevent real API calls."""
    with patch("hot_words_manager.VocabularyService") as mock:
        mock.return_value = MagicMock()
        yield mock


class TestHotWordsManagerInit:
    def test_import_and_create(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager()
        assert mgr.hot_words == []

    def test_custom_api_key(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager(api_key="test-key")
        assert mgr.api_key == "test-key"


class TestHotWordsFileLoading:
    def test_load_single_hot_words_file(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("hello\nworld\n")
            f.write("  \n")  # empty line
            f.write("foo bar\n")
            temp_path = f.name

        try:
            hot_words = mgr.load_hot_words_from_file(temp_path, "test")
            words = [w["text"] for w in hot_words]
            assert words == ["hello", "world", "foo bar"]
        finally:
            os.unlink(temp_path)

    def test_load_file_with_whitespace_lines(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("  apple  \n")
            f.write("banana\n")
            f.write("  \n")
            f.write("\n")
            temp_path = f.name

        try:
            hot_words = mgr.load_hot_words_from_file(temp_path, "test")
            words = [w["text"] for w in hot_words]
            assert words == ["apple", "banana"]
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager()
        hot_words = mgr.load_hot_words_from_file("/nonexistent/path/file.txt", "test")
        assert hot_words == []

    def test_load_file_with_comments(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("hello\n")
            f.write("# this is a comment\n")
            f.write("world\n")
            f.write("// another comment\n")
            temp_path = f.name

        try:
            hot_words = mgr.load_hot_words_from_file(temp_path, "test")
            words = [w["text"] for w in hot_words]
            assert "hello" in words
            assert "world" in words
        finally:
            os.unlink(temp_path)


class TestHotWordsManagement:
    def test_get_hot_words_returns_list(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager()
        mgr.hot_words = [{"text": "hello", "weight": 4}]
        result = mgr.get_hot_words()
        assert result == mgr.hot_words

    def test_language_code_mapping(self):
        from hot_words_manager import HotWordsManager
        assert HotWordsManager.LANG_CODE_MAP["zh-cn"] == "zh"
        assert HotWordsManager.LANG_CODE_MAP["en"] == "en"
        assert HotWordsManager.LANG_CODE_MAP["ja"] == "ja"
        assert HotWordsManager.LANG_CODE_MAP["ko"] == "ko"
        assert HotWordsManager.LANG_CODE_MAP["zh-tw"] == "zh"

    def test_enabled_languages_default(self):
        from hot_words_manager import HotWordsManager
        assert "zh-cn" in HotWordsManager.ENABLED_LANGUAGES
        assert "en" in HotWordsManager.ENABLED_LANGUAGES
        assert "ja" in HotWordsManager.ENABLED_LANGUAGES


class TestHotWordsVocabularyCreation:
    def test_create_vocabulary(self):
        from hot_words_manager import HotWordsManager

        mock_service = MagicMock()
        mock_service.create_vocabulary.return_value = "vocab-123"

        mgr = HotWordsManager()
        mgr.vocabulary_service = mock_service
        mgr.hot_words = [{"text": "hello", "weight": 4}]

        with patch.object(mgr, "_cleanup_old_vocabularies"):
            vocab_id = mgr.create_vocabulary()
            assert vocab_id == "vocab-123"
            mock_service.create_vocabulary.assert_called_once()

    def test_vocabulary_name_format(self):
        from hot_words_manager import HotWordsManager
        # Verify the vocabulary prefix constraints
        prefix = HotWordsManager.VOCABULARY_PREFIX
        assert prefix.isalnum()
        assert prefix.islower()
        assert len(prefix) < 10


class TestAcrossLanguages:
    def test_hot_words_across_languages(self):
        from hot_words_manager import HotWordsManager
        mgr = HotWordsManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock hot words files
            zh_file = os.path.join(tmpdir, "zh-cn.txt")
            en_file = os.path.join(tmpdir, "en.txt")

            with open(zh_file, "w", encoding="utf-8") as f:
                f.write("中文热词\n另一个词\n")
            with open(en_file, "w", encoding="utf-8") as f:
                f.write("english word\nanother one\n")

            # Test loading from specific file
            zh_words = mgr.load_hot_words_from_file(zh_file, "zh-cn")
            en_words = mgr.load_hot_words_from_file(en_file, "en")

            assert len(zh_words) == 2
            assert len(en_words) == 2
            texts_zh = [w["text"] for w in zh_words]
            texts_en = [w["text"] for w in en_words]
            assert "中文热词" in texts_zh
            assert "english word" in texts_en
