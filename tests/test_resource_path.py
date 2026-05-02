"""Tests for resource_path."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from resource_path import (
    ensure_dir,
    get_base_path,
    get_hot_words_path,
    get_hot_words_private_path,
    get_resource_path,
    get_ui_static_path,
    get_ui_template_path,
    get_user_data_path,
)


def _normalized_path_endswith(path: str, suffix: str) -> bool:
    return os.path.normpath(path).endswith(os.path.normpath(suffix))


class TestGetResourcePath:
    def test_relative_path_returns_absolute(self):
        result = get_resource_path("hot_words/zh-cn.txt")
        assert os.path.isabs(result)
        assert _normalized_path_endswith(result, "hot_words/zh-cn.txt")

    def test_frozen_uses_meipass(self):
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "_MEIPASS", "/fake/meipass", create=True):
                result = get_resource_path("ui/templates/index.html")
                assert result.startswith("/fake/meipass")
                assert _normalized_path_endswith(result, "ui/templates/index.html")

    def test_not_frozen_uses_project_root(self):
        with patch.object(sys, "frozen", False, create=True):
            result = get_resource_path("config.py")
            assert os.path.exists(result)


class TestGetBasePath:
    def test_returns_project_root(self):
        result = get_base_path()
        assert os.path.isdir(result)
        assert os.path.exists(os.path.join(result, "main.py"))

    def test_frozen_returns_meipass(self):
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "_MEIPASS", "/fake/meipass", create=True):
                assert get_base_path() == "/fake/meipass"


class TestEnsureDir:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "new_dir", "nested")
            ensure_dir(test_dir)
            assert os.path.isdir(test_dir)

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_dir(tmpdir)
            assert os.path.isdir(tmpdir)

    def test_relative_path_creates_under_project(self):
        with patch.object(sys, "frozen", False, create=True):
            test_dir = "_test_temp_dir"
            try:
                ensure_dir(test_dir)
                assert os.path.isdir(os.path.join(get_base_path(), test_dir))
            finally:
                import shutil
                shutil.rmtree(os.path.join(get_base_path(), test_dir), ignore_errors=True)


class TestGetUserDataPath:
    def test_no_relative_path(self):
        result = get_user_data_path()
        assert os.path.isabs(result)

    def test_with_relative_path(self):
        result = get_user_data_path("hot_words_private")
        assert result.endswith("hot_words_private")

    def test_frozen_uses_executable_dir(self):
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", "/usr/local/bin/yakutan", create=True):
                result = get_user_data_path("data")
                assert "yakutan" in result or "local" in result


class TestConveniencePaths:
    def test_get_hot_words_path(self):
        result = get_hot_words_path("en.txt")
        assert _normalized_path_endswith(result, "hot_words/en.txt")

    def test_get_hot_words_private_path(self):
        result = get_hot_words_private_path("en.txt")
        assert _normalized_path_endswith(result, "hot_words_private/en.txt")

    def test_get_ui_template_path(self):
        result = get_ui_template_path("index.html")
        assert "ui" in result and "templates" in result and result.endswith("index.html")

    def test_get_ui_static_path(self):
        result = get_ui_static_path("js/app.js")
        assert "ui" in result and "static" in result and _normalized_path_endswith(result, "js/app.js")
