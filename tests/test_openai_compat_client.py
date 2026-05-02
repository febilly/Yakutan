"""Tests for openai_compat_client — LLM client with key rotation."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestOpenAICompatClientBase:
    def test_requires_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from openai_compat_client import OpenAICompatClientBase
            with pytest.raises(ValueError, match="LLM API Key"):
                OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")

    def test_uses_llm_api_key(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "llm-key"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")
            assert client.api_key == "llm-key"

    def test_falls_back_to_openai_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "oa-key"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")
            assert client.api_key == "oa-key"

    def test_falls_back_to_openrouter_api_key(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")
            assert client.api_key == "or-key"

    def test_llm_app_url_headers(self):
        with patch.dict(os.environ, {
            "LLM_API_KEY": "key",
            "LLM_APP_URL": "https://myapp.com",
            "LLM_APP_TITLE": "MyApp",
        }, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")
            mock_openai.assert_called_once()
            _, kwargs = mock_openai.call_args
            headers = kwargs.get("default_headers", {})
            assert headers.get("HTTP-Referer") == "https://myapp.com"
            assert headers.get("X-Title") == "MyApp"

    def test_no_headers_when_not_configured(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "key"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")
            _, kwargs = mock_openai.call_args
            assert "default_headers" not in kwargs or kwargs["default_headers"] == {}

    def test_key_parsing_single(self):
        from openai_compat_client import OpenAICompatClientBase
        keys = OpenAICompatClientBase._parse_api_keys("key1")
        assert keys == ["key1"]

    def test_key_parsing_multiple(self):
        from openai_compat_client import OpenAICompatClientBase
        keys = OpenAICompatClientBase._parse_api_keys("key1, key2, key3")
        assert keys == ["key1", "key2", "key3"]

    def test_key_parsing_empty(self):
        from openai_compat_client import OpenAICompatClientBase
        assert OpenAICompatClientBase._parse_api_keys("") == []
        assert OpenAICompatClientBase._parse_api_keys(None) == []

    def test_key_rotation(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "key1,key2,key3"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")
            assert client.api_key == "key1"
            client._maybe_rotate_key()
            assert client.api_key == "key2"
            client._maybe_rotate_key()
            assert client.api_key == "key3"
            client._maybe_rotate_key()
            assert client.api_key == "key1"

    def test_single_key_does_not_rotate(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "onlykey"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(base_url="https://test.ai/v1", model="test")
            client._maybe_rotate_key()
            assert client.api_key == "onlykey"

    def test_clean_response_strips_think_tags(self):
        from openai_compat_client import OpenAICompatClientBase
        result = OpenAICompatClientBase.clean_response(
            "<think>Let me translate</think>Hello world"
        )
        assert "think" not in result
        assert result == "Hello world"

    def test_clean_response_no_think(self):
        from openai_compat_client import OpenAICompatClientBase
        result = OpenAICompatClientBase.clean_response("Hello world")
        assert result == "Hello world"

    def test_clean_response_empty(self):
        from openai_compat_client import OpenAICompatClientBase
        assert OpenAICompatClientBase.clean_response("") == ""

    def test_clean_response_unclosed_think(self):
        from openai_compat_client import OpenAICompatClientBase
        result = OpenAICompatClientBase.clean_response("<think>partial")
        assert result == ""

    def test_is_openrouter_url(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "key"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(
                base_url="https://openrouter.ai/api/v1", model="test"
            )
            assert client._is_openrouter_base_url() is True

    def test_is_not_openrouter_url(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "key"}, clear=True), \
             patch("openai_compat_client.OpenAI") as mock_openai:
            from openai_compat_client import OpenAICompatClientBase
            client = OpenAICompatClientBase(
                base_url="https://api.openai.com/v1", model="test"
            )
            assert client._is_openrouter_base_url() is False
