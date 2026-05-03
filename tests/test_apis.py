"""Tests for all translation API backends and merge_with_draft."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from streaming_translation.api.base import BaseTranslationAPI
from streaming_translation.api.openrouter import merge_with_draft


# ── Abstract base class ───────────────────────────────────────────────

class TestBaseTranslationAPI:
    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseTranslationAPI()  # type: ignore

    def test_default_supports_context(self):
        assert BaseTranslationAPI.SUPPORTS_CONTEXT is False

    def test_concrete_subclass(self):
        class Concrete(BaseTranslationAPI):
            def translate(self, **kw):
                return "hello"
        instance = Concrete()
        assert instance.translate(text="hi") == "hello"


# ── DeepLAPI ──────────────────────────────────────────────────────────

class TestDeepLAPI:
    def test_missing_key_raises(self):
        with patch.dict("os.environ", clear=True):
            with pytest.raises(ValueError, match="DeepL API key"):
                from streaming_translation.api.deepl import DeepLAPI
                DeepLAPI(proxy_url=None)

    def test_construction_success(self):
        mock_client = MagicMock()
        mock_client.translate_text.return_value.text = "Hallo"
        with patch("streaming_translation.api.deepl.deepl") as mock_deepl:
            mock_deepl.DeepLClient.return_value = mock_client
            from streaming_translation.api.deepl import DeepLAPI
            api = DeepLAPI(api_key="test-key", proxy_url=None)
            assert api.formality == "default"
            result = api.translate("Hello", source_language="auto", target_language="de")
            assert result == "Hallo"

    def test_language_mapping(self):
        mock_client = MagicMock()
        with patch("streaming_translation.api.deepl.deepl") as mock_deepl:
            mock_deepl.DeepLClient.return_value = mock_client
            mock_deepl.AuthorizationException = type("AuthExc", (Exception,), {})
            mock_deepl.QuotaExceededException = type("QuotaExc", (Exception,), {})
            mock_deepl.DeepLException = type("DeepLExc", (Exception,), {})
            from streaming_translation.api.deepl import DeepLAPI
            api = DeepLAPI(api_key="test-key", proxy_url=None)
            # Reset calls to ignore warm-up
            mock_client.reset_mock()
            mock_client.translate_text.return_value.text = "你好"
            api.translate("Hello", target_language="zh-cn")
            _, kwargs = mock_client.translate_text.call_args
            assert kwargs["target_lang"] == "ZH-HANS"

    def test_authorization_error(self):
        mock_client = MagicMock()
        with patch("streaming_translation.api.deepl.deepl") as mock_deepl:
            mock_deepl.DeepLClient.return_value = mock_client
            mock_deepl.AuthorizationException = type("AuthExc", (Exception,), {})
            mock_deepl.QuotaExceededException = type("QuotaExc", (Exception,), {})
            mock_deepl.DeepLException = type("DeepLExc", (Exception,), {})
            from streaming_translation.api.deepl import DeepLAPI
            api = DeepLAPI(api_key="test-key", proxy_url=None)
            mock_client.translate_text.side_effect = mock_deepl.AuthorizationException()
            result = api.translate("Hello")
            assert "[ERROR]" in result and "auth" in result.lower()

    def test_quota_error(self):
        mock_client = MagicMock()
        with patch("streaming_translation.api.deepl.deepl") as mock_deepl:
            mock_deepl.DeepLClient.return_value = mock_client
            mock_deepl.AuthorizationException = type("AuthExc", (Exception,), {})
            mock_deepl.QuotaExceededException = type("QuotaExc", (Exception,), {})
            mock_deepl.DeepLException = type("DeepLExc", (Exception,), {})
            from streaming_translation.api.deepl import DeepLAPI
            api = DeepLAPI(api_key="test-key", proxy_url=None)
            mock_client.translate_text.side_effect = mock_deepl.QuotaExceededException()
            result = api.translate("Hello")
            assert "[ERROR]" in result and "quota" in result.lower()

    def test_context_passed_correctly(self):
        mock_client = MagicMock()
        with patch("streaming_translation.api.deepl.deepl") as mock_deepl:
            mock_deepl.DeepLClient.return_value = mock_client
            mock_deepl.AuthorizationException = type("AuthExc", (Exception,), {})
            mock_deepl.QuotaExceededException = type("QuotaExc", (Exception,), {})
            mock_deepl.DeepLException = type("DeepLExc", (Exception,), {})
            from streaming_translation.api.deepl import DeepLAPI
            api = DeepLAPI(api_key="test-key", proxy_url=None)
            mock_client.reset_mock()
            mock_client.translate_text.return_value.text = "Hallo"
            api.translate("Hello", context="Previous conversation. ", target_language="de")
            _, kwargs = mock_client.translate_text.call_args
            assert kwargs.get("context") == "Previous conversation. "

    def test_context_pairs_used(self):
        mock_client = MagicMock()
        with patch("streaming_translation.api.deepl.deepl") as mock_deepl:
            mock_deepl.DeepLClient.return_value = mock_client
            mock_deepl.AuthorizationException = type("AuthExc", (Exception,), {})
            mock_deepl.QuotaExceededException = type("QuotaExc", (Exception,), {})
            mock_deepl.DeepLException = type("DeepLExc", (Exception,), {})
            from streaming_translation.api.deepl import DeepLAPI
            api = DeepLAPI(api_key="test-key", proxy_url=None)
            mock_client.reset_mock()
            mock_client.translate_text.return_value.text = "Hola"
            api.translate("Hello", context_pairs=[{"source": "Hi", "target": "Hola"}],
                         target_language="es")
            _, kwargs = mock_client.translate_text.call_args
            assert "context" in kwargs
            assert "Hi" in kwargs["context"]

    def test_vrcx_context_kept_with_context_pairs(self):
        mock_client = MagicMock()
        with patch("streaming_translation.api.deepl.deepl") as mock_deepl:
            mock_deepl.DeepLClient.return_value = mock_client
            mock_deepl.AuthorizationException = type("AuthExc", (Exception,), {})
            mock_deepl.QuotaExceededException = type("QuotaExc", (Exception,), {})
            mock_deepl.DeepLException = type("DeepLExc", (Exception,), {})
            from streaming_translation.api.deepl import DeepLAPI
            api = DeepLAPI(api_key="test-key", proxy_url=None)
            mock_client.reset_mock()
            mock_client.translate_text.return_value.text = "Hola"
            api.translate(
                "Hello",
                context="Base\n<VRCHAT_CONTEXT>\nWorld: Test World\n</VRCHAT_CONTEXT>",
                context_pairs=[{"source": "Hi", "target": "Hola"}],
                target_language="es",
            )
            _, kwargs = mock_client.translate_text.call_args
            assert "World: Test World" in kwargs["context"]
            assert "Hi" in kwargs["context"]


# ── GoogleWebAPI ──────────────────────────────────────────────────────

class TestGoogleWebAPI:
    def test_context_raises(self):
        with patch("streaming_translation.api.google_web.GoogleWebTranslatorAPI"):
            from streaming_translation.api.google_web import GoogleWebAPI
            api = GoogleWebAPI(proxy_url=None)
            with pytest.raises(NotImplementedError, match="does not support native context"):
                api.translate("Hello", context="context")

    def test_translate_success(self):
        mock_translate = MagicMock()
        mock_translate.text = "Hola"
        # await needs an awaitable — wrap the mock in an async function
        async def async_translate(*a, **kw):
            return mock_translate
        mock_gt = MagicMock()
        mock_gt.translate = async_translate
        with patch("streaming_translation.api.google_web.GoogleWebTranslatorAPI",
                   return_value=mock_gt):
            from streaming_translation.api.google_web import GoogleWebAPI
            api = GoogleWebAPI(proxy_url=None)
            result = api.translate("Hello", source_language="en", target_language="es")
            assert result == "Hola"


# ── GoogleDictionaryAPI ───────────────────────────────────────────────

class TestGoogleDictionaryAPI:
    def test_context_raises(self):
        from streaming_translation.api.google_dictionary import GoogleDictionaryAPI
        api = GoogleDictionaryAPI(max_retries=1, proxy_url=None)
        with pytest.raises(NotImplementedError, match="does not support native context"):
            api.translate("Hello", context="ctx")

    @patch("streaming_translation.api.google_dictionary.aiohttp.ClientSession")
    def test_translate_success(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__aenter__.return_value = mock_resp

        async def mock_text():
            return '{"translateResponse": {"translateText": "Hola"}}'
        mock_resp.text = mock_text

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_resp
        mock_session_instance.closed = False
        mock_session.return_value = mock_session_instance

        from streaming_translation.api.google_dictionary import GoogleDictionaryAPI
        api = GoogleDictionaryAPI(max_retries=0, proxy_url=None)
        result = api.translate("Hello", target_language="es")
        assert result == "Hola"

    def test_coerce_language_code(self):
        from streaming_translation.api.google_dictionary import GoogleDictionaryAPI
        assert GoogleDictionaryAPI._coerce_language_code("") == "en"
        assert GoogleDictionaryAPI._coerce_language_code("zh-cn") == "zh-cn"
        assert GoogleDictionaryAPI._coerce_language_code("zh-hans") == "zh-cn"
        assert GoogleDictionaryAPI._coerce_language_code("zh-tw") == "zh-tw"
        assert GoogleDictionaryAPI._coerce_language_code("zh-hant") == "zh-tw"
        assert GoogleDictionaryAPI._coerce_language_code(None) == "en"
        assert GoogleDictionaryAPI._coerce_language_code("fr") == "fr"


# ── QwenMTAPI ─────────────────────────────────────────────────────────

class TestQwenMTAPI:
    def test_missing_key_raises(self):
        with patch.dict("os.environ", clear=True):
            with pytest.raises(ValueError, match="DashScope API key"):
                from streaming_translation.api.qwen_mt import QwenMTAPI
                QwenMTAPI(proxy_url=None)

    @patch("streaming_translation.api.qwen_mt.OpenAI")
    def test_translate_success(self, mock_openai):
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hola"
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        from streaming_translation.api.qwen_mt import QwenMTAPI
        api = QwenMTAPI(api_key="test-key", proxy_url=None)
        result = api.translate("Hello", target_language="es")
        assert result == "Hola"

    @patch("streaming_translation.api.qwen_mt.OpenAI")
    def test_language_code_mapping(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from streaming_translation.api.qwen_mt import QwenMTAPI
        api = QwenMTAPI(api_key="test-key", proxy_url=None)
        # _get_language_code is internal but let's test via translate args:
        api.client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "你好"
        api.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        api.translate("Hello", target_language="zh-cn")
        _, kwargs = api.client.chat.completions.create.call_args
        opts = kwargs["extra_body"]["translation_options"]
        assert opts["target_lang"] == "zh"  # zh-cn -> zh

    @patch("streaming_translation.api.qwen_mt.OpenAI")
    def test_tm_list_included(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from streaming_translation.api.qwen_mt import QwenMTAPI
        api = QwenMTAPI(api_key="test-key", proxy_url=None)
        api.client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hola"
        api.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        api.translate("Hello", context_pairs=[{"source": "Hi", "target": "Hola"}],
                     target_language="es")
        _, kwargs = api.client.chat.completions.create.call_args
        opts = kwargs["extra_body"]["translation_options"]
        assert "tm_list" in opts
        assert len(opts["tm_list"]) == 1
        assert opts["tm_list"][0]["source"] == "Hi"

    @patch("streaming_translation.api.qwen_mt.OpenAI")
    def test_vrcx_context_added_to_domains(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from streaming_translation.api.qwen_mt import QwenMTAPI
        api = QwenMTAPI(api_key="test-key", proxy_url=None)
        api.client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hola"
        api.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        api.translate(
            "Hello",
            context="Base\n<VRCHAT_CONTEXT>\nWorld: Test World\n</VRCHAT_CONTEXT>",
            target_language="es",
        )
        _, kwargs = api.client.chat.completions.create.call_args
        opts = kwargs["extra_body"]["translation_options"]
        assert "World: Test World" in opts["domains"]


# ── OpenRouterAPI ─────────────────────────────────────────────────────

class TestOpenRouterAPI:
    def test_requires_key(self):
        with patch.dict("os.environ", {"LLM_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="LLM API Key"):
                from streaming_translation.api.openrouter import OpenRouterAPI
                OpenRouterAPI(base_url="https://test.ai/v1", model="test",
                            api_key=None, proxy_url=None)

    @patch("streaming_translation.api.openrouter.OpenAI")
    def test_construction_and_translate(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from streaming_translation.api.openrouter import OpenRouterAPI
        api = OpenRouterAPI(base_url="https://test.ai/v1", model="test-model",
                          api_key="test-key", proxy_url=None)
        assert api.model == "test-model"
        assert api.temperature == 0.2

    @patch("streaming_translation.api.openrouter.OpenAI")
    def test_custom_params(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from streaming_translation.api.openrouter import OpenRouterAPI
        api = OpenRouterAPI(base_url="https://test.ai/v1", model="m",
                          api_key="k", temperature=0.8, timeout=15,
                          formality="high", style="standard",
                          parallel_fastest_mode="all", proxy_url=None)
        assert api.temperature == 0.8
        assert api.timeout == 15
        assert api.formality == "high"
        assert api.style == "standard"
        assert api.parallel_fastest_mode == "all"

    @patch("streaming_translation.api.openrouter.OpenAI")
    def test_standard_translate(self, mock_openai):
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Bonjour"
        mock_completion.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        from streaming_translation.api.openrouter import OpenRouterAPI
        api = OpenRouterAPI(base_url="https://test.ai/v1", model="m",
                          api_key="k", proxy_url=None)
        result = api.translate("Hello", source_language="en", target_language="fr")
        assert result == "Bonjour"

    @patch("streaming_translation.api.openrouter.OpenAI")
    def test_error_response(self, mock_openai):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        mock_openai.return_value = mock_client

        from streaming_translation.api.openrouter import OpenRouterAPI
        api = OpenRouterAPI(base_url="https://test.ai/v1", model="m",
                          api_key="k", proxy_url=None)
        result = api.translate("Hello")
        assert result.startswith("[ERROR]")

    def test_describe_language(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        assert "auto-detected" in OpenRouterAPI._describe_language("auto")
        assert "JAPANESE" in OpenRouterAPI._describe_language("ja")
        assert "ENGLISH" in OpenRouterAPI._describe_language("en")
        assert OpenRouterAPI._describe_language("xx") == "XX (xx)"

    def test_is_cjk(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        assert OpenRouterAPI._is_cjk("zh") is True
        assert OpenRouterAPI._is_cjk("ja") is True
        assert OpenRouterAPI._is_cjk("ko") is True
        assert OpenRouterAPI._is_cjk("en") is False
        assert OpenRouterAPI._is_cjk("") is False
        assert OpenRouterAPI._is_cjk(None) is False

    def test_clean_response_strips_think_tags(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        result = OpenRouterAPI._clean_response(
            "<think>Let me translate this</think>Hola mundo"
        )
        assert "think" not in result
        assert result == "Hola mundo"

    def test_clean_response_unclosed_think(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        result = OpenRouterAPI._clean_response("<think>partial Hola mundo")
        assert result == ""

    def test_strip_trailing_partial_ellipsis(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        assert OpenRouterAPI._strip_trailing_partial_ellipsis("Hello...") == "Hello"
        assert OpenRouterAPI._strip_trailing_partial_ellipsis("Hello…") == "Hello"
        assert OpenRouterAPI._strip_trailing_partial_ellipsis("Hello") == "Hello"
        assert OpenRouterAPI._strip_trailing_partial_ellipsis("") == ""
        assert OpenRouterAPI._strip_trailing_partial_ellipsis("[ERROR] fail") == "[ERROR] fail"

    def test_build_context_block(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        # With context_pairs
        block = OpenRouterAPI._build_context_block(
            None, [{"source": "Hi", "target": "Hola"}])
        assert block is not None
        assert "Previous completed translations" in block
        assert "Hi" in block
        assert "Hola" in block
        # With context string
        block2 = OpenRouterAPI._build_context_block("Previous chat.", None)
        assert block2 is not None
        assert "Additional context notes" in block2
        assert "Previous chat." in block2
        # VRCX context is preserved even when translation-memory pairs are present
        block3 = OpenRouterAPI._build_context_block(
            "Base\n"
            "VRChat/VRCX local context for disambiguating world names.\n"
            "<VRCHAT_CONTEXT>\nWorld: Test World\n</VRCHAT_CONTEXT>",
            [{"source": "Hi", "target": "Hola"}, {"source": "Current", "target": ""}],
        )
        assert block3 is not None
        assert "World: Test World" in block3
        assert "Hi" in block3
        assert "Current" not in block3
        assert "local context for disambiguating" not in block3
        assert "Additional context notes:\nBase" in block3
        assert OpenRouterAPI._build_context_block(
            None, [{"source": "Current", "target": ""}]
        ) is None
        # Neither
        assert OpenRouterAPI._build_context_block(None, None) is None
        assert OpenRouterAPI._build_context_block("", None) is None

    def test_content_completeness_short_translation(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        needs, reason = OpenRouterAPI._check_content_completeness(
            "A very long English text here", "Hi",
            previous_translation="Hi",
            target_language="ja",  # cross CJK boundary -> 0.15 threshold
        )
        assert needs is True
        assert "ratio" in reason

    def test_content_completeness_ok(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        needs, reason = OpenRouterAPI._check_content_completeness(
            "Hello", "Bonjour le monde",
            previous_translation="Bonjour",
            target_language="fr",
        )
        assert needs is False

    def test_formality_style_guides(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        generic = OpenRouterAPI._get_formality_guide("en")
        assert isinstance(generic, str)
        jp = OpenRouterAPI._get_formality_guide("ja")
        assert isinstance(jp, str)
        ko = OpenRouterAPI._get_formality_guide("ko")
        assert isinstance(ko, str)

    def test_get_style_guide(self):
        from streaming_translation.api.openrouter import OpenRouterAPI
        guide = OpenRouterAPI._get_style_guide("en")
        assert isinstance(guide, str)

    @patch("streaming_translation.api.openrouter.OpenAI")
    def test_system_prompt_uses_configured_formality_and_style(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from streaming_translation.api.openrouter import OpenRouterAPI
        api = OpenRouterAPI(
            base_url="https://test.ai/v1",
            model="m",
            api_key="k",
            formality="high",
            style="standard",
            proxy_url=None,
        )
        prompt = api._build_system_prompt("JAPANESE (ja)", "ja")
        assert "clearly polite and refined teineigo" in prompt
        assert "natural, stable, and neutral" in prompt


# ── OpenRouterStreamingAPI ────────────────────────────────────────────

class TestOpenRouterStreamingAPI:
    @patch("streaming_translation.api.openrouter.OpenAI")
    def test_streaming_mode_flag(self, mock_openai):
        mock_openai.return_value = MagicMock()
        from streaming_translation.api.openrouter import OpenRouterStreamingAPI
        api = OpenRouterStreamingAPI(base_url="https://test.ai/v1", model="m",
                                   api_key="k", proxy_url=None)
        assert api.streaming_mode is True


# ── merge_with_draft ──────────────────────────────────────────────────

class TestMergeWithDraft:
    def test_fresh_starts_with_draft(self):
        assert merge_with_draft("hello world", "hello") == "hello world"

    def test_draft_starts_with_fresh(self):
        # draft longer than fresh: fresh is kept (would not happen in real rescue path)
        assert merge_with_draft("hello", "hello world") == "hello"

    def test_common_prefix_merge(self):
        result = merge_with_draft("hello universe", "hello world")
        assert "universe" in result
        assert result.startswith("hello")

    def test_different_no_common_prefix(self):
        result = merge_with_draft("completely different", "short")
        # Falls back to fresh
        assert result == "completely different"

    def test_empty_draft(self):
        assert merge_with_draft("hello world", "") == "hello world"

    def test_empty_fresh(self):
        # When fresh is empty there's nothing to merge — returns empty
        assert merge_with_draft("", "draft") == ""

    def test_both_empty(self):
        assert merge_with_draft("", "") == ""

    def test_identical(self):
        assert merge_with_draft("same text", "same text") == "same text"

    def test_unicode_merge(self):
        result = merge_with_draft("你好世界", "你好")
        assert result.startswith("你好")
        assert "世界" in result


# ── Proxy detection helper ────────────────────────────────────────────

class TestProxyDetection:
    def test_detect_system_proxy_no_proxy(self):
        import os
        from streaming_translation._proxy import detect_system_proxy
        with patch.dict(os.environ, {}, clear=True):
            result = detect_system_proxy()
            # May still detect from urllib; just check it doesn't crash
            assert result is None or isinstance(result, str)

    def test_detect_system_proxy_env_set(self):
        import os
        from streaming_translation._proxy import detect_system_proxy
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy:8080"}, clear=True):
            result = detect_system_proxy()
            assert result == "http://proxy:8080"
