import os
from typing import Optional, List, Dict

from .base import BaseTranslationAPI
from .._config import TranslationConfig
from .._proxy import detect_system_proxy

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai library not installed. Run: pip install openai"
    )


class QwenMTAPI(BaseTranslationAPI):
    """Qwen-MT translation API via DashScope (OpenAI-compatible)."""

    SUPPORTS_CONTEXT = True

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    BASE_URL_INTERNATIONAL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    LANGUAGE_MAP = {
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh-tw",
        "en-us": "en",
        "en-gb": "en",
        "en-au": "en",
        "pt-br": "pt",
        "pt-pt": "pt",
    }

    DOMAINS = (
        "The text is casual conversation from VRChat, "
        "a social virtual reality platform. "
        "Keep translations natural, friendly and colloquial."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-mt-flash",
        use_international: bool = False,
        proxy_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. Pass it or set DASHSCOPE_API_KEY."
            )

        self.model = model
        base_url = self.BASE_URL_INTERNATIONAL if use_international else self.BASE_URL

        resolved_proxy = proxy_url or detect_system_proxy()
        client_kwargs = {"api_key": self.api_key, "base_url": base_url}
        if resolved_proxy:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxy=resolved_proxy)

        self.client = OpenAI(**client_kwargs)

    def _get_language_code(self, lang_code: str) -> str:
        code = lang_code.lower()
        return self.LANGUAGE_MAP.get(code, code)

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        context_pairs: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        try:
            source_lang = self._get_language_code(source_language)
            target_lang = self._get_language_code(target_language)

            translation_options = {
                "source_lang": source_lang,
                "target_lang": target_lang,
            }

            if context_pairs:
                translation_options["tm_list"] = [
                    {"source": p["source"], "target": p["target"]}
                    for p in context_pairs
                ]

            if self.DOMAINS:
                translation_options["domains"] = self.DOMAINS

            messages = [{"role": "user", "content": text}]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"translation_options": translation_options},
            )

            result = completion.choices[0].message.content
            return result.strip() if result else ""

        except Exception as e:
            return f"[ERROR] {e}"
