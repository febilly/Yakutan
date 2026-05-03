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

    @staticmethod
    def _extract_vrcx_context(context: Optional[str]) -> str:
        if not context:
            return ""
        start_marker = "<VRCHAT_CONTEXT>"
        end_marker = "</VRCHAT_CONTEXT>"
        start = context.find(start_marker)
        end = context.find(end_marker, start + len(start_marker))
        if start < 0 or end <= start:
            return ""
        text = context[start + len(start_marker):end].strip()
        return text[:3000].rstrip()

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
                tm_list = [
                    {"source": p["source"], "target": p["target"]}
                    for p in context_pairs
                    if str(p.get("source") or "").strip()
                    and str(p.get("target") or "").strip()
                ]
                if tm_list:
                    translation_options["tm_list"] = tm_list

            domain_text = self.DOMAINS
            vrcx_context = self._extract_vrcx_context(context)
            if vrcx_context:
                domain_text = (
                    f"{domain_text}\n"
                    "Local VRChat context for names, world and references:\n"
                    f"{vrcx_context}"
                )

            if domain_text:
                translation_options["domains"] = domain_text

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
