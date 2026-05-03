import os
from typing import Optional, List, Dict

from .base import BaseTranslationAPI
from .._proxy import detect_system_proxy

try:
    import deepl
except ImportError:
    raise ImportError(
        "DeepL library not installed. Run: pip install streaming-translation[deepl]"
    )


class DeepLAPI(BaseTranslationAPI):
    """DeepL translation API wrapper."""

    SUPPORTS_CONTEXT = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        proxy_url: Optional[str] = None,
        formality: str = "default",
    ):
        auth_key = api_key or os.environ.get("DEEPL_API_KEY")
        if not auth_key:
            raise ValueError(
                "DeepL API key is required. Pass it explicitly or set DEEPL_API_KEY."
            )

        resolved_proxy = proxy_url or detect_system_proxy()
        proxy_config = None
        if resolved_proxy:
            proxy_config = {"https": resolved_proxy, "http": resolved_proxy}

        if proxy_config:
            self.client = deepl.DeepLClient(auth_key, proxy=proxy_config)
        else:
            self.client = deepl.DeepLClient(auth_key)

        self.formality = formality
        # warm-up
        self.translate("Hello", source_language="auto", target_language="en")

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
        return context[start + len(start_marker):end].strip()[:3000].rstrip()

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
            target_lang = target_language.upper()
            lang_map = {
                "zh": "ZH-HANS",
                "zh-cn": "ZH-HANS",
                "zh-hans": "ZH-HANS",
                "zh-sg": "ZH-HANS",
                "zh-tw": "ZH-HANT",
                "zh-hant": "ZH-HANT",
                "zh-hk": "ZH-HANT",
                "zh-mo": "ZH-HANT",
                "en": "EN-US",
                "pt": "PT-BR",
            }
            target_lang = lang_map.get(target_lang.lower(), target_lang)

            source_lang = None if source_language.lower() == "auto" else source_language.upper()

            final_context = None
            vrcx_context = self._extract_vrcx_context(context)
            if context_pairs:
                context_texts = [
                    p["source"]
                    for p in context_pairs
                    if str(p.get("source") or "").strip()
                    and str(p.get("target") or "").strip()
                ]
                final_context = " ".join(context_texts)
                if vrcx_context:
                    final_context = (
                        "VRChat context for names, world and references: "
                        f"{vrcx_context}\n{final_context}"
                    )
            elif context:
                final_context = context

            common_kwargs = dict(
                source_lang=source_lang,
                target_lang=target_lang,
                formality=self.formality,
                model_type="prefer_quality_optimized",
            )
            if final_context:
                result = self.client.translate_text(text, context=final_context, **common_kwargs)
            else:
                result = self.client.translate_text(text, **common_kwargs)

            return result.text

        except deepl.AuthorizationException:
            return "[ERROR] DeepL API auth failed"
        except deepl.QuotaExceededException:
            return "[ERROR] DeepL API quota exceeded"
        except deepl.DeepLException as e:
            return f"[ERROR] DeepL API error: {e}"
        except Exception as e:
            return f"[ERROR] {e}"
