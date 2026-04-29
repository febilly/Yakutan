import asyncio
from typing import Optional

from .base import BaseTranslationAPI
from .._proxy import detect_system_proxy

try:
    from googletrans import Translator as GoogleWebTranslatorAPI
except ImportError:
    raise ImportError(
        "googletrans not installed. Run: pip install streaming-translation[google]"
    )


class GoogleWebAPI(BaseTranslationAPI):
    """Google Web Translator API wrapper (via googletrans)."""

    SUPPORTS_CONTEXT = False

    def __init__(self, proxy_url: Optional[str] = None):
        resolved = proxy_url or detect_system_proxy()
        if resolved:
            self.google_translator = GoogleWebTranslatorAPI(proxy=resolved)
        else:
            self.google_translator = GoogleWebTranslatorAPI()

    async def _translate_async(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        result = await self.google_translator.translate(
            text, src=source_language, dest=target_language
        )
        return result.text

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        **kwargs,
    ) -> str:
        if context is not None:
            raise NotImplementedError(
                "Google Web Translator does not support native context. "
                "Use ContextAwareTranslator wrapper instead."
            )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self._translate_async(text, source_language, target_language)
        )
