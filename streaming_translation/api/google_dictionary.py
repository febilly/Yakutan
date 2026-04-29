import asyncio
import urllib.parse
import json
from typing import Optional

from .base import BaseTranslationAPI
from .._proxy import detect_system_proxy

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp not installed. Run: pip install aiohttp"
    )


class GoogleDictionaryAPI(BaseTranslationAPI):
    """Google Dictionary Extension API wrapper."""

    SUPPORTS_CONTEXT = False

    API_KEY = "AIzaSyA6EEtrDCfBkHV8uU2lgGY-N383ZgAOo7Y"
    API_ENDPOINT = "https://dictionaryextension-pa.googleapis.com/v1/dictionaryExtensionData"
    TIMEOUT = 2

    @staticmethod
    def _coerce_language_code(target_language: str) -> str:
        if not target_language:
            return "en"
        key = str(target_language).strip().lower().replace("_", "-")
        aliases = {
            "zh-hans": "zh-cn",
            "zh-hant": "zh-tw",
            "zh": "zh-cn",
            "zh-cn": "zh-cn",
            "zh-tw": "zh-tw",
            "zh-sg": "zh-cn",
            "zh-hk": "zh-tw",
            "zh-mo": "zh-tw",
        }
        return aliases.get(key, str(target_language).strip())

    def __init__(self, max_retries: int = 3, proxy_url: Optional[str] = None):
        self.max_retries = max_retries
        self.proxy_url = proxy_url or detect_system_proxy()
        self._session_timeout = aiohttp.ClientTimeout(total=self.TIMEOUT)
        self._session: Optional[aiohttp.ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        if self.proxy_url:
            self.proxy_url = self.proxy_url

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._session_timeout)
        return self._session

    async def _reset_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        return await self._get_session()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _translate_async(
        self, text: str, source_language: str, target_language: str
    ) -> str:
        api_language = self._coerce_language_code(target_language)
        for attempt in range(self.max_retries + 1):
            try:
                encoded = urllib.parse.quote(text)
                url = (
                    f"{self.API_ENDPOINT}?"
                    f"language={api_language}&"
                    f"key={self.API_KEY}&"
                    f"term={encoded}&"
                    f"strategy=2"
                )
                headers = {
                    "x-referer": "chrome-extension://mgijmajocgfcbeboacabfgobmjgjcoja"
                }
                session = await self._get_session()
                async with session.get(url, headers=headers, proxy=self.proxy_url) as resp:
                    if resp.status == 200:
                        data = json.loads(await resp.text())
                        if "translateResponse" in data:
                            return data["translateResponse"].get("translateText", "")
                        return "[ERROR] Unexpected API response format"
                    return f"[ERROR] HTTP {resp.status}: {await resp.text()}"
            except (
                aiohttp.ClientConnectionError,
                aiohttp.ClientSSLError,
                ConnectionError,
                BrokenPipeError,
            ) as e:
                if attempt < self.max_retries:
                    await self._reset_session()
                    continue
                return f"[ERROR] Connection failed after {self.max_retries + 1} attempts: {e}"
            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    await self._reset_session()
                    continue
                return f"[ERROR] Timeout after {self.max_retries + 1} attempts"
            except Exception as e:
                return f"[ERROR] {e}"
        return "[ERROR] Unknown error"

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
                "Google Dictionary API does not support native context."
            )
        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if in_running_loop:
            tmp_loop = asyncio.new_event_loop()
            try:
                result = tmp_loop.run_until_complete(
                    self._translate_async(text, source_language, target_language)
                )
                try:
                    tmp_loop.run_until_complete(self.close())
                except Exception:
                    pass
                return result
            finally:
                try:
                    tmp_loop.close()
                except Exception:
                    pass

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(
            self._translate_async(text, source_language, target_language)
        )

    def __del__(self):
        try:
            if self._loop and not self._loop.is_closed() and not self._loop.is_running():
                try:
                    self._loop.run_until_complete(self.close())
                except Exception:
                    pass
                try:
                    self._loop.close()
                except Exception:
                    pass
                self._loop = None
        except Exception:
            pass
