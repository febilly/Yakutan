"""
OpenAI-compatible client helper.
Shared API key rotation and client construction.
"""
import os
import threading
from typing import List

from proxy_detector import detect_system_proxy

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI 库未安装。请运行以下命令安装：\n"
        "pip install --upgrade openai"
    )


class OpenAICompatClientBase:
    """Shared OpenAI-compatible client logic with API key rotation."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

        self._key_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._api_keys = self._parse_api_keys(self._resolve_raw_api_keys())
        if not self._api_keys:
            raise ValueError(
                "OpenRouter API Key 未设置。请在网页控制面板的 'API Keys 配置' 中填写 OpenRouter API Key。"
            )
        self._key_index = 0
        self.api_key = self._api_keys[0]

        self.app_url = os.getenv("OPENROUTER_APP_URL", "")
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "")

        self._create_client()

    @staticmethod
    def _parse_api_keys(raw_keys: str) -> List[str]:
        if not raw_keys:
            return []
        return [key.strip() for key in raw_keys.split(',') if key.strip()]

    @staticmethod
    def _resolve_raw_api_keys() -> str:
        openai_keys = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_keys:
            return openai_keys
        return os.getenv("OPENROUTER_API_KEY", "").strip()

    def _get_next_api_key(self) -> str:
        if len(self._api_keys) <= 1:
            return self.api_key
        with self._key_lock:
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            return self._api_keys[self._key_index]

    def _maybe_rotate_key(self) -> None:
        next_key = self._get_next_api_key()
        if next_key != self.api_key:
            with self._client_lock:
                if next_key != self.api_key:
                    self.api_key = next_key
                    self._create_client()

    def _create_client(self) -> None:
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }

        default_headers = {}
        if self.app_url:
            default_headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            default_headers["X-Title"] = self.app_title
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        proxies = detect_system_proxy()
        if proxies:
            import httpx
            proxy_url = proxies.get('https') or proxies.get('http')
            if proxy_url:
                client_kwargs["http_client"] = httpx.Client(proxy=proxy_url)

        self.client = OpenAI(**client_kwargs)

    def _is_openrouter_base_url(self) -> bool:
        return 'openrouter.ai' in (self.base_url or '').lower()

    @staticmethod
    def clean_response(text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\r\n", "\n")
        while True:
            start_idx = cleaned.find("<think>")
            if start_idx == -1:
                break
            end_idx = cleaned.find("</think>", start_idx)
            if end_idx == -1:
                cleaned = cleaned[:start_idx]
                break
            cleaned = cleaned[:start_idx] + cleaned[end_idx + len("</think>"):]
        return cleaned.lstrip("\n").strip()
