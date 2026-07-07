"""
Minimal proxy-detection helper (extracted from ``proxy_detector.py`` so the
translation library does **not** depend on the host app's utility module).
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional

_MANAGED_ENV_PREFIX = "YAKUTAN_MANAGED_PROXY_"
_PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "WS_PROXY",
    "ws_proxy",
    "WSS_PROXY",
    "wss_proxy",
    "NO_PROXY",
    "no_proxy",
)


def _marker_name(env_name: str) -> str:
    return f"{_MANAGED_ENV_PREFIX}{env_name}"


def _clear_managed_proxy_env() -> None:
    """Remove proxy env values previously injected by the host app."""
    for env_name in _PROXY_ENV_NAMES:
        marker = _marker_name(env_name)
        previous_value = os.environ.get(marker)
        if previous_value is None:
            continue
        current_value = os.environ.get(env_name)
        if current_value == previous_value:
            os.environ.pop(env_name, None)
        os.environ.pop(marker, None)


def _set_managed_proxy_env(proxy_url: str) -> None:
    for env_name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        if os.environ.get(env_name):
            continue
        os.environ[env_name] = proxy_url
        os.environ[_marker_name(env_name)] = proxy_url


def detect_system_proxy() -> Optional[str]:
    """Detect system HTTP/HTTPS proxy from environment variables.

    Returns a single proxy URL (preferring HTTPS) or ``None``.
    This is a simplified version of the host app's ``proxy_detector``.
    """
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if https_proxy:
        return https_proxy

    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if http_proxy:
        return http_proxy

    try:
        handler = urllib.request.ProxyHandler()
        proxies = handler.proxies
        if proxies:
            return proxies.get("https") or proxies.get("http")
    except Exception:
        pass

    return None


def refresh_system_proxy() -> Optional[str]:
    """Refresh system proxy detection and return the current proxy URL."""
    _clear_managed_proxy_env()
    proxy_url = detect_system_proxy()
    if proxy_url:
        _set_managed_proxy_env(proxy_url)
    return proxy_url
