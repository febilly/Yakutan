"""
Minimal proxy-detection helper (extracted from ``proxy_detector.py`` so the
translation library does **not** depend on the host app's utility module).
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional


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
