"""Tests for proxy_detector."""

from __future__ import annotations

import os
from unittest.mock import patch

from proxy_detector import (
    _normalize_proxies,
    apply_system_proxy,
    detect_system_proxy,
    print_proxy_info,
)


class TestNormalizeProxies:
    def test_none_returns_none(self):
        assert _normalize_proxies(None) is None

    def test_empty_dict(self):
        assert _normalize_proxies({}) is None

    def test_basic_http(self):
        result = _normalize_proxies({"http": "http://proxy:8080"})
        assert result is not None
        assert result["http"] == "http://proxy:8080"
        assert result["https"] == "http://proxy:8080"

    def test_https_fallback(self):
        result = _normalize_proxies({"https": "https://proxy:8443"})
        assert result["https"] == "https://proxy:8443"
        assert result["http"] == "https://proxy:8443"

    def test_both_separate(self):
        result = _normalize_proxies({
            "http": "http://http-proxy:8080",
            "https": "https://https-proxy:8443",
        })
        assert result["http"] == "http://http-proxy:8080"
        assert result["https"] == "https://https-proxy:8443"

    def test_ws_wss_derived(self):
        result = _normalize_proxies({"http": "http://proxy:8080"})
        assert result["ws"] == "http://proxy:8080"
        assert result["wss"] == "http://proxy:8080"

    def test_all_proxy(self):
        result = _normalize_proxies({"all": "socks5://proxy:1080"})
        assert result["all"] == "socks5://proxy:1080"

    def test_no_proxy(self):
        result = _normalize_proxies({"http": "http://proxy:8080", "no": "localhost"})
        assert result["no"] == "localhost"


class TestDetectSystemProxy:
    def test_no_proxy(self):
        with patch.dict(os.environ, {}, clear=True):
            result = detect_system_proxy()
            assert result is None or isinstance(result, dict)

    def test_http_proxy_env(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}, clear=True):
            result = detect_system_proxy()
            assert result is not None
            assert result.get("http") == "http://proxy:8080"

    def test_https_proxy_env(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "https://proxy:8443"}, clear=True):
            result = detect_system_proxy()
            assert result is not None
            assert result.get("https") == "https://proxy:8443"

    def test_lowercase_env(self):
        with patch.dict(os.environ, {"http_proxy": "http://proxy:8080"}, clear=True):
            result = detect_system_proxy()
            assert result is not None
            assert result.get("http") == "http://proxy:8080"

    def test_both_proxies(self):
        with patch.dict(os.environ, {
            "HTTP_PROXY": "http://proxy:8080",
            "HTTPS_PROXY": "https://proxy:8443",
        }, clear=True):
            result = detect_system_proxy()
            assert result["http"] == "http://proxy:8080"
            assert result["https"] == "https://proxy:8443"


class TestApplySystemProxy:
    def test_applies_to_env(self):
        with patch.dict(os.environ, {}, clear=True):
            proxies = {"http": "http://proxy:8080", "https": "https://proxy:8443"}
            result = apply_system_proxy(proxies)
            assert result is not None
            assert os.environ.get("HTTP_PROXY") == "http://proxy:8080"
            assert os.environ.get("HTTPS_PROXY") == "https://proxy:8443"
            # lowercase variants too
            assert os.environ.get("http_proxy") == "http://proxy:8080"
            assert os.environ.get("https_proxy") == "https://proxy:8443"

    def test_does_not_override_existing(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://existing:8080"}, clear=True):
            result = apply_system_proxy({"http": "http://new:8080"}, override=False)
            assert os.environ["HTTP_PROXY"] == "http://existing:8080"

    def test_override_existing(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://existing:8080"}, clear=True):
            result = apply_system_proxy({"http": "http://new:8080"}, override=True)
            assert os.environ["HTTP_PROXY"] == "http://new:8080"

    def test_none_proxies_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert apply_system_proxy(None) is None


class TestPrintProxyInfo:
    def test_no_output_for_none(self, capsys):
        print_proxy_info(None)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_proxy(self, capsys):
        print_proxy_info({"http": "http://proxy:8080", "https": "https://proxy:8443"})
        captured = capsys.readouterr()
        assert "HTTP" in captured.out
        assert "http://proxy:8080" in captured.out
        assert "HTTPS" in captured.out
