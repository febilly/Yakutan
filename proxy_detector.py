"""
系统代理检测工具
自动检测并应用系统网络代理设置
"""
import os
import urllib.request


def _normalize_proxies(proxies):
    """
    规范化代理字典，尽量补齐常见协议所需键名。

    Args:
        proxies: 原始代理字典

    Returns:
        dict | None: 规范化后的代理字典
    """
    if not proxies:
        return None

    normalized = {}
    http_proxy = proxies.get('http') or proxies.get('https')
    https_proxy = proxies.get('https') or proxies.get('http')

    if http_proxy:
        normalized['http'] = http_proxy
        normalized['ws'] = proxies.get('ws') or http_proxy
    if https_proxy:
        normalized['https'] = https_proxy
        normalized['wss'] = proxies.get('wss') or https_proxy

    all_proxy = proxies.get('all') or proxies.get('all_proxy')
    if all_proxy:
        normalized['all'] = all_proxy
    elif https_proxy or http_proxy:
        normalized['all'] = https_proxy or http_proxy

    no_proxy = proxies.get('no') or proxies.get('no_proxy')
    if no_proxy:
        normalized['no'] = no_proxy

    return normalized or None


def detect_system_proxy():
    """
    检测系统代理设置
    
    Returns:
        dict: 包含代理信息的字典，如果没有代理则返回 None
              格式: {'http': 'http://proxy:port', 'https': 'https://proxy:port'}
    """
    proxies = {}
    
    # 检查环境变量中的代理设置
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
    
    # 如果环境变量中没有设置，尝试使用 urllib 的系统代理检测
    if not proxies:
        try:
            proxy_handler = urllib.request.ProxyHandler()
            if proxy_handler.proxies:
                proxies = proxy_handler.proxies
        except Exception:
            pass
    
    return _normalize_proxies(proxies)


def apply_system_proxy(proxies=None, override=False):
    """
    将检测到的系统代理写入当前进程环境变量，供 requests/httpx/urllib/
    huggingface_hub/websockets 等库统一复用。

    Args:
        proxies: 可选的代理字典；未提供时自动检测
        override: 是否覆盖现有环境变量

    Returns:
        dict | None: 实际应用的代理字典
    """
    resolved = _normalize_proxies(proxies or detect_system_proxy())
    if not resolved:
        return None

    env_map = {
        'http': ('HTTP_PROXY', 'http_proxy'),
        'https': ('HTTPS_PROXY', 'https_proxy'),
        'all': ('ALL_PROXY', 'all_proxy'),
        'ws': ('WS_PROXY', 'ws_proxy'),
        'wss': ('WSS_PROXY', 'wss_proxy'),
        'no': ('NO_PROXY', 'no_proxy'),
    }

    for proxy_type, env_names in env_map.items():
        value = resolved.get(proxy_type)
        if not value:
            continue
        for env_name in env_names:
            if override or not os.environ.get(env_name):
                os.environ[env_name] = value

    return resolved


def print_proxy_info(proxies):
    """
    在命令行输出代理信息
    
    Args:
        proxies: 代理信息字典
    """
    if not proxies:
        return
    
    print('[代理] 检测到系统网络代理设置，已应用到当前进程：')
    for protocol, proxy_url in proxies.items():
        print(f'  {protocol.upper()}: {proxy_url}')
