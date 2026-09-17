#!/usr/bin/env python3
"""Yakutan 远程推理服务 (Windows) 共享工具。

launcher.py 与 server_windows.py 共用，避免两份脚本各自复制一份实现后漂移。
"""

from __future__ import annotations

import socket

REQUIRED_MODULES: list[tuple[str, str]] = [
    ("websockets", "websockets>=14.0"),
    ("numpy", "numpy>=1.24.0"),
    ("onnxruntime", "onnxruntime / onnxruntime-directml / onnxruntime-gpu"),
    ("kaldi_native_fbank", "kaldi_native_fbank"),
    ("sentencepiece", "sentencepiece>=0.2.0"),
    ("soundfile", "soundfile>=0.12.0"),
    ("gguf", "gguf>=0.18.0"),
]


def check_port_in_use(port: int, hosts: tuple[str, ...] = ("127.0.0.1",)) -> bool:
    """探测端口是否被任一候选地址上的进程监听。"""
    for host in hosts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            continue
    return False


def get_local_ips() -> list[str]:
    """返回本机可用的 IPv4 地址列表（优先返回默认路由网卡对应的地址）。"""
    ips: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            preferred = s.getsockname()[0]
            if preferred and not preferred.startswith("127."):
                ips.append(preferred)
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = str(info[4][0])
            if ":" not in ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips or ["127.0.0.1"]


def check_dependencies() -> tuple[list[str], list[str]]:
    """逐个导入 REQUIRED_MODULES，返回 (已安装标签, 缺失标签)。"""
    installed: list[str] = []
    missing: list[str] = []
    for mod_name, label in REQUIRED_MODULES:
        try:
            __import__(mod_name)
            installed.append(label)
        except ImportError:
            missing.append(label)
    return installed, missing
