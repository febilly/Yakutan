"""
串行化对 PyAudio / PortAudio 的敏感访问。

Windows 下 PortAudio 在并发 init/open/terminate/device-enumeration 时容易出现
崩溃或返回异常结果，因此这里提供一个跨模块共享的全局锁。
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Iterator

_PORTAUDIO_LOCK = threading.RLock()


@contextmanager
def _suppress_stderr() -> Iterator[None]:
    try:
        original_stderr = sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            yield
    finally:
        sys.stderr = original_stderr


@contextmanager
def hold_portaudio(label: str = "") -> Iterator[None]:
    del label  # 仅保留调用点的语义，不在此模块内使用
    with _PORTAUDIO_LOCK:
        yield
