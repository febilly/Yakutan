"""Recent original speech for ASR hints, independent of translation history."""
from collections import deque
import threading
import time


class RecentASRContext:
    MAX_ITEMS = 5
    MAX_AGE_SECONDS = 120.0
    MAX_TEXT_CHARS = 400

    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._items = deque(maxlen=self.MAX_ITEMS)

    def add(self, text: str, *, foreign: bool = False) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        # Both interlocutors are humans: neither is an assistant reply.
        prefix = "对方：" if foreign else "自己："
        text = prefix + text.strip()[:self.MAX_TEXT_CHARS - len(prefix)]
        with self._lock:
            self._items.append((self._clock(), foreign, text))

    def snapshot(self) -> list[str]:
        with self._lock:
            cutoff = self._clock() - self.MAX_AGE_SECONDS
            while self._items and self._items[0][0] < cutoff:
                self._items.popleft()
            return [item[2] for item in self._items]

    def clear_foreign(self) -> None:
        with self._lock:
            self._items = deque(
                (item for item in self._items if not item[1]), maxlen=self.MAX_ITEMS,
            )
