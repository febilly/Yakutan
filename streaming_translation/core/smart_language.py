from __future__ import annotations

from collections import deque, Counter, defaultdict
from typing import Optional

from .._config import TranslationConfig


class SmartTargetLanguageSelector:
    """Selects translation target language(s) based on detected source language history.

    Useful for bilingual conversations where the target should follow the
    language being spoken to you.
    """

    def __init__(self, config: TranslationConfig):
        self._config = config
        self._history: deque[str] = deque()

    def reload_config(self, config: TranslationConfig) -> None:
        """Hot-reload configuration without recreating the selector.

        Call this when the caller's config changes at runtime so the
        selector picks up new ``smart_target_*`` settings immediately.
        History is preserved.
        """
        self._config = config
        window_size = self._config.smart_target_window_size
        if self._history.maxlen != window_size:
            self._history = deque(self._history, maxlen=window_size)

    def clear_history(self) -> None:
        self._history.clear()

    def record_language(self, detected_language: str) -> None:
        if not detected_language:
            return
        window_size = self._config.smart_target_window_size
        if self._history.maxlen != window_size:
            self._history = deque(self._history, maxlen=window_size)
        self._history.append(detected_language)

    def select_target_language(self, self_language: Optional[str] = None) -> list[str]:
        if not self._config.smart_target_primary_enabled and not self._config.smart_target_secondary_enabled:
            return []

        strategy = self._config.smart_target_strategy
        count = self._config.smart_target_count
        exclude_self = self._config.smart_target_exclude_self
        manual_secondary = self._config.smart_target_manual_secondary
        fallback = self._config.smart_target_fallback

        if count <= 0:
            return []

        history = list(self._history)

        if exclude_self and self_language is not None:
            norm_self = str(self_language).strip().lower()
            history = [lang for lang in history if str(lang).strip().lower() != norm_self]

        if not history:
            if count >= 2 and manual_secondary is not None:
                return [fallback, manual_secondary]
            return [fallback]

        selected: list[str] = []

        if strategy == "most_common":
            lang_counter = Counter(history)
            last_index = {lang: i for i, lang in enumerate(history)}
            ranked = sorted(
                lang_counter.items(),
                key=lambda item: (-item[1], -last_index.get(item[0], 0)),
            )
            for lang, _ in ranked[:count]:
                selected.append(lang)

        elif strategy == "latest":
            if history:
                selected.append(history[-1])

        elif strategy == "weighted":
            weights: dict[str, float] = defaultdict(float)
            decay = 0.9
            for i, lang in enumerate(history):
                weight = decay ** (len(history) - 1 - i)
                weights[lang] += weight
            last_index = {lang: i for i, lang in enumerate(history)}
            ranked = sorted(
                weights.items(),
                key=lambda item: (-item[1], -last_index.get(item[0], 0)),
            )
            for lang, _ in ranked[:count]:
                selected.append(lang)

        else:
            if history:
                selected.append(history[-1])

        primary = selected[0] if selected else fallback

        if count == 1:
            return [primary]

        if manual_secondary is not None:
            return [primary, manual_secondary]

        secondary = selected[1] if len(selected) > 1 else None
        return [primary, secondary]
