"""
智能目标语言选择器
根据最近接收到的外语语音自动推断翻译目标语言
"""
from collections import deque, Counter
from typing import Optional


class SmartTargetLanguageSelector:
    """根据外语语音的历史语言分布，自动推断最适合的翻译目标语言。"""

    def __init__(self, config_module):
        self.config = config_module
        self._history: deque[str] = deque()

    def clear_history(self) -> None:
        """清空历史记录。"""
        self._history.clear()

    def record_language(self, detected_language: str) -> None:
        """记录一条 foreign speech 的语言。"""
        if not detected_language:
            return
        window_size = getattr(self.config, 'SMART_TARGET_LANGUAGE_WINDOW_SIZE', 10)
        if self._history.maxlen != window_size:
            self._history = deque(self._history, maxlen=window_size)
        self._history.append(detected_language)

    def select_target_language(self, self_language: Optional[str] = None) -> list[str]:
        """
        返回建议的目标语言列表。

        若 SMART_TARGET_LANGUAGE_ENABLED=False 返回空列表（由调用方回退到原有逻辑）。
        """
        if not getattr(self.config, 'SMART_TARGET_LANGUAGE_ENABLED', False):
            return []

        strategy = getattr(self.config, 'SMART_TARGET_LANGUAGE_STRATEGY', 'most_common')
        count = getattr(self.config, 'SMART_TARGET_LANGUAGE_COUNT', 1)
        exclude_self = getattr(self.config, 'SMART_TARGET_LANGUAGE_EXCLUDE_SELF_LANGUAGE', True)
        manual_secondary = getattr(self.config, 'SMART_TARGET_LANGUAGE_MANUAL_SECONDARY', None)
        fallback = getattr(self.config, 'SMART_TARGET_LANGUAGE_FALLBACK', 'en')

        if count <= 0:
            return []

        history = list(self._history)

        if exclude_self and self_language is not None:
            normalized_self = str(self_language).strip().lower()
            history = [
                lang for lang in history
                if str(lang).strip().lower() != normalized_self
            ]

        if not history:
            if count >= 2 and manual_secondary is not None:
                return [fallback, manual_secondary]
            return [fallback]

        selected: list[str] = []

        if strategy == 'most_common':
            lang_counter = Counter(history)
            last_index = {}
            for i, lang in enumerate(history):
                last_index[lang] = i
            ranked = sorted(
                lang_counter.items(),
                key=lambda item: (-item[1], -last_index.get(item[0], 0)),
            )
            for lang, _ in ranked[:count]:
                selected.append(lang)

        elif strategy == 'latest':
            if history:
                selected.append(history[-1])

        elif strategy == 'weighted':
            from collections import defaultdict
            weights: dict[str, float] = defaultdict(float)
            decay = 0.9
            for i, lang in enumerate(history):
                weight = decay ** (len(history) - 1 - i)
                weights[lang] += weight
            last_index = {}
            for i, lang in enumerate(history):
                last_index[lang] = i
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

        secondary = selected[1] if len(selected) > 1 else fallback
        return [primary, secondary]


_selector_instance: Optional[SmartTargetLanguageSelector] = None


def get_smart_selector() -> SmartTargetLanguageSelector:
    """获取全局单例 SmartTargetLanguageSelector。"""
    global _selector_instance
    if _selector_instance is None:
        import config
        _selector_instance = SmartTargetLanguageSelector(config)
    return _selector_instance
