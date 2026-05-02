from __future__ import annotations

from collections import deque
from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.base import BaseTranslationAPI
    from terminology_manager import TerminologyManager

CONTEXT_MARKER = "\U0001f524"  # 🔤


class TranslationHistoryEntry:
    """A single entry in the translation history."""

    def __init__(
        self,
        source_text: str,
        translated_text: str,
        target_language: str,
        speaker: Literal["me", "others"] = "me",
    ):
        self.source_text = source_text
        self.translated_text = translated_text
        self.target_language = target_language
        self.speaker = speaker


class ContextAwareTranslator:
    """Wraps a ``BaseTranslationAPI`` with context-awareness.

    Maintains a sliding window of recent translation history and passes
    that context to the underlying API on each call.
    """

    def __init__(
        self,
        translation_api: BaseTranslationAPI,
        api_name: str = "DefaultAPI",
        max_context_size: int = 6,
        target_language: str = "zh-CN",
        context_aware: bool = True,
        terminology_manager: Optional[TerminologyManager] = None,
    ):
        self.translation_api = translation_api
        self.api_name = api_name
        self.max_context_size = max_context_size
        self.target_language = target_language
        self.context_aware = context_aware
        self.terminology_manager = terminology_manager
        self._native_context_support = getattr(translation_api, "SUPPORTS_CONTEXT", False)
        self._contexts: deque = deque(maxlen=max_context_size)
        self._lock = __import__("threading").RLock()

    @property
    def display_contexts(self):
        with self._lock:
            return list(reversed(list(self._contexts)))

    def _previous_caption(self, count: Optional[int] = None) -> str:
        if count is None:
            count = len(self._contexts)
        if count <= 0:
            return ""
        with self._lock:
            entries = self.display_contexts[:count]
            if not entries:
                return ""
            parts = []
            for e in entries:
                prefix = "[Me] " if e.speaker == "me" else "[Others] "
                parts.append(f"{prefix}{e.source_text}")
            prefix = " ".join(parts)
            if prefix and not prefix.endswith((".", "\u3002", "!", "\uff01", "?", "\uff1f")):
                if any("\u4e00" <= c <= "\u9fff" for c in prefix):
                    prefix += "\u3002"
                else:
                    prefix += "."
            if prefix and not prefix.endswith(" "):
                prefix += " "
            return prefix

    def _previous_context_pairs(self, count: Optional[int] = None) -> list:
        if count is None:
            count = len(self._contexts)
        if count <= 0:
            return []
        with self._lock:
            entries = self.display_contexts[:count]
            pairs = []
            for e in entries:
                speaker_prefix = "[Me] " if e.speaker == "me" else "[Others] "
                pairs.append(
                    {
                        "source": f"{speaker_prefix}{e.source_text}",
                        "target": e.translated_text,
                        "speaker": e.speaker,
                    }
                )
            return pairs

    def add_external_speech(self, source_text: str) -> None:
        src = (source_text or "").strip()
        if not src:
            return
        with self._lock:
            self._contexts.append(
                TranslationHistoryEntry(
                    source_text=src,
                    translated_text="",
                    target_language="",
                    speaker="others",
                )
            )

    def append_history_entry(
        self,
        source_text: str,
        translated_text: str,
        target_language: Optional[str] = None,
        speaker: Literal["me", "others"] = "me",
    ) -> None:
        src = (source_text or "").strip()
        tgt = (translated_text or "").strip()
        if not src or not tgt or tgt.startswith("[ERROR]"):
            return
        lang = target_language if target_language is not None else self.target_language
        with self._lock:
            self._contexts.append(
                TranslationHistoryEntry(
                    source_text=src,
                    translated_text=tgt,
                    target_language=lang,
                    speaker=speaker,
                )
            )

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: Optional[str] = None,
        context_prefix: str = "",
        record_history: bool = True,
        **kwargs,
    ) -> str:
        if not text or not text.strip():
            return ""

        actual_target = target_language if target_language is not None else self.target_language

        try:
            if self._native_context_support:
                translated = self._translate_native(
                    text, source_language, actual_target, context_prefix, **kwargs
                )
            else:
                translated = self._translate_marker(
                    text, source_language, actual_target, **kwargs
                )

            translated = translated.strip()

            is_partial = bool(kwargs.get("is_partial"))
            if record_history and not is_partial and not translated.startswith("[ERROR]"):
                with self._lock:
                    self._contexts.append(
                        TranslationHistoryEntry(
                            source_text=text.strip(),
                            translated_text=translated,
                            target_language=actual_target,
                            speaker="me",
                        )
                    )

            return translated

        except Exception as e:
            return f"[ERROR] {e}"

    def _translate_native(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context_prefix: str,
        **kwargs,
    ) -> str:
        if self.context_aware and (len(self._contexts) > 0 or context_prefix):
            prev = self._previous_caption()
            current = f"[Me] {text.strip()}"
            if prev:
                ctx = f"{context_prefix}\n{prev}{current}"
            else:
                ctx = f"{context_prefix}\n{current}"
            if ctx and not ctx.endswith((".", "\u3002", "!", "\uff01", "?", "\uff1f")):
                if any("\u4e00" <= c <= "\u9fff" for c in ctx):
                    ctx += "\u3002"
                else:
                    ctx += "."
            if ctx and not ctx.endswith(" "):
                ctx += " "

            if self.terminology_manager is not None:
                hints = self.terminology_manager.get_terminology_hints(text.strip(), target_language)
                if hints:
                    ctx = f"{ctx}\n{hints}\n"

            pairs = self._previous_context_pairs()
            pairs.append({"source": text.strip(), "target": "", "speaker": "me"})

            return self.translation_api.translate(
                text,
                source_language=source_language,
                target_language=target_language,
                context=ctx,
                context_pairs=pairs,
                **kwargs,
            )
        else:
            if self.terminology_manager is not None:
                hints = self.terminology_manager.get_terminology_hints(text.strip(), target_language)
                if hints:
                    return self.translation_api.translate(
                        text,
                        source_language=source_language,
                        target_language=target_language,
                        context=hints,
                        **kwargs,
                    )
            return self.translation_api.translate(
                text,
                source_language=source_language,
                target_language=target_language,
                **kwargs,
            )

    def _translate_marker(
        self,
        text: str,
        source_language: str,
        target_language: str,
        **kwargs,
    ) -> str:
        prev = self._previous_caption() if self.context_aware and len(self._contexts) > 0 else ""
        input_text = f"{prev}\n{CONTEXT_MARKER}{text}{CONTEXT_MARKER}"
        translated = self.translation_api.translate(
            input_text,
            source_language=source_language,
            target_language=target_language,
            **kwargs,
        )
        if CONTEXT_MARKER in translated:
            try:
                end = translated.rfind(CONTEXT_MARKER)
                start = translated.rfind(CONTEXT_MARKER, 0, end)
                if start != -1 and end > start:
                    extracted = translated[start + len(CONTEXT_MARKER):end]
                    translated = extracted.strip()
            except Exception:
                pass
        return translated.replace(CONTEXT_MARKER, "").strip()

    def translate_with_context(self, text: str, source_language: str = "auto") -> Tuple[str, dict]:
        translated = self.translate(text, source_language)
        with self._lock:
            info = {
                "contexts_count": len(self._contexts),
                "previous_contexts": [
                    {"source": e.source_text, "translated": e.translated_text, "speaker": e.speaker}
                    for e in self.display_contexts[:-1]
                ],
            }
        return translated, info

    def set_context_aware(self, enabled: bool):
        with self._lock:
            self.context_aware = enabled

    def clear_contexts(self):
        with self._lock:
            self._contexts.clear()

    def set_target_language(self, language_code: str):
        with self._lock:
            self.target_language = language_code

    def __repr__(self):
        return (
            f"ContextAwareTranslator(api={self.api_name}, "
            f"max_context_size={self.max_context_size}, "
            f"target_language='{self.target_language}', "
            f"context_aware={self.context_aware}, "
            f"current_contexts={len(self._contexts)})"
        )

    def get_contexts(self) -> list:
        with self._lock:
            return [
                {
                    "source": e.source_text,
                    "translated": e.translated_text,
                    "language": e.target_language,
                    "speaker": e.speaker,
                }
                for e in self.display_contexts
            ]
