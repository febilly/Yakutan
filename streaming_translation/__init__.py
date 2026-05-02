"""
StreamingTranslation – a context-aware, multi-backend translation library.

Provides pluggable API backends (DeepL, Google, Qwen-MT, LLM via
OpenAI-compatible endpoints) wrapped in a ``ContextAwareTranslator`` that
maintains a sliding history window for coherent streaming output.
"""

from ._config import TranslationConfig, config_from_module
from .api.base import BaseTranslationAPI
from .api.deepl import DeepLAPI
from .api.google_dictionary import GoogleDictionaryAPI
from .api.google_web import GoogleWebAPI
from .api.openrouter import OpenRouterAPI, OpenRouterStreamingAPI, merge_with_draft
from .api.qwen_mt import QwenMTAPI
from .core.context_aware import ContextAwareTranslator, TranslationHistoryEntry
from .core.smart_language import SmartTargetLanguageSelector
from .pipeline import (
    TRANSLATION_API_CLASS_REGISTRY,
    DEFAULT_API_TYPE,
    clear_translation_contexts,
    ensure_secondary_translator,
    is_streaming_deepl_hybrid_mode,
    is_streaming_translation_mode,
    reinitialize_translator,
    reverse_translation,
    translate_with_backend,
    update_secondary_translator,
)

__all__ = [
    "BaseTranslationAPI",
    "ContextAwareTranslator",
    "DeepLAPI",
    "DEFAULT_API_TYPE",
    "GoogleDictionaryAPI",
    "GoogleWebAPI",
    "OpenRouterAPI",
    "OpenRouterStreamingAPI",
    "QwenMTAPI",
    "SmartTargetLanguageSelector",
    "TRANSLATION_API_CLASS_REGISTRY",
    "TranslationConfig",
    "TranslationHistoryEntry",
    "config_from_module",
    "clear_translation_contexts",
    "ensure_secondary_translator",
    "is_streaming_deepl_hybrid_mode",
    "is_streaming_translation_mode",
    "merge_with_draft",
    "reinitialize_translator",
    "reverse_translation",
    "translate_with_backend",
    "update_secondary_translator",
]
