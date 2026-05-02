"""
Translation configuration data class.

Replaces the global ``import config`` pattern with a clean, injectable config object.
All consumers of the library receive a ``TranslationConfig`` instance rather than
importing a project-wide config module.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Optional


@dataclass
class TranslationConfig:
    """Central configuration for the streaming translation library.

    .. rubric:: Usage

    Create a config from the host application's settings::

        cfg = TranslationConfig(
            target_language="ja",
            translation_api_type="deepl",
            context_prefix="VRChat conversation:",
            proxy_url="http://127.0.0.1:7890",
        )

    Then pass ``cfg`` to ``reinitialize_translator``, API constructors, etc.
    """

    # ── Source / target language ──────────────────────────────────────
    source_language: str = "auto"
    """Source language code (``"auto"`` = auto-detect)."""

    target_language: str = "ja"
    """Primary target language code."""

    secondary_target_language: Optional[str] = None
    """Optional secondary target language for dual-output."""

    fallback_language: str = "en"
    """Fallback target when source == primary target."""

    # ── Translation API selection ─────────────────────────────────────
    translation_api_type: str = "qwen_mt"
    """Which API backend to use (key into the API class registry)."""

    translate_partial_results: bool = False
    """Whether to translate partial (streaming, mid-sentence) results."""

    # ── Context-aware translation ─────────────────────────────────────
    context_prefix: str = ""
    """Prefix describing the conversation setting (e.g. "VRChat voice chat")."""

    translation_context_size: int = 6
    """Number of recent messages kept as context."""

    translation_context_aware: bool = True
    """Enable/disable context-aware translation."""

    # ── Smart target-language selection ───────────────────────────────
    smart_target_primary_enabled: bool = False
    smart_target_secondary_enabled: bool = False
    smart_target_strategy: str = "most_common"
    smart_target_window_size: int = 5
    smart_target_exclude_self: bool = True
    smart_target_fallback: str = "en"
    smart_target_min_samples: int = 3
    smart_target_count: int = 2
    smart_target_manual_secondary: Optional[str] = None

    # ── LLM (OpenRouter / OpenAI-compatible) ──────────────────────────
    llm_base_url: str = ""
    """Base URL for the OpenAI-compatible API."""

    llm_model: str = ""
    """Model name for LLM translation."""

    llm_temperature: float = 0.2
    llm_timeout: int = 30
    llm_max_retries: int = 3
    llm_formality: str = "medium"
    llm_style: str = "light"
    llm_extra_body_json: str = ""
    llm_parallel_fastest_mode: str = "off"

    # ── Qwen-MT ───────────────────────────────────────────────────────
    use_international_endpoint: bool = False
    """Use DashScope international endpoint instead of domestic."""

    # ── Proxy ─────────────────────────────────────────────────────────
    proxy_url: Optional[str] = None
    """Optional HTTP/HTTPS proxy URL applied to all API backends."""

    # ── API keys (injected externally) ────────────────────────────────
    # These live in environment variables in the host application;
    # the library does **not** read env vars — it expects the caller to
    # pass them explicitly.
    deepl_api_key: Optional[str] = None
    dashscope_api_key: Optional[str] = None
    llm_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # ── Formality / style overrides ───────────────────────────────────
    deepl_formality: str = "default"
    """DeepL formality setting (``"default"``, ``"prefer_more"``, ``"prefer_less"``)."""


def _get_module_attr_or_env(module: object, attr_name: str, *env_names: str) -> Optional[str]:
    value = getattr(module, attr_name, None)
    if value is not None:
        return value
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    return None


def config_from_module(module: object) -> TranslationConfig:
    """Build a ``TranslationConfig`` from a module-like object (e.g. ``import config``).

    This is a convenience helper for the host application so you don't have to
    manually map every attribute::

        import config
        from streaming_translation._config import config_from_module

        cfg = config_from_module(config)
    """
    return TranslationConfig(
        source_language=getattr(module, "SOURCE_LANGUAGE", "auto"),
        target_language=getattr(module, "TARGET_LANGUAGE", "ja"),
        secondary_target_language=getattr(module, "SECONDARY_TARGET_LANGUAGE", None),
        fallback_language=getattr(module, "FALLBACK_LANGUAGE", "en"),
        translation_api_type=getattr(module, "TRANSLATION_API_TYPE", "qwen_mt"),
        translate_partial_results=getattr(module, "TRANSLATE_PARTIAL_RESULTS", False),
        context_prefix=getattr(module, "CONTEXT_PREFIX", ""),
        translation_context_size=getattr(module, "TRANSLATION_CONTEXT_SIZE", 6),
        translation_context_aware=getattr(module, "TRANSLATION_CONTEXT_AWARE", True),
        smart_target_primary_enabled=getattr(module, "SMART_TARGET_PRIMARY_ENABLED", False),
        smart_target_secondary_enabled=getattr(module, "SMART_TARGET_SECONDARY_ENABLED", False),
        smart_target_strategy=getattr(module, "SMART_TARGET_LANGUAGE_STRATEGY", "most_common"),
        smart_target_window_size=getattr(module, "SMART_TARGET_LANGUAGE_WINDOW_SIZE", 5),
        smart_target_exclude_self=getattr(module, "SMART_TARGET_LANGUAGE_EXCLUDE_SELF_LANGUAGE", True),
        smart_target_fallback=getattr(module, "SMART_TARGET_LANGUAGE_FALLBACK", "en"),
        smart_target_min_samples=getattr(module, "SMART_TARGET_LANGUAGE_MIN_SAMPLES", 3),
        smart_target_count=getattr(module, "SMART_TARGET_LANGUAGE_COUNT", 2),
        smart_target_manual_secondary=getattr(module, "SMART_TARGET_LANGUAGE_MANUAL_SECONDARY", None),
        llm_base_url=getattr(module, "LLM_BASE_URL", ""),
        llm_model=getattr(module, "LLM_MODEL", ""),
        llm_temperature=getattr(module, "LLM_TRANSLATION_TEMPERATURE", 0.2),
        llm_timeout=getattr(module, "LLM_TRANSLATION_TIMEOUT", 30),
        llm_max_retries=getattr(module, "LLM_TRANSLATION_MAX_RETRIES", 3),
        llm_formality=getattr(module, "LLM_TRANSLATION_FORMALITY", "medium"),
        llm_style=getattr(module, "LLM_TRANSLATION_STYLE", "light"),
        llm_extra_body_json=getattr(module, "OPENAI_COMPAT_EXTRA_BODY_JSON", ""),
        llm_parallel_fastest_mode=getattr(module, "LLM_PARALLEL_FASTEST_MODE", "off"),
        use_international_endpoint=getattr(module, "USE_INTERNATIONAL_ENDPOINT", False),
        proxy_url=None,
        deepl_api_key=_get_module_attr_or_env(module, "DEEPL_API_KEY", "DEEPL_API_KEY"),
        dashscope_api_key=_get_module_attr_or_env(module, "DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY"),
        llm_api_key=_get_module_attr_or_env(module, "LLM_API_KEY", "LLM_API_KEY", "OPENROUTER_API_KEY"),
        openai_api_key=_get_module_attr_or_env(module, "OPENAI_API_KEY", "OPENAI_API_KEY"),
        deepl_formality=getattr(module, "DEEPL_FORMALITY", "default"),
    )
