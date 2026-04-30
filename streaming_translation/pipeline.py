"""
Translation pipeline – backend registry, factory, and high-level helpers.

Everything in this module accepts a ``TranslationConfig`` (or individual
parameters) instead of importing a global ``config`` module, making it
suitable for reuse outside of the original host application.
"""
from __future__ import annotations

import logging
from typing import Optional

from ._config import TranslationConfig
from .api.base import BaseTranslationAPI
from .api.deepl import DeepLAPI
from .api.google_dictionary import GoogleDictionaryAPI
from .api.google_web import GoogleWebAPI
from .api.openrouter import OpenRouterAPI, OpenRouterStreamingAPI
from .api.qwen_mt import QwenMTAPI
from .core.context_aware import ContextAwareTranslator

logger = logging.getLogger(__name__)

# ── API class registry ────────────────────────────────────────────────

TRANSLATION_API_CLASS_REGISTRY: dict[str, type[BaseTranslationAPI]] = {
    "google_web": GoogleWebAPI,
    "google_dictionary": GoogleDictionaryAPI,
    "openrouter": OpenRouterAPI,
    "openrouter_streaming": OpenRouterStreamingAPI,
    "openrouter_streaming_deepl_hybrid": OpenRouterStreamingAPI,
    "deepl": DeepLAPI,
    "qwen_mt": QwenMTAPI,
}

DEFAULT_API_TYPE = "qwen_mt"


# ── Helpers ───────────────────────────────────────────────────────────

def _normalize_optional_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip()
    return normalized or None


def is_streaming_translation_mode(api_type: str) -> bool:
    return api_type in ("openrouter_streaming", "openrouter_streaming_deepl_hybrid")


def is_streaming_deepl_hybrid_mode(api_type: str) -> bool:
    return api_type == "openrouter_streaming_deepl_hybrid"


def _get_api_class(api_type: str) -> type[BaseTranslationAPI]:
    return TRANSLATION_API_CLASS_REGISTRY.get(
        api_type,
        TRANSLATION_API_CLASS_REGISTRY[DEFAULT_API_TYPE],
    )


# ── Factory ───────────────────────────────────────────────────────────

def _build_api(api_class: type[BaseTranslationAPI], cfg: TranslationConfig) -> BaseTranslationAPI:
    """Instantiate a translation API backend from its class and config."""
    kwargs: dict = {}

    # Common: proxy
    if cfg.proxy_url:
        kwargs["proxy_url"] = cfg.proxy_url

    if issubclass(api_class, DeepLAPI):
        kwargs["api_key"] = cfg.deepl_api_key
        kwargs["formality"] = cfg.deepl_formality

    elif issubclass(api_class, (OpenRouterAPI, OpenRouterStreamingAPI)):
        kwargs["base_url"] = cfg.llm_base_url
        kwargs["model"] = cfg.llm_model
        kwargs["api_key"] = cfg.llm_api_key or cfg.openai_api_key
        kwargs["temperature"] = cfg.llm_temperature
        kwargs["timeout"] = cfg.llm_timeout
        kwargs["max_retries"] = cfg.llm_max_retries
        kwargs["formality"] = cfg.llm_formality
        kwargs["style"] = cfg.llm_style
        kwargs["extra_body_json"] = cfg.llm_extra_body_json
        kwargs["parallel_fastest_mode"] = cfg.llm_parallel_fastest_mode

    elif issubclass(api_class, QwenMTAPI):
        kwargs["api_key"] = cfg.dashscope_api_key
        kwargs["use_international"] = cfg.use_international_endpoint

    elif issubclass(api_class, GoogleDictionaryAPI):
        kwargs["proxy_url"] = cfg.proxy_url

    elif issubclass(api_class, GoogleWebAPI):
        kwargs["proxy_url"] = cfg.proxy_url

    return api_class(**kwargs)


def _build_context_translator(
    api_class: type[BaseTranslationAPI],
    target_language: str,
    cfg: TranslationConfig,
) -> tuple[BaseTranslationAPI, ContextAwareTranslator]:
    api_instance = _build_api(api_class, cfg)
    translator = ContextAwareTranslator(
        translation_api=api_instance,
        max_context_size=cfg.translation_context_size,
        target_language=target_language,
        context_aware=cfg.translation_context_aware,
    )
    return api_instance, translator


# ── State helpers (for host-app state objects) ────────────────────────

def _is_primary_config_changed(state, cfg: TranslationConfig) -> bool:
    return (
        cfg.translation_api_type != getattr(state, "translation_api_type", None)
        or cfg.target_language != getattr(state, "target_language", None)
        or cfg.llm_formality != getattr(state, "_last_llm_formality", None)
        or cfg.llm_style != getattr(state, "_last_llm_style", None)
        or cfg.deepl_formality != getattr(state, "_last_deepl_formality", None)
    )


def ensure_secondary_translator(
    state,
    target_language: Optional[str],
    cfg: Optional[TranslationConfig] = None,
) -> bool:
    """Ensure a secondary translator exists for *target_language*.

    Returns ``True`` if a translator is ready, ``False`` otherwise.
    ``state`` is any object with ``secondary_translator``,
    ``secondary_target_language``, etc. attributes.
    """
    if not target_language:
        state.secondary_translation_api = None
        state.secondary_translator = None
        state.secondary_deepl_fallback_translation_api = None
        state.secondary_deepl_fallback_translator = None
        state.secondary_target_language = None
        return False

    if (
        state.secondary_translator is not None
        and state.secondary_target_language == target_language
    ):
        return True

    state.secondary_translation_api = None
    state.secondary_translator = None
    state.secondary_deepl_fallback_translation_api = None
    state.secondary_deepl_fallback_translator = None

    if cfg is None:
        cfg = getattr(state, "_translation_config", None)
    if cfg is None:
        cfg = TranslationConfig()

    api_class = _get_api_class(cfg.translation_api_type)
    state.secondary_translation_api, state.secondary_translator = _build_context_translator(
        api_class, target_language, cfg,
    )
    if is_streaming_deepl_hybrid_mode(cfg.translation_api_type):
        try:
            (
                state.secondary_deepl_fallback_translation_api,
                state.secondary_deepl_fallback_translator,
            ) = _build_context_translator(DeepLAPI, target_language, cfg)
        except Exception as e:
            logger.warning("DeepL secondary init failed: %s", e)

    state.secondary_target_language = target_language
    logger.info("Secondary translator created: %s", target_language)
    return True


def reinitialize_translator(state, cfg: TranslationConfig) -> None:
    """(Re)initialise all translator instances on *state* from *cfg*."""
    if is_streaming_translation_mode(cfg.translation_api_type):
        cfg.translate_partial_results = True

    api_class = _get_api_class(cfg.translation_api_type)

    # Primary
    state.translation_api, state.translator = _build_context_translator(
        api_class, cfg.target_language, cfg,
    )

    # Secondary
    state.secondary_translation_api = None
    state.secondary_translator = None
    secondary_target = _normalize_optional_language_code(cfg.secondary_target_language)
    if secondary_target:
        state.secondary_translation_api, state.secondary_translator = _build_context_translator(
            api_class, secondary_target, cfg,
        )

    # Backwards (always Google Dictionary)
    state.backwards_translation_api = _build_api(GoogleDictionaryAPI, cfg)
    state.backwards_translator = ContextAwareTranslator(
        translation_api=state.backwards_translation_api,
        max_context_size=6,
        target_language="en",
        context_aware=False,
    )

    # DeepL fallback (hybrid mode only)
    state.deepl_fallback_translation_api = None
    state.deepl_fallback_translator = None
    state.secondary_deepl_fallback_translation_api = None
    state.secondary_deepl_fallback_translator = None
    if is_streaming_deepl_hybrid_mode(cfg.translation_api_type):
        try:
            state.deepl_fallback_translation_api, state.deepl_fallback_translator = (
                _build_context_translator(DeepLAPI, cfg.target_language, cfg)
            )
            if secondary_target:
                (
                    state.secondary_deepl_fallback_translation_api,
                    state.secondary_deepl_fallback_translator,
                ) = _build_context_translator(DeepLAPI, secondary_target, cfg)
        except Exception as e:
            logger.warning("DeepL fallback init failed: %s", e)

    state.translation_api_type = cfg.translation_api_type
    state.target_language = cfg.target_language
    state.secondary_target_language = secondary_target
    state._last_llm_formality = cfg.llm_formality
    state._last_llm_style = cfg.llm_style
    state._last_deepl_formality = cfg.deepl_formality


def update_secondary_translator(state, cfg: TranslationConfig) -> None:
    """Check and update the secondary translator if the config changed."""
    new_secondary = _normalize_optional_language_code(cfg.secondary_target_language)
    current_secondary = getattr(state, "secondary_target_language", None)
    if new_secondary == current_secondary:
        return

    state.secondary_translation_api = None
    state.secondary_translator = None
    state.secondary_deepl_fallback_translation_api = None
    state.secondary_deepl_fallback_translator = None

    if new_secondary:
        api_class = _get_api_class(cfg.translation_api_type)
        state.secondary_translation_api, state.secondary_translator = _build_context_translator(
            api_class, new_secondary, cfg,
        )
        if is_streaming_deepl_hybrid_mode(cfg.translation_api_type):
            try:
                (
                    state.secondary_deepl_fallback_translation_api,
                    state.secondary_deepl_fallback_translator,
                ) = _build_context_translator(DeepLAPI, new_secondary, cfg)
            except Exception as e:
                logger.warning("DeepL secondary update failed: %s", e)

    state.secondary_target_language = new_secondary
    logger.info("Secondary translator updated: %s -> %s", current_secondary, new_secondary)


# ── Translation execution ─────────────────────────────────────────────

def translate_with_backend(
    translator_instance: ContextAwareTranslator,
    deepl_translator_instance: Optional[ContextAwareTranslator],
    text: str,
    target_language: str,
    previous_translation: Optional[str] = None,
    prefer_deepl: bool = False,
    previous_source_text: Optional[str] = None,
    detected_source_language: Optional[str] = None,
    source_language: str = "auto",
    context_prefix: str = "",
    record_history: bool = True,
) -> str:
    """Translate *text* with optional DeepL-first fallback."""
    translate_kwargs = {
        "source_language": source_language,
        "target_language": target_language,
        "context_prefix": context_prefix,
        "is_partial": False,
        "record_history": record_history,
    }
    if previous_translation is not None:
        translate_kwargs["previous_translation"] = previous_translation
    if previous_source_text is not None:
        translate_kwargs["previous_source_text"] = previous_source_text
    if detected_source_language is not None:
        translate_kwargs["detected_source_language"] = detected_source_language

    if prefer_deepl and deepl_translator_instance is not None:
        result = deepl_translator_instance.translate(text, **translate_kwargs)
        if result and not result.startswith("[ERROR]"):
            return result

    return translator_instance.translate(text, **translate_kwargs)


def reverse_translation(
    backwards_translator: ContextAwareTranslator,
    translated_text: str,
    source_language: str,
    target_language: str,
) -> Optional[str]:
    """Reverse-translate the result (e.g. for verification display)."""
    try:
        result = backwards_translator.translate(
            translated_text,
            source_language=source_language,
            target_language=target_language,
        )
        return result
    except Exception as e:
        logger.warning("Reverse translation failed: %s", e)
        return None
