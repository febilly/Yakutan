"""
翻译管道模块 - 负责翻译器实始化、API 注册表和翻译执行
"""
import logging
from typing import Optional

import config
from translators.context_aware_translator import ContextAwareTranslator
from translators.translation_apis.deepl_api import DeepLAPI
from translators.translation_apis.google_dictionary_api import GoogleDictionaryAPI
from translators.translation_apis.google_web_api import GoogleWebAPI
from translators.translation_apis.openrouter_api import (
    OpenRouterAPI,
    OpenRouterStreamingAPI,
)
from translators.translation_apis.qwen_mt_api import QwenMTAPI
from text_processor import normalize_optional_language_code

logger = logging.getLogger(__name__)

# ============ 翻译 API 注册表 ============
# 直接保存类对象，避免 importlib 动态导入在 PyInstaller 单文件打包时漏收子模块。
TRANSLATION_API_CLASS_REGISTRY = {
    'google_web': GoogleWebAPI,
    'google_dictionary': GoogleDictionaryAPI,
    'openrouter': OpenRouterAPI,
    'openrouter_streaming': OpenRouterStreamingAPI,
    'openrouter_streaming_deepl_hybrid': OpenRouterStreamingAPI,
    'deepl': DeepLAPI,
    'qwen_mt': QwenMTAPI,
}

DEFAULT_API_TYPE = 'qwen_mt'


def is_streaming_translation_mode(api_type: str) -> bool:
    return api_type in ('openrouter_streaming', 'openrouter_streaming_deepl_hybrid')


def is_streaming_deepl_hybrid_mode() -> bool:
    return config.TRANSLATION_API_TYPE == 'openrouter_streaming_deepl_hybrid'


def _get_translation_api_class(api_type: str):
    """根据 API 类型字符串，从注册表获取翻译 API 类。"""
    return TRANSLATION_API_CLASS_REGISTRY.get(
        api_type,
        TRANSLATION_API_CLASS_REGISTRY[DEFAULT_API_TYPE],
    )


def _build_context_translator(api_factory, target_language: str):
    """创建翻译 API 实例及其 ContextAwareTranslator 包装。"""
    translation_api_instance = api_factory()
    translator_instance = ContextAwareTranslator(
        translation_api=translation_api_instance,
        max_context_size=6,
        target_language=target_language,
        context_aware=True,
    )
    return translation_api_instance, translator_instance


def _is_primary_translator_config_changed(state) -> bool:
    return (
        config.TRANSLATION_API_TYPE != getattr(state, 'translation_api_type', None)
        or config.TARGET_LANGUAGE != getattr(state, 'target_language', None)
    )


def update_secondary_translator(state):
    new_secondary = normalize_optional_language_code(
        getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)
    )
    current_secondary = getattr(state, 'secondary_target_language', None)
    if new_secondary == current_secondary:
        return

    state.secondary_translation_api = None
    state.secondary_translator = None
    state.secondary_deepl_fallback_translation_api = None
    state.secondary_deepl_fallback_translator = None

    if new_secondary:
        TranslationAPIClass = _get_translation_api_class(config.TRANSLATION_API_TYPE)
        state.secondary_translation_api, state.secondary_translator = (
            _build_context_translator(TranslationAPIClass, new_secondary)
        )
        if is_streaming_deepl_hybrid_mode():
            try:
                (
                    state.secondary_deepl_fallback_translation_api,
                    state.secondary_deepl_fallback_translator,
                ) = _build_context_translator(DeepLAPI, new_secondary)
            except Exception as e:
                logger.warning(
                    "[Translation] 混合模式下 DeepL 第二翻译器初始化失败，"
                    "将回退 LLM 终译: %s",
                    e,
                )

    state.secondary_target_language = new_secondary
    logger.info(
        "[Translation] 第二翻译器已更新: %s -> %s", current_secondary, new_secondary
    )


def reinitialize_translator(state):
    """根据当前配置动态（重）初始化翻译器实例。

    Args:
        state: AppState 实例，翻译器实例将直接设置到其属性上。
    """
    if is_streaming_translation_mode(config.TRANSLATION_API_TYPE):
        config.TRANSLATE_PARTIAL_RESULTS = True

    TranslationAPIClass = _get_translation_api_class(config.TRANSLATION_API_TYPE)

    # 主翻译器
    state.translation_api, state.translator = _build_context_translator(
        TranslationAPIClass, config.TARGET_LANGUAGE,
    )

    # 第二翻译器（可选）
    state.secondary_translation_api = None
    state.secondary_translator = None
    secondary_target_language = normalize_optional_language_code(
        getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)
    )
    if secondary_target_language:
        state.secondary_translation_api, state.secondary_translator = (
            _build_context_translator(TranslationAPIClass, secondary_target_language)
        )

    # 反向翻译器（始终使用 Google Dictionary）
    state.backwards_translation_api = GoogleDictionaryAPI()
    state.backwards_translator = ContextAwareTranslator(
        translation_api=state.backwards_translation_api,
        max_context_size=6,
        target_language="en",
        context_aware=False,
    )

    # DeepL fallback（仅混合模式）
    state.deepl_fallback_translation_api = None
    state.deepl_fallback_translator = None
    state.secondary_deepl_fallback_translation_api = None
    state.secondary_deepl_fallback_translator = None
    if is_streaming_deepl_hybrid_mode():
        try:
            state.deepl_fallback_translation_api, state.deepl_fallback_translator = (
                _build_context_translator(DeepLAPI, config.TARGET_LANGUAGE)
            )
            if secondary_target_language:
                (
                    state.secondary_deepl_fallback_translation_api,
                    state.secondary_deepl_fallback_translator,
                ) = _build_context_translator(DeepLAPI, secondary_target_language)
        except Exception as e:
            logger.warning(
                "[Translation] 混合模式下 DeepL 初始化失败，将回退 LLM 终译: %s", e
            )

    state.translation_api_type = config.TRANSLATION_API_TYPE
    state.target_language = config.TARGET_LANGUAGE
    state.secondary_target_language = secondary_target_language


def translate_with_backend(
    translator_instance: ContextAwareTranslator,
    deepl_translator_instance: Optional[ContextAwareTranslator],
    text: str,
    target_language: str,
    previous_translation: Optional[str] = None,
    prefer_deepl: bool = False,
    previous_source_text: Optional[str] = None,
    detected_source_language: Optional[str] = None,
    record_history: bool = True,
) -> str:
    """使用指定翻译器执行翻译，可选 DeepL 优先。"""
    translate_kwargs = {
        'source_language': config.SOURCE_LANGUAGE,
        'target_language': target_language,
        'context_prefix': config.CONTEXT_PREFIX,
        'is_partial': False,
        'record_history': record_history,
    }
    if previous_translation is not None:
        translate_kwargs['previous_translation'] = previous_translation
    if previous_source_text is not None:
        translate_kwargs['previous_source_text'] = previous_source_text
    if detected_source_language is not None:
        translate_kwargs['detected_source_language'] = detected_source_language

    if prefer_deepl and deepl_translator_instance is not None:
        translated_text = deepl_translator_instance.translate(text, **translate_kwargs)
        if translated_text and not translated_text.startswith("[ERROR]"):
            return translated_text

    return translator_instance.translate(text, **translate_kwargs)


def reverse_translation(backwards_translator, translated_text, source_language, target_language):
    """对翻译结果进行反向翻译。"""
    try:
        backwards_translated = backwards_translator.translate(
            translated_text,
            source_language=source_language,
            target_language=target_language,
        )
        print(f'反向翻译：{backwards_translated}')
        return backwards_translated
    except Exception as e:
        print(f'反向翻译失败: {e}')
        return None
