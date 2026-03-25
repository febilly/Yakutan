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


def translate_with_backend(
    translator_instance: ContextAwareTranslator,
    deepl_translator_instance: Optional[ContextAwareTranslator],
    text: str,
    target_language: str,
    previous_translation: Optional[str] = None,
    prefer_deepl: bool = False,
) -> str:
    """使用指定翻译器执行翻译，可选 DeepL 优先。"""
    translate_kwargs = {
        'source_language': config.SOURCE_LANGUAGE,
        'target_language': target_language,
        'context_prefix': config.CONTEXT_PREFIX,
        'is_partial': False,
    }
    if previous_translation is not None:
        translate_kwargs['previous_translation'] = previous_translation

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
