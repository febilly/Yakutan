import logging
import os
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording)
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pyaudio

from dotenv import load_dotenv
from translators.context_aware_translator import ContextAwareTranslator
from hot_words_manager import HotWordsManager
from proxy_detector import detect_system_proxy, print_proxy_info

from translators.translation_apis.google_dictionary_api import GoogleDictionaryAPI as BackwardsTranslationAPI
from speech_recognizers.base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
    SpeechRecognizer,
)
from speech_recognizers.recognizer_factory import (
    init_dashscope_api_key,
    create_recognizer,
    select_backend,
)

# 导入配置
import config
from audio_debug_recorder import WaveDebugRecorder
from audio_resampler import AudioResampler

# 加载 .env 文件中的环境变量
load_dotenv()

from osc_manager import osc_manager

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ============ 根据配置选择语言检测器 ============
if config.LANGUAGE_DETECTOR_TYPE == 'fasttext':
    from language_detectors.fasttext_detector import FasttextDetector as LanguageDetector
elif config.LANGUAGE_DETECTOR_TYPE == 'enzh':
    from language_detectors.enzh_detector import EnZhDetector as LanguageDetector
else:  # 默认使用 cjke
    from language_detectors.cjke_detector import CJKEDetector as LanguageDetector

# ============ 根据配置选择翻译 API ============
if config.TRANSLATION_API_TYPE == 'google_web':
    from translators.translation_apis.google_web_api import GoogleWebAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'google_dictionary':
    from translators.translation_apis.google_dictionary_api import GoogleDictionaryAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'openrouter':
    from translators.translation_apis.openrouter_api import OpenRouterAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'openrouter_streaming':
    from translators.translation_apis.openrouter_api import OpenRouterStreamingAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'openrouter_streaming_deepl_hybrid':
    from translators.translation_apis.openrouter_api import OpenRouterStreamingAPI as TranslationAPI
elif config.TRANSLATION_API_TYPE == 'deepl':
    from translators.translation_apis.deepl_api import DeepLAPI as TranslationAPI
else:  # 默认使用 qwen_mt
    from translators.translation_apis.qwen_mt_api import QwenMTAPI as TranslationAPI


def is_streaming_translation_mode(api_type: str) -> bool:
    return api_type in ('openrouter_streaming', 'openrouter_streaming_deepl_hybrid')


def is_streaming_deepl_hybrid_mode() -> bool:
    return config.TRANSLATION_API_TYPE == 'openrouter_streaming_deepl_hybrid'


DUAL_OUTPUT_SEPARATOR = "\n"
DUAL_OUTPUT_TOTAL_MAX_CHARS = 144
DUAL_OUTPUT_BODY_BUDGET = max(0, DUAL_OUTPUT_TOTAL_MAX_CHARS - len(DUAL_OUTPUT_SEPARATOR))
DUAL_OUTPUT_MAX_CHARS_PER_RESULT = DUAL_OUTPUT_BODY_BUDGET // 2

# 双语裁剪时：中日韩权重=1，其他语言权重=2。
COMPACT_SCRIPT_LANGUAGE_BASES = {'zh', 'ja', 'ko'}
COMPACT_SCRIPT_BUDGET_WEIGHT = 1
ALPHABETIC_SCRIPT_BUDGET_WEIGHT = 2


def _normalize_optional_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip()
    return normalized or None


def _normalize_lang_code(lang):
    """标准化语言代码"""
    if not lang:
        return 'auto'
    lang_lower = str(lang).lower()
    if lang_lower in ['zh', 'zh-cn', 'zh-tw', 'zh-hans', 'zh-hant']:
        return 'zh'
    if lang_lower in ['en', 'en-us', 'en-gb']:
        return 'en'
    return lang_lower


def has_secondary_translation_target() -> bool:
    return _normalize_optional_language_code(getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)) is not None


def resolve_output_target_language(source_language: str, requested_target_language: Optional[str]) -> Optional[str]:
    target_language = _normalize_optional_language_code(requested_target_language)
    if target_language is None:
        return None

    fallback_language = _normalize_optional_language_code(getattr(config, 'FALLBACK_LANGUAGE', None))
    if fallback_language and _normalize_lang_code(source_language) == _normalize_lang_code(target_language):
        return fallback_language

    return target_language


def _sanitize_output_line(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    return " ".join(part.strip() for part in normalized.split('\n') if part.strip())


def _normalize_language_base(language: Optional[str]) -> str:
    if language is None:
        return ""
    normalized = str(language).strip().lower().replace('_', '-')
    if not normalized:
        return ""
    return normalized.split('-', 1)[0]


def _is_compact_script_language(language: Optional[str]) -> bool:
    return _normalize_language_base(language) in COMPACT_SCRIPT_LANGUAGE_BASES


def _get_language_budget_weight(language: Optional[str]) -> float:
    if _is_compact_script_language(language):
        return COMPACT_SCRIPT_BUDGET_WEIGHT
    return ALPHABETIC_SCRIPT_BUDGET_WEIGHT


def _allocate_dual_output_budgets(
    primary_language: Optional[str],
    secondary_language: Optional[str],
    total_chars: int = DUAL_OUTPUT_BODY_BUDGET,
) -> tuple[int, int]:
    if total_chars <= 0:
        return 0, 0

    primary_weight = _get_language_budget_weight(primary_language)
    secondary_weight = _get_language_budget_weight(secondary_language)
    total_weight = primary_weight + secondary_weight

    if total_chars == 1:
        return 1, 0

    if total_weight <= 0:
        primary_budget = total_chars // 2
    else:
        primary_budget = int(round(total_chars * (primary_weight / total_weight)))

    primary_budget = max(1, min(total_chars - 1, primary_budget))
    secondary_budget = total_chars - primary_budget
    return primary_budget, secondary_budget


def limit_dual_output_text(text: str, max_chars: int = DUAL_OUTPUT_MAX_CHARS_PER_RESULT) -> str:
    sanitized = _sanitize_output_line(text)
    if len(sanitized) <= max_chars:
        return sanitized
    return sanitized[:max_chars].rstrip()


def translate_with_backend(
    translator_instance: ContextAwareTranslator,
    deepl_translator_instance: Optional[ContextAwareTranslator],
    text: str,
    target_language: str,
    previous_translation: Optional[str] = None,
    prefer_deepl: bool = False,
) -> str:
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


def get_display_translation_text(translated_text: str, target_language: str) -> str:
    display_text = add_furigana_if_needed(translated_text, target_language)
    return add_pinyin_if_needed(display_text, target_language)


def build_dual_output_display(
    primary_text: str,
    secondary_text: Optional[str],
    primary_language: Optional[str] = None,
    secondary_language: Optional[str] = None,
) -> str:
    if secondary_text is None:
        return limit_dual_output_text(primary_text)

    primary_sanitized = _sanitize_output_line(primary_text)
    secondary_sanitized = _sanitize_output_line(secondary_text)
    full_text = DUAL_OUTPUT_SEPARATOR.join([primary_sanitized, secondary_sanitized])

    # 能完整装下时不做任何裁剪。
    if len(full_text) <= DUAL_OUTPUT_TOTAL_MAX_CHARS:
        return full_text

    primary_budget, secondary_budget = _allocate_dual_output_budgets(
        primary_language,
        secondary_language,
        total_chars=DUAL_OUTPUT_BODY_BUDGET,
    )

    clipped_primary = limit_dual_output_text(primary_sanitized, max_chars=primary_budget)
    clipped_secondary = limit_dual_output_text(secondary_sanitized, max_chars=secondary_budget)
    return DUAL_OUTPUT_SEPARATOR.join([clipped_primary, clipped_secondary])


def build_streaming_output_line(text: str) -> str:
    formatted_text = _sanitize_output_line(text)
    if formatted_text:
        return f"{formatted_text}……"
    return "……"

# ============ 可选的日语假名标注支持 ============
try:
    from pykakasi import kakasi as _kakasi_factory

    _kakasi = _kakasi_factory()
    _kakasi.setMode("J", "H")  # Kanji -> Hiragana
    _kakasi.setMode("K", "H")  # Katakana -> Hiragana
    _kakasi.setMode("H", "H")  # Hiragana stays Hiragana
except Exception:
    _kakasi = None

# ============ 可选的中文拼音标注支持 ============
try:
    from pypinyin import pinyin, Style
    _pypinyin_available = True
except ImportError:
    _pypinyin_available = False


def _contains_kanji(text: str) -> bool:
    """Check if the text contains any CJK ideographs."""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def _contains_chinese(text: str) -> bool:
    """Check if the text contains Chinese characters."""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def add_furigana(text: str) -> str:
    """Add hiragana readings to Japanese text with kanji.
    
    This function adds furigana regardless of configuration - 
    the caller should check ENABLE_JA_FURIGANA before calling.
    """
    if not text:
        return text

    if _kakasi is None:
        return text

    try:
        tokens = _kakasi.convert(text)
        parts = []
        for token in tokens:
            orig = token.get('orig', '')
            hira = token.get('hira') or token.get('kana')

            if orig and _contains_kanji(orig) and hira and hira != orig:
                parts.append(f"{orig}({hira})")
            else:
                parts.append(orig)

        return "".join(parts)
    except Exception:
        return text


def add_pinyin(text: str) -> str:
    """Add pinyin with tones to Chinese text, grouped by words.
    
    Uses jieba for word segmentation. Output format: 大家dà'jiā晚上好wǎn'shàng'hǎo
    Pinyin is obtained for the entire sentence at once for correct pronunciation of polyphonic characters.
    """
    if not text or not _pypinyin_available:
        return text
    
    if not _contains_chinese(text):
        return text
    
    try:
        import jieba
        
        # 先获取整句话的拼音（这样多音字会根据上下文正确发音）
        # pypinyin 会为每个字符返回一个列表项，包括标点符号
        full_pinyin = pinyin(text, style=Style.TONE)
        
        # 建立字符位置到拼音的映射（只取中文字符的拼音）
        char_to_pinyin = {}
        for i, char in enumerate(text):
            if i < len(full_pinyin):
                py = full_pinyin[i][0]
                # 只有当拼音与原字符不同时才记录（标点符号的"拼音"等于自身）
                if _contains_chinese(char) and py != char:
                    char_to_pinyin[i] = py
        
        # 使用 jieba 分词
        words = list(jieba.cut(text))
        
        result_parts = []
        char_index = 0
        
        for word in words:
            if _contains_chinese(word):
                # 收集这个词中每个中文字符的拼音
                word_pinyins = []
                for char in word:
                    if char_index in char_to_pinyin:
                        word_pinyins.append(char_to_pinyin[char_index])
                    char_index += 1
                
                # 用单引号连接多个音节
                if word_pinyins:
                    py_str = "'".join(word_pinyins)
                    result_parts.append(f"{word}{py_str}")
                else:
                    result_parts.append(word)
            else:
                # 非中文词（含标点、空格、英文等），直接添加
                result_parts.append(word)
                char_index += len(word)
        
        return "".join(result_parts)
    except ImportError:
        # jieba 未安装
        return text
    except Exception:
        return text


def add_furigana_if_needed(text: str, language: str) -> str:
    """Add furigana to text if it's Japanese and furigana is enabled."""
    if not text or not getattr(config, 'ENABLE_JA_FURIGANA', False):
        return text
    
    lang = (language or '').lower()
    if not lang.startswith('ja'):
        return text
    
    return add_furigana(text)


def add_pinyin_if_needed(text: str, language: str) -> str:
    """Add pinyin to text if it's Chinese and pinyin is enabled."""
    if not text or not getattr(config, 'ENABLE_ZH_PINYIN', False):
        return text
    
    lang = (language or '').lower()
    if not lang.startswith('zh'):
        return text
    
    return add_pinyin(text)


# ============ 全局变量 ============
RECOGNIZER_CHANNELS = 1
mic = None
stream = None
input_sample_rate = config.SAMPLE_RATE  # 实际采集采样率（可能与 config.SAMPLE_RATE 不同）
input_block_size = config.BLOCK_SIZE   # 实际采集每次 read 的帧数
capture_channels = config.CHANNELS  # 实际采集声道数（可能与送往识别器的声道数不同）
_resampler: 'AudioResampler | None' = None  # 实时重采样器
_debug_pre_audio_recorder: 'WaveDebugRecorder | None' = None  # 重采样前调试音频录制器
_debug_audio_recorder: 'WaveDebugRecorder | None' = None  # 调试音频录制器
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
stop_event = None  # 将在 main() 函数中创建，避免绑定到错误的事件循环
recognition_active = False  # 标记识别是否正在运行
recognition_started = False  # 标记是否已建立识别会话
recognition_instance: Optional[SpeechRecognizer] = None  # 全局识别实例
mute_delay_task = None  # 延迟停止任务
main_loop: Optional[asyncio.AbstractEventLoop] = None  # 主事件循环（用于线程安全调度）
CURRENT_ASR_BACKEND = config.PREFERRED_ASR_BACKEND
vocabulary_id = None  # 热词表 ID
recognition_callback = None  # 识别回调实例（用于静音触发状态联动）
PAUSE_RESUME_BACKENDS = {'qwen', 'soniox', 'doubao_file'}

# 字幕状态（供控制面板轮询）
subtitles_state = {
    "original": "",
    "translated": "",
    "reverse_translated": "",
    "ongoing": False
}

def update_subtitles(original: str, translated: str, ongoing: bool, reverse_translated: str = ""):
    subtitles_state["original"] = original
    subtitles_state["translated"] = translated
    subtitles_state["reverse_translated"] = reverse_translated
    subtitles_state["ongoing"] = ongoing

def is_doubao_file_backend() -> bool:
    return CURRENT_ASR_BACKEND == 'doubao_file'


def is_effective_mic_control_enabled() -> bool:
    return config.ENABLE_MIC_CONTROL or is_doubao_file_backend()


def should_output_partial_results() -> bool:
    return config.SHOW_PARTIAL_RESULTS and not is_doubao_file_backend()

# ============ 初始化服务实例 ============

def reinitialize_translator():
    """根据当前配置动态（重）初始化翻译器实例。
    在运行时切换翻译 API 或目标语言后调用此函数以使后端使用新配置。
    """
    global translation_api, translator, secondary_translation_api, secondary_translator
    global backwards_translation_api, backwards_translator
    global deepl_fallback_translation_api, deepl_fallback_translator
    global secondary_deepl_fallback_translation_api, secondary_deepl_fallback_translator

    if is_streaming_translation_mode(config.TRANSLATION_API_TYPE):
        config.TRANSLATE_PARTIAL_RESULTS = True

    # 根据配置选择翻译 API 类（延迟导入以避免循环导入或不必要的实例化）
    if config.TRANSLATION_API_TYPE == 'google_web':
        from translators.translation_apis.google_web_api import GoogleWebAPI as TranslationAPIClass
    elif config.TRANSLATION_API_TYPE == 'google_dictionary':
        from translators.translation_apis.google_dictionary_api import GoogleDictionaryAPI as TranslationAPIClass
    elif config.TRANSLATION_API_TYPE == 'openrouter':
        from translators.translation_apis.openrouter_api import OpenRouterAPI as TranslationAPIClass
    elif config.TRANSLATION_API_TYPE == 'openrouter_streaming':
        from translators.translation_apis.openrouter_api import OpenRouterStreamingAPI as TranslationAPIClass
    elif config.TRANSLATION_API_TYPE == 'openrouter_streaming_deepl_hybrid':
        from translators.translation_apis.openrouter_api import OpenRouterStreamingAPI as TranslationAPIClass
    elif config.TRANSLATION_API_TYPE == 'deepl':
        from translators.translation_apis.deepl_api import DeepLAPI as TranslationAPIClass
    else:  # 默认使用 qwen_mt
        from translators.translation_apis.qwen_mt_api import QwenMTAPI as TranslationAPIClass

    def _build_context_translator(api_factory, target_language: str):
        translation_api_instance = api_factory()
        translator_instance = ContextAwareTranslator(
            translation_api=translation_api_instance,
            max_context_size=6,
            target_language=target_language,
            context_aware=True
        )
        return translation_api_instance, translator_instance

    # 创建新的翻译 API 实例并注入 ContextAwareTranslator
    translation_api, translator = _build_context_translator(TranslationAPIClass, config.TARGET_LANGUAGE)

    secondary_translation_api = None
    secondary_translator = None
    secondary_target_language = _normalize_optional_language_code(getattr(config, 'SECONDARY_TARGET_LANGUAGE', None))
    if secondary_target_language:
        secondary_translation_api, secondary_translator = _build_context_translator(
            TranslationAPIClass,
            secondary_target_language,
        )

    # 反向翻译器保持使用原先的后端（Google Dictionary），重新创建以确保同步配置
    backwards_translation_api = BackwardsTranslationAPI()
    backwards_translator = ContextAwareTranslator(
        translation_api=backwards_translation_api,
        max_context_size=6,
        target_language="en",
        context_aware=False
    )

    deepl_fallback_translation_api = None
    deepl_fallback_translator = None
    secondary_deepl_fallback_translation_api = None
    secondary_deepl_fallback_translator = None
    if is_streaming_deepl_hybrid_mode():
        try:
            from translators.translation_apis.deepl_api import DeepLAPI
            deepl_fallback_translation_api, deepl_fallback_translator = _build_context_translator(
                DeepLAPI,
                config.TARGET_LANGUAGE,
            )
            if secondary_target_language:
                secondary_deepl_fallback_translation_api, secondary_deepl_fallback_translator = _build_context_translator(
                    DeepLAPI,
                    secondary_target_language,
                )
        except Exception as e:
            logger.warning(f"[Translation] 混合模式下 DeepL 初始化失败，将回退 LLM 终译: {e}")


# 在模块加载时进行一次初始化
reinitialize_translator()

language_detector = LanguageDetector()
# ================================


def reverse_translation(translated_text, source_language, target_language):
    """
    对翻译结果进行反向翻译，从目标语言翻译回原始语言
    
    Args:
        translated_text: 已翻译的文本
        source_language: **本方法进行的翻译的** 源语言代码
        target_language: **本方法进行的翻译的** 目标语言代码
    
    Returns:
        反向翻译后的文本
    """
    try:
        backwards_translated = backwards_translator.translate(
            translated_text,
            source_language=source_language,
            target_language=target_language
        )
        print(f'反向翻译：{backwards_translated}')
        return backwards_translated
    except Exception as e:
        print(f'反向翻译失败: {e}')
        return None


# Real-time speech recognition callback
class VRChatRecognitionCallback(SpeechRecognitionCallback):
    def __init__(self):
        self.loop = None  # 将在主线程中设置
        self.last_partial_translation = None # 用于流式翻译
        self.last_partial_translation_secondary = None
        self.translating_partial = False # 标记是否正在进行流式翻译
        self.last_partial_source_segment = None
        self.pending_partial_segment = None
        self._partial_request_seq = 0
        self._latest_partial_request_id = 0
        self._partial_inflight = 0
        self._finalized_seq = 0
        self._final_output_version = 0
        self.partial_translation_update_count = 0
        self._prefer_deepl_on_next_final = False
        self._partial_debounce_handle: Optional[asyncio.TimerHandle] = None

    def mark_mute_finalization_requested(self) -> None:
        self._prefer_deepl_on_next_final = True

    def clear_mute_finalization_requested(self) -> None:
        self._prefer_deepl_on_next_final = False

    @staticmethod
    def _normalize_lang(lang):
        """标准化语言代码"""
        return _normalize_lang_code(lang)

    @staticmethod
    def _extract_streaming_segment(text: str) -> Optional[str]:
        """从识别文本中截取可用于流式翻译的片段"""
        if not text:
            return None

        punctuation_chars = ("。", "？", "！", "，", "、", ".", "?", "!", ",")
        for idx in range(len(text) - 1, -1, -1):
            if text[idx] in punctuation_chars:
                remainder = text[idx + 1:]
                if remainder and remainder.strip():
                    segment = text[:idx + 1].strip()
                    if segment:
                        return segment
        return None

    @staticmethod
    def _should_trigger_partial_translation(segment: Optional[str]) -> bool:
        if not segment:
            return False

        min_chars = max(0, int(getattr(config, 'MIN_PARTIAL_TRANSLATION_CHARS', 2)))
        normalized_segment = segment.strip().rstrip("。？！，、.?!,… ")
        return len(normalized_segment) >= min_chars

    def _cancel_partial_debounce(self) -> None:
        """取消尚未触发的流式翻译消抖定时器。"""

        def _cancel() -> None:
            if self._partial_debounce_handle is not None:
                self._partial_debounce_handle.cancel()
                self._partial_debounce_handle = None

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(_cancel)
        else:
            _cancel()

    def _schedule_partial_translation_with_debounce(
        self,
        segment: str,
        request_id: int,
        finalized_seq: int,
        final_output_version: int,
    ) -> None:
        """对流式翻译请求做消抖：仅在短时间内无更新时才发送请求。"""
        if not self.loop:
            return

        debounce_ms = max(0, int(getattr(config, 'PARTIAL_TRANSLATION_DEBOUNCE_MS', 50)))
        debounce_seconds = debounce_ms / 1000.0

        def _dispatch() -> None:
            self._partial_debounce_handle = None
            if (
                request_id != self._latest_partial_request_id or
                finalized_seq != self._finalized_seq or
                final_output_version != self._final_output_version
            ):
                return
            asyncio.create_task(
                self._translate_partial_task(
                    segment,
                    request_id,
                    finalized_seq,
                    final_output_version,
                )
            )

        def _schedule() -> None:
            if self._partial_debounce_handle is not None:
                self._partial_debounce_handle.cancel()
            self._partial_debounce_handle = self.loop.call_later(
                debounce_seconds,
                _dispatch,
            )

        self.loop.call_soon_threadsafe(_schedule)
    
    def on_session_started(self) -> None:
        logger.info('Speech recognizer session opened.')
        self._cancel_partial_debounce()
        self.last_partial_translation = None
        self.last_partial_translation_secondary = None
        self.last_partial_source_segment = None
        self.pending_partial_segment = None
        self._latest_partial_request_id = 0
        self.partial_translation_update_count = 0
        self._prefer_deepl_on_next_final = False
        self._final_output_version = 0

    def on_session_stopped(self) -> None:
        self._cancel_partial_debounce()
        logger.info('Speech recognizer session closed.')

    def on_error(self, error: Exception) -> None:
        logger.error('Speech recognizer failed: %s', error)

    async def _translate_partial_task(self, segment: str, request_id: int, finalized_seq: int, final_output_version: int):
        """异步流式翻译任务"""
        self._partial_inflight += 1
        self.translating_partial = True
        success = False
        try:
            if finalized_seq != self._finalized_seq:
                return
            # 检测实际语言（即使手动指定了源语言，也要检测识别结果的真实语言）
            detected_lang_info = language_detector.detect(segment)
            detected_lang = detected_lang_info['language']
            actual_target = resolve_output_target_language(detected_lang, config.TARGET_LANGUAGE)
            requested_secondary_target = _normalize_optional_language_code(
                getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)
            )
            actual_secondary_target = resolve_output_target_language(detected_lang, requested_secondary_target)
            use_secondary_output = actual_secondary_target is not None and secondary_translator is not None

            # 按“发送翻译请求”计数：在真正发起请求前递增
            self.partial_translation_update_count += 1
            
            # 在 executor 中运行同步翻译
            loop = asyncio.get_running_loop()
            primary_future = loop.run_in_executor(
                executor,
                lambda: translator.translate(
                    segment,
                    source_language='auto',
                    target_language=actual_target,
                    context_prefix=config.CONTEXT_PREFIX,
                    is_partial=True,
                    previous_translation=self.last_partial_translation
                )
            )
            secondary_future = None
            if use_secondary_output:
                secondary_future = loop.run_in_executor(
                    executor,
                    lambda: secondary_translator.translate(
                        segment,
                        source_language='auto',
                        target_language=actual_secondary_target,
                        context_prefix=config.CONTEXT_PREFIX,
                        is_partial=True,
                        previous_translation=self.last_partial_translation_secondary
                    )
                )

            translated_text = await primary_future
            secondary_translated_text = await secondary_future if secondary_future is not None else None
            if (
                request_id != self._latest_partial_request_id or
                finalized_seq != self._finalized_seq or
                final_output_version != self._final_output_version
            ):
                return
            success = True
            
            if translated_text and not translated_text.startswith("[ERROR]"):
                self.last_partial_translation = translated_text
            else:
                translated_text = ""

            if secondary_translated_text and not secondary_translated_text.startswith("[ERROR]"):
                self.last_partial_translation_secondary = secondary_translated_text
            else:
                secondary_translated_text = ""

            display_translation = get_display_translation_text(translated_text, actual_target)
            translation_display = build_streaming_output_line(display_translation)
            current_original_display = subtitles_state.get("original", "") or f"{segment.strip()}……"

            if use_secondary_output and actual_secondary_target is not None:
                secondary_display_translation = get_display_translation_text(
                    secondary_translated_text,
                    actual_secondary_target,
                )
                secondary_translation_display = build_streaming_output_line(secondary_display_translation)
                display_text = build_dual_output_display(
                    translation_display,
                    secondary_translation_display,
                    actual_target,
                    actual_secondary_target,
                )
            else:
                show_tag = getattr(config, 'SHOW_ORIGINAL_AND_LANG_TAG', True)
                if show_tag:
                    # 构建显示文本：使用检测到的实际语言
                    source_lang = self._normalize_lang(detected_lang)
                    target_lang = self._normalize_lang(actual_target)
                    display_text = f"[{source_lang}→{target_lang}] {translation_display} ({current_original_display})"
                    # 如果消息过长，尝试去掉原文部分
                    if len(display_text) > 144:
                        display_text = f"[{source_lang}→{target_lang}] {translation_display}"
                else:
                    display_text = translation_display
            
            # 发送 OSC
            await osc_manager.send_text(display_text, ongoing=True)

            current_reverse_trans = subtitles_state.get("reverse_translated", "")

            if use_secondary_output and actual_secondary_target is not None:
                update_subtitles(
                    current_original_display,
                    f"{translation_display}\n{secondary_translation_display}",
                    True,
                    current_reverse_trans,
                )
            else:
                update_subtitles(
                    current_original_display,
                    translation_display,
                    True,
                    current_reverse_trans,
                )
            
        except Exception as e:
            # logger.error(f"流式翻译错误: {e}")
            pass
        finally:
            self._partial_inflight = max(0, self._partial_inflight - 1)
            self.translating_partial = self._partial_inflight > 0
            if (
                request_id == self._latest_partial_request_id and
                finalized_seq == self._finalized_seq and
                final_output_version == self._final_output_version
            ):
                self.pending_partial_segment = None
                if success:
                    self.last_partial_source_segment = segment

    def on_result(self, event: RecognitionEvent) -> None:
        text = event.text
        if not text:
            return

        is_translated = False
        display_text = None
        is_ongoing = not event.is_final

        if is_ongoing:
            print(f'部分：{text}', end='\r')
            display_text = text
            # 无论是否使用流式翻译，都在产生最终结果前保留上一句的翻译在屏幕上
            current_trans = subtitles_state.get("translated", "")
            current_reverse_trans = subtitles_state.get("reverse_translated", "")
            update_subtitles(text, current_trans, True, current_reverse_trans)
            
            # 如果启用了流式翻译且当前 API 支持
            if config.ENABLE_TRANSLATION and getattr(config, 'TRANSLATE_PARTIAL_RESULTS', False):
                segment = self._extract_streaming_segment(text)
                if (
                    self._should_trigger_partial_translation(segment) and
                    segment != self.last_partial_source_segment and
                    segment != self.pending_partial_segment and
                    self.loop
                ):
                    self._partial_request_seq += 1
                    request_id = self._partial_request_seq
                    self._latest_partial_request_id = request_id
                    self.pending_partial_segment = segment
                    final_output_version = self._final_output_version
                    self._schedule_partial_translation_with_debounce(
                        segment,
                        request_id,
                        self._finalized_seq,
                        final_output_version,
                    )

        else:
            # 如果禁用翻译，直接显示识别结果（但仍应用假名/拼音标注）
            if not config.ENABLE_TRANSLATION:
                # 检测语言并应用标注
                source_lang_info = language_detector.detect(text)
                source_lang = source_lang_info['language']
                display_text = add_furigana_if_needed(text, source_lang)
                display_text = add_pinyin_if_needed(display_text, source_lang)
                print(f'识别：{display_text}')
                update_subtitles(display_text, "", is_ongoing, "")
            else:
                # 启用翻译，执行翻译逻辑
                source_lang_info = language_detector.detect(text)
                source_lang = source_lang_info['language']

                normalized_source = self._normalize_lang(source_lang)
                requested_target = config.TARGET_LANGUAGE
                requested_secondary_target = _normalize_optional_language_code(
                    getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)
                )

                actual_target = resolve_output_target_language(source_lang, requested_target)
                actual_secondary_target = resolve_output_target_language(source_lang, requested_secondary_target)

                print(f'原文：{text} [{source_lang_info["language"]}]')
                if actual_target != _normalize_optional_language_code(requested_target):
                    print(f'检测到主输出语言与源语言相同，使用备用语言: {config.FALLBACK_LANGUAGE}')
                if requested_secondary_target and actual_secondary_target != requested_secondary_target:
                    print(f'检测到第二输出语言与源语言相同，使用备用语言: {config.FALLBACK_LANGUAGE}')

                use_deepl_final = False
                max_updates = max(0, int(getattr(config, 'STREAMING_FINAL_DEEPL_MAX_UPDATES', 2)))
                if (
                    self._prefer_deepl_on_next_final and
                    is_streaming_deepl_hybrid_mode() and
                    self.partial_translation_update_count <= max_updates and
                    deepl_fallback_translator is not None
                ):
                    use_deepl_final = True

                use_secondary_output = actual_secondary_target is not None and secondary_translator is not None

                if use_secondary_output:
                    primary_future = executor.submit(
                        translate_with_backend,
                        translator,
                        deepl_fallback_translator,
                        text,
                        actual_target,
                        self.last_partial_translation,
                        use_deepl_final,
                    )
                    secondary_future = executor.submit(
                        translate_with_backend,
                        secondary_translator,
                        secondary_deepl_fallback_translator,
                        text,
                        actual_secondary_target,
                        self.last_partial_translation_secondary,
                        use_deepl_final,
                    )
                    translated_text = primary_future.result()
                    secondary_translated_text = secondary_future.result()
                else:
                    translated_text = translate_with_backend(
                        translator,
                        deepl_fallback_translator,
                        text,
                        actual_target,
                        self.last_partial_translation,
                        use_deepl_final,
                    )
                    secondary_translated_text = None
                self._finalized_seq += 1
                self._final_output_version += 1
                self._cancel_partial_debounce()
                
                # 重置流式翻译状态
                self.last_partial_translation = None
                self.last_partial_translation_secondary = None
                self.last_partial_source_segment = None
                self.pending_partial_segment = None
                self._latest_partial_request_id = 0
                self.partial_translation_update_count = 0
                self._prefer_deepl_on_next_final = False
                
                is_translated = True
                print(f'主目标语言：{actual_target}')
                
                # 为译文添加标注（日语假名/中文拼音）
                display_translated_text = get_display_translation_text(translated_text, actual_target)
                print(f'主译文：{display_translated_text}')

                secondary_display_translated_text = None
                if use_secondary_output and actual_secondary_target is not None:
                    print(f'第二目标语言：{actual_secondary_target}')
                    secondary_display_translated_text = get_display_translation_text(
                        secondary_translated_text,
                        actual_secondary_target,
                    )
                    print(f'第二译文：{secondary_display_translated_text}')
                
                # 为源文本添加标注（日语假名/中文拼音）
                display_source_text = add_furigana_if_needed(text, source_lang)
                display_source_text = add_pinyin_if_needed(display_source_text, source_lang)

                if use_secondary_output and secondary_display_translated_text is not None:
                    display_text = build_dual_output_display(
                        display_translated_text,
                        secondary_display_translated_text,
                        actual_target,
                        actual_secondary_target,
                    )
                else:
                    show_tag = getattr(config, 'SHOW_ORIGINAL_AND_LANG_TAG', True)
                    if show_tag:
                        display_text = f"[{normalized_source}→{actual_target}] {display_translated_text} ({display_source_text})"

                        # 如果消息过长，尝试去掉原文部分
                        if len(display_text) > 144:
                            display_text = f"[{normalized_source}→{actual_target}] {display_translated_text}"
                    else:
                        display_text = str(display_translated_text)


        if display_text is None:
            return

        if is_translated:
            if use_secondary_output and secondary_display_translated_text is not None:
                update_subtitles(
                    display_source_text,
                    f"{display_translated_text}\n{secondary_display_translated_text}",
                    is_ongoing,
                    "",
                )
            else:
                update_subtitles(display_source_text, str(display_translated_text), is_ongoing, "")

        should_send = (not is_ongoing) or should_output_partial_results()

        if self.loop:
            if should_send:
                asyncio.run_coroutine_threadsafe(
                    osc_manager.send_text(display_text, ongoing=is_ongoing),
                    self.loop
                )
            elif is_ongoing:
                asyncio.run_coroutine_threadsafe(
                    osc_manager.set_typing(is_ongoing),
                    self.loop
                )
        else:
            print('[OSC] Warning: Event loop not set, cannot send OSC message.')

        if is_translated and config.ENABLE_REVERSE_TRANSLATION:
            reverse_translated_text = reverse_translation(translated_text, actual_target, normalized_source)
            if reverse_translated_text is not None:
                current_original = subtitles_state.get("original", "")
                current_translated = subtitles_state.get("translated", "")
                current_ongoing = subtitles_state.get("ongoing", False)
                update_subtitles(
                    current_original,
                    current_translated,
                    current_ongoing,
                    str(reverse_translated_text),
                )


async def init_audio_stream():
    """异步初始化音频流"""
    global mic, stream, input_sample_rate, input_block_size, capture_channels, _resampler
    global _debug_pre_audio_recorder, _debug_audio_recorder
    loop = asyncio.get_event_loop()
    
    def _init():
        global mic, stream, input_sample_rate, input_block_size, capture_channels, _resampler
        global _debug_pre_audio_recorder, _debug_audio_recorder
        mic = pyaudio.PyAudio()
        device_index = getattr(config, 'MIC_DEVICE_INDEX', None)
        target_rate = int(config.SAMPLE_RATE)
        target_channels = RECOGNIZER_CHANNELS

        def _get_device_info(idx: Optional[int]) -> Optional[dict]:
            try:
                return mic.get_device_info_by_index(int(idx)) if idx is not None else mic.get_default_input_device_info()
            except Exception:
                return None

        def _resolve_capture_channels(idx: Optional[int]) -> int:
            requested_channels = target_channels
            info = _get_device_info(idx)
            if not info:
                return requested_channels

            max_in = int(info.get('maxInputChannels', 0) or 0)
            device_name = str(info.get('name', '') or '')
            normalized_name = device_name.lower()
            is_virtual_mixer = 'voicemeeter' in normalized_name or 'vb-audio' in normalized_name

            if requested_channels == 1 and is_virtual_mixer and max_in >= 2:
                print(f'[Audio] 检测到虚拟混音输入 {device_name}，将按 2 声道采集后在程序内转换为单声道')
                return 2

            if max_in > 0:
                return max(1, min(requested_channels, max_in))
            return requested_channels

        def _open_with(rate: int, frames_per_buffer: int, idx: Optional[int], channels: int):
            kwargs = dict(
                format=pyaudio.paInt16,
                channels=int(channels),
                rate=int(rate),
                input=True,
                frames_per_buffer=int(frames_per_buffer),
            )
            if idx is not None:
                kwargs['input_device_index'] = int(idx)
            return mic.open(**kwargs)

        def _get_device_default_rate(idx: Optional[int]) -> Optional[int]:
            info = _get_device_info(idx)
            if info:
                r = info.get('defaultSampleRate')
                if r:
                    return int(round(float(r)))
            return None

        def _init_resampler(in_rate: int, out_rate: int):
            """创建或重置重采样器（采样率相同时置 None）。"""
            global _resampler
            if in_rate != out_rate:
                _resampler = AudioResampler(
                    input_rate=in_rate,
                    output_rate=out_rate,
                    channels=target_channels,
                    sample_width=2,
                )
            else:
                _resampler = None

        def _init_debug_audio_recorders(in_rate: int, out_rate: int):
            """初始化调试音频录制器。"""
            global _debug_pre_audio_recorder, _debug_audio_recorder

            if _debug_pre_audio_recorder is not None:
                _debug_pre_audio_recorder.close()
                _debug_pre_audio_recorder = None

            if _debug_audio_recorder is not None:
                _debug_audio_recorder.close()
                _debug_audio_recorder = None

            output_dir = getattr(config, 'DEBUG_AUDIO_OUTPUT_DIR', 'debug_audio')

            if getattr(config, 'SAVE_PRE_RESAMPLE_AUDIO', False):
                try:
                    _debug_pre_audio_recorder = WaveDebugRecorder(
                        output_dir=output_dir,
                        input_rate=in_rate,
                        sample_rate=in_rate,
                        channels=int(capture_channels),
                        sample_width=2,
                        file_prefix='pre_resample',
                    )
                    print(f'[Audio] 正在录制重采样前的音频: {_debug_pre_audio_recorder.file_path}')
                except Exception as e:
                    _debug_pre_audio_recorder = None
                    print(f'[Audio] 无法初始化重采样前调试录音: {e}')

            if getattr(config, 'SAVE_POST_RESAMPLE_AUDIO', False):
                try:
                    _debug_audio_recorder = WaveDebugRecorder(
                        output_dir=output_dir,
                        input_rate=in_rate,
                        sample_rate=out_rate,
                        channels=target_channels,
                        sample_width=2,
                        file_prefix='post_resample',
                    )
                    print(f'[Audio] 正在录制重采样后的音频: {_debug_audio_recorder.file_path}')
                except Exception as e:
                    _debug_audio_recorder = None
                    print(f'[Audio] 无法初始化重采样后调试录音: {e}')

        # 先尝试按 16k（ASR 期望）打开；失败则按设备默认采样率打开并重采样
        try:
            input_sample_rate = target_rate
            input_block_size = int(config.BLOCK_SIZE)
            capture_channels = _resolve_capture_channels(int(device_index) if device_index is not None else None)
            _init_resampler(target_rate, target_rate)
            stream = _open_with(
                target_rate,
                input_block_size,
                int(device_index) if device_index is not None else None,
                capture_channels,
            )
        except Exception as e_open_16k:
            device_rate = _get_device_default_rate(int(device_index) if device_index is not None else None)

            if device_rate is not None and device_rate > 0 and device_rate != target_rate:
                # 保持时间窗口大致一致：按采集采样率缩放 block size
                scaled_block = int(round(config.BLOCK_SIZE * (device_rate / target_rate)))
                scaled_block = max(256, scaled_block)
                try:
                    input_sample_rate = int(device_rate)
                    input_block_size = int(scaled_block)
                    capture_channels = _resolve_capture_channels(int(device_index) if device_index is not None else None)
                    _init_resampler(input_sample_rate, target_rate)
                    stream = _open_with(
                        input_sample_rate,
                        input_block_size,
                        int(device_index) if device_index is not None else None,
                        capture_channels,
                    )
                    print(f"[Audio] 设备不支持 {target_rate}Hz，已使用 {input_sample_rate}Hz / {capture_channels}ch 采集并实时重采样")
                except Exception as e_open_device_rate:
                    # 最后回退系统默认
                    try:
                        input_sample_rate = target_rate
                        input_block_size = int(config.BLOCK_SIZE)
                        capture_channels = _resolve_capture_channels(None)
                        _init_resampler(target_rate, target_rate)
                        stream = _open_with(target_rate, input_block_size, None, capture_channels)
                        print(f"[Audio] 指定麦克风设备不可用，已回退到系统默认：{e_open_device_rate}")
                    except Exception:
                        raise
            else:
                # 没拿到设备默认采样率：直接回退系统默认
                try:
                    input_sample_rate = target_rate
                    input_block_size = int(config.BLOCK_SIZE)
                    capture_channels = _resolve_capture_channels(None)
                    _init_resampler(target_rate, target_rate)
                    stream = _open_with(target_rate, input_block_size, None, capture_channels)
                    print(f"[Audio] 指定麦克风设备不可用，已回退到系统默认：{e_open_16k}")
                except Exception:
                    raise

        print(f'[Audio] 实际采集格式: {input_sample_rate}Hz / {capture_channels}ch / 16-bit')
        if capture_channels != target_channels:
            print(f'[Audio] 发送给识别器前将转换为: {target_rate}Hz / {target_channels}ch / 16-bit')

        _init_debug_audio_recorders(input_sample_rate, target_rate)
        return stream
    
    return await loop.run_in_executor(executor, _init)


async def close_audio_stream():
    """异步关闭音频流"""
    global mic, stream, _debug_pre_audio_recorder, _debug_audio_recorder
    loop = asyncio.get_event_loop()
    
    def _close():
        global mic, stream, _debug_pre_audio_recorder, _debug_audio_recorder
        if stream:
            stream.stop_stream()
            stream.close()
        if mic:
            mic.terminate()
        if _debug_pre_audio_recorder:
            saved_file = _debug_pre_audio_recorder.file_path
            _debug_pre_audio_recorder.close()
            print(f'[Audio] 重采样前的音频已保存到: {saved_file}')
            _debug_pre_audio_recorder = None
        if _debug_audio_recorder:
            saved_file = _debug_audio_recorder.file_path
            _debug_audio_recorder.close()
            print(f'[Audio] 重采样后的音频已保存到: {saved_file}')
            _debug_audio_recorder = None
        stream = None
        mic = None
    
    await loop.run_in_executor(executor, _close)


async def read_audio_data():
    """异步读取音频数据"""
    global stream, _resampler, capture_channels, _debug_pre_audio_recorder, _debug_audio_recorder
    if not stream:
        return None
    
    loop = asyncio.get_event_loop()
    
    def _read():
        try:
            data = stream.read(input_block_size, exception_on_overflow=False)
            if not data:
                return None
            if _debug_pre_audio_recorder is not None:
                _debug_pre_audio_recorder.write(data)

            capture_data = data
            if capture_channels != RECOGNIZER_CHANNELS:
                samples = np.frombuffer(data, dtype=np.int16)
                num_frames = len(samples) // int(capture_channels)
                if num_frames <= 0:
                    return b''
                frames = samples[: num_frames * int(capture_channels)].reshape(num_frames, int(capture_channels)).astype(np.int32)
                mono = np.rint(np.mean(frames, axis=1))
                mono = np.clip(mono, -32768, 32767).astype(np.int16)
                capture_data = mono.tobytes()

            processed_data = _resampler.resample(capture_data) if _resampler is not None else capture_data
            if _debug_audio_recorder is not None and processed_data:
                _debug_audio_recorder.write(processed_data)
            return processed_data
        except Exception as e:
            print(f'Error reading audio data: {e}')
            return None
    
    return await loop.run_in_executor(executor, _read)


async def send_audio_frame_async(recognizer: SpeechRecognizer, data: bytes):
    """异步发送音频帧"""
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(executor, recognizer.send_audio_frame, data)
    except Exception as e:
        pass
    

async def audio_capture_task(recognizer: SpeechRecognizer):
    """异步音频捕获任务"""
    global recognition_active
    print('Starting audio capture...')
    try:
        while not stop_event.is_set():
            # 始终读取音频数据,避免缓冲区积压
            data = await read_audio_data()
            if data is None:
                break
            if not data:
                await asyncio.sleep(0.001)
                continue
            
            # 只有在识别激活时才发送音频数据,否则丢弃
            if recognition_active:
                await send_audio_frame_async(recognizer, data)
            # 静音时数据被读取但不发送,自动丢弃
            
            await asyncio.sleep(0.001)  # 避免阻塞事件循环
    except asyncio.CancelledError:
        print('Audio capture task cancelled.')
    except Exception as e:
        print(f'Audio capture error: {e}')
    finally:
        print('Audio capture stopped.')


def signal_handler(sig, frame):
    print('Ctrl+C pressed, stop recognition ...')
    # 在异步环境中安全地设置停止事件
    if stop_event is not None:
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(stop_event.set)
        except:
            stop_event.set()


async def stop_recognition_async(recognizer: SpeechRecognizer):
    """异步暂停或停止识别服务"""
    global recognition_active, recognition_started
    if not recognition_active:
        return  # 已经暂停

    loop = asyncio.get_event_loop()

    recognition_active = False

    # 统一由各后端在 pause 内部决定：
    # - 真正 pause（保持连接）
    # - 或执行 finalize/stop（并尽快返回本段最终结果）
    try:
        await loop.run_in_executor(executor, recognizer.pause)
    except Exception:
        pass


async def start_recognition_async(recognizer: SpeechRecognizer):
    """异步开始或恢复识别服务"""
    global recognition_active, recognition_started
    if recognition_active:
        print('Recognition already active.')
        return  # 已经在运行中

    loop = asyncio.get_event_loop()

    try:
        # 支持 pause/resume 的后端直接恢复，其他后端需要完整 start
        if CURRENT_ASR_BACKEND in PAUSE_RESUME_BACKENDS and recognition_started:
            await loop.run_in_executor(executor, recognizer.resume)
        else:
            await loop.run_in_executor(executor, recognizer.start)
            recognition_started = True
    except Exception:
        pass

    recognition_active = True


async def handle_mute_change(is_muted):
    """
    处理静音状态变化的回调函数
    
    Args:
        is_muted: True表示静音(停止识别), False表示取消静音(开始识别)
    """
    global recognition_active, recognition_instance, mute_delay_task, recognition_started, recognition_callback
    
    # 如果禁用了麦克风控制，则忽略所有麦克风状态变化
    if not is_effective_mic_control_enabled():
        return
    
    if recognition_instance is None:
        print('[ASR] 识别实例未初始化')
        return
    
    stop_word = '暂停' if CURRENT_ASR_BACKEND in PAUSE_RESUME_BACKENDS else '停止'
    start_word = '恢复' if CURRENT_ASR_BACKEND in PAUSE_RESUME_BACKENDS and recognition_started else '开始'

    if is_muted:
        # 静音状态 - 延迟停止识别
        if recognition_active:
            if recognition_callback is not None:
                recognition_callback.mark_mute_finalization_requested()
            # 如果已有延迟任务在运行，先取消它
            if mute_delay_task and not mute_delay_task.done():
                mute_delay_task.cancel()
            
            if config.MUTE_DELAY_SECONDS > 0:
                print(f'[ASR] 检测到静音，将在 {config.MUTE_DELAY_SECONDS} 秒后{stop_word}语音识别...')
                
                async def delayed_stop():
                    global recognition_active
                    try:
                        await asyncio.sleep(config.MUTE_DELAY_SECONDS)
                        if recognition_active:  # 再次检查，确保期间没有取消静音
                            print(f'[ASR] 延迟时间到，{stop_word}语音识别')
                            await stop_recognition_async(recognition_instance)
                            logger.info(f'[ASR] 语音识别已{stop_word}')
                    except asyncio.CancelledError:
                        print('[ASR] 停止识别已取消（取消静音）')
                
                mute_delay_task = asyncio.create_task(delayed_stop())
            else:
                # 延迟为0，立即停止
                print(f'[ASR] 检测到静音，立即{stop_word}语音识别...')
                await stop_recognition_async(recognition_instance)
                logger.info(f'[ASR] 语音识别已{stop_word}')
    else:
        # 取消静音 - 开始识别
        if recognition_callback is not None:
            recognition_callback.clear_mute_finalization_requested()
        # 如果有延迟停止任务，取消它
        if mute_delay_task and not mute_delay_task.done():
            mute_delay_task.cancel()
            print('[ASR] 检测到取消静音，已取消延迟停止任务')
        
        if not recognition_active:
            print(f'[ASR] 检测到取消静音，{start_word}语音识别...')
            await start_recognition_async(recognition_instance)
            logger.info(f'[ASR] 语音识别已{start_word}')


def handle_mute_change_sync(is_muted):
    """同步桥接：将OSC线程中的静音事件安全投递到主事件循环。"""
    loop = main_loop
    if loop is None or not loop.is_running():
        logger.warning('[ASR] 主事件循环不可用，忽略静音状态变化')
        return

    try:
        asyncio.run_coroutine_threadsafe(handle_mute_change(is_muted), loop)
    except Exception as e:
        logger.error(f'[ASR] 投递静音状态变化失败: {e}')


async def main(keep_oscquery_alive: bool = False):
    """主异步函数"""
    global recognition_instance, recognition_active, vocabulary_id, CURRENT_ASR_BACKEND, recognition_started, executor, stop_event, main_loop, recognition_callback

    # 启动时清空翻译屏幕残留
    update_subtitles("", "", False)

    main_loop = asyncio.get_running_loop()
    
    # 创建当前事件循环的 stop_event
    stop_event = asyncio.Event()
    
    # 重新创建executor（如果已经shutdown）
    if executor._shutdown:
        executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
    
    vocabulary_id = None
    corpus_text: Optional[str] = None

    # 检测并应用系统代理设置
    system_proxies = detect_system_proxy()
    print_proxy_info(system_proxies)
    
    # 初始化 DashScope API Key
    init_dashscope_api_key()
    print('Initializing ...')

    # 选择可用的识别后端
    backend = select_backend(config.PREFERRED_ASR_BACKEND, config.VALID_ASR_BACKENDS)
    if backend != config.PREFERRED_ASR_BACKEND:
        print(f'[ASR] 已切换语音识别后端为 {backend}')
    else:
        print(f'[ASR] 目标识别后端: {backend}')

    CURRENT_ASR_BACKEND = backend
    recognition_active = False
    recognition_started = False

    # 初始化热词（仅 qwen / dashscope）
    if config.ENABLE_HOT_WORDS and backend in {'qwen', 'dashscope'}:
        print('\n[热词] 初始化热词资源...')
        try:
            hot_words_manager = HotWordsManager()
            hot_words_manager.load_all_hot_words()
            if backend == 'qwen':
                words = [entry.get('text') for entry in hot_words_manager.get_hot_words() if entry.get('text')]
                if words:
                    corpus_text = "\n".join(words)
                    print(f'[热词] 已生成 Qwen 语料文本，共 {len(words)} 条\n')
                else:
                    print('[热词] 未加载到热词条目，跳过 Qwen 语料配置\n')
            else:
                vocabulary_id = hot_words_manager.create_vocabulary(target_model='fun-asr-realtime')
                print(f'[热词] 热词表创建成功，ID: {vocabulary_id}\n')
        except Exception as e:
            print(f'[热词] 热词初始化失败: {e}')
            print('[热词] 将继续运行但不使用热词\n')
            vocabulary_id = None
            corpus_text = None

    # 启动OSC服务器
    print('[OSC] 启动OSC服务器...')
    await osc_manager.start_server(app_name="Yakutan")
    
    # 设置静音状态回调
    osc_manager.set_mute_callback(handle_mute_change_sync)
    print('[OSC] 已设置静音状态回调')

    # 创建识别回调
    callback = VRChatRecognitionCallback()
    callback.loop = asyncio.get_event_loop()
    recognition_callback = callback

    # 使用工厂创建识别实例
    recognition_instance = create_recognizer(
        backend=backend,
        callback=callback,
        sample_rate=config.SAMPLE_RATE,
        audio_format=config.FORMAT_PCM,
        source_language=config.SOURCE_LANGUAGE,
        vocabulary_id=vocabulary_id,
        corpus_text=corpus_text,
        enable_vad=config.ENABLE_VAD,
        vad_threshold=config.VAD_THRESHOLD,
        vad_silence_duration_ms=config.VAD_SILENCE_DURATION_MS,
        keepalive_interval=config.KEEPALIVE_INTERVAL,
    )
    
    if vocabulary_id and backend == 'dashscope':
        print(f'[ASR] 使用热词表: {vocabulary_id}')
    
    if backend == 'qwen':
        vad_status = '启用' if config.ENABLE_VAD else '禁用'
        print(f'[ASR] VAD状态: {vad_status}')
        if config.ENABLE_VAD:
            print(f'[ASR] VAD配置: 阈值={config.VAD_THRESHOLD}, 静音时长={config.VAD_SILENCE_DURATION_MS}ms')
        
        if config.KEEPALIVE_INTERVAL > 0:
            print(f'[ASR] WebSocket心跳已启用: 间隔={config.KEEPALIVE_INTERVAL}秒')
        else:
            print('[ASR] WebSocket心跳已禁用')
    
    print('[ASR] 识别实例已创建')
    
    # 初始化音频流
    await init_audio_stream()

    # 只在主线程中设置信号处理器
    try:
        signal.signal(signal.SIGINT, signal_handler)  #@IgnoreException
    except ValueError:
        # 在非主线程中运行时，signal.signal会抛出ValueError
        # 这种情况下由Web UI的stop接口处理停止逻辑
        pass
    
    # 根据配置决定是否立即启动识别
    effective_mic_control = is_effective_mic_control_enabled()

    if effective_mic_control:
        if backend == 'doubao_file' and not config.ENABLE_MIC_CONTROL:
            print('[模式] 豆包文件转录已强制启用“游戏静音时暂停转录”（仅运行时生效）')
        stop_hint = '暂停' if backend in PAUSE_RESUME_BACKENDS else '停止'
        resume_hint = '恢复' if backend in PAUSE_RESUME_BACKENDS else '开始'
        print("=" * 60)
        print("[模式] 麦克风控制模式已启用")
        print("等待VRChat静音状态变化...")
        print(f"取消静音(MuteSelf=False)将{resume_hint}语音识别")
        print(f"启用静音(MuteSelf=True)将{stop_hint}语音识别")
        print("按 'Ctrl+C' 退出程序")
        print("=" * 60)
    else:
        print("=" * 60)
        print("[模式] 麦克风控制模式已禁用")
        print("语音识别将立即启动，忽略麦克风开关状态")
        print("按 'Ctrl+C' 退出程序")
        print("=" * 60)
        # 立即启动识别
        await start_recognition_async(recognition_instance)
        print('[ASR] 语音识别已启动')

    # 创建音频捕获任务
    capture_task = asyncio.create_task(audio_capture_task(recognition_instance))
    
    try:
        # 等待停止事件
        await stop_event.wait()
        
        # 取消捕获任务
        capture_task.cancel()
        
        # 等待捕获任务完成(带超时)
        try:
            await asyncio.wait_for(capture_task, timeout=2.0)
        except asyncio.TimeoutError:
            print('Audio capture task timeout, forcing stop.')
        except asyncio.CancelledError:
            pass
        
        # 如果识别正在运行,停止它
        if recognition_active:
            await stop_recognition_async(recognition_instance)
            halt_word = 'paused' if CURRENT_ASR_BACKEND in PAUSE_RESUME_BACKENDS else 'stopped'
            print(f'Recognition {halt_word}.')
        
        # 获取统计信息(使用异步方式)
        if recognition_instance:
            loop = asyncio.get_event_loop()
            try:
                request_id = await loop.run_in_executor(executor, recognition_instance.get_last_request_id)
                first_delay = await loop.run_in_executor(executor, recognition_instance.get_first_package_delay)
                last_delay = await loop.run_in_executor(executor, recognition_instance.get_last_package_delay)
                
                print(
                    '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
                    .format(request_id, first_delay, last_delay))
            except Exception as e:
                print(f'[Metric] 获取统计信息失败: {e}')
    
    finally:
        # 清除OSC回调
        osc_manager.clear_mute_callback()
        osc_manager.reset_runtime_state()

        loop = asyncio.get_event_loop()

        if recognition_instance:
            try:
                await loop.run_in_executor(executor, recognition_instance.stop)
            except Exception:
                pass
            recognition_started = False
            recognition_active = False
        
        # 关闭音频流
        await close_audio_stream()
        
        # 仅在非复用模式下停止OSCQuery服务
        if not keep_oscquery_alive:
            await osc_manager.stop_server()
        
        # 异步关闭线程池
        await loop.run_in_executor(None, executor.shutdown, False)


# main function
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nProgram terminated by user.')
    # except Exception as e:
    #     print(f'Error: {e}')
    finally:
        print('Cleanup completed.')