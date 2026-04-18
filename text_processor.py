"""
文本处理模块 - 负责假名标注、拼音标注、双语裁剪和显示格式化
"""
from typing import Optional

import config

# ============ 双语输出常量 ============
DUAL_OUTPUT_SEPARATOR = "\n"

# 双语裁剪时：中日韩权重=1，其他语言权重=2。
COMPACT_SCRIPT_LANGUAGE_BASES = {'zh', 'ja', 'ko'}
COMPACT_SCRIPT_BUDGET_WEIGHT = 1
ALPHABETIC_SCRIPT_BUDGET_WEIGHT = 2


def _get_dual_output_limits() -> tuple[Optional[int], Optional[int], Optional[int]]:
    total_max_chars = config.get_effective_osc_text_max_length()
    if total_max_chars is None:
        return None, None, None

    body_budget = max(0, total_max_chars - len(DUAL_OUTPUT_SEPARATOR))
    max_chars_per_result = body_budget // 2
    return total_max_chars, body_budget, max_chars_per_result


# ============ 语言代码工具函数 ============

def normalize_optional_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip()
    return normalized or None


def normalize_lang_code(lang):
    """标准化语言代码。

    中文区分简体 (zh-hans) 与繁体 (zh-hant)，以便在目标语为 zh-CN / zh-TW 等时
    仍能触发简繁互译；检测器返回的泛化 ``zh`` 视为简体（与 zh-CN 一致）。
    """
    if not lang:
        return 'auto'
    lang_norm = str(lang).strip().lower().replace('_', '-')
    if lang_norm == 'auto':
        return 'auto'
    if lang_norm in ('en', 'en-us', 'en-gb', 'en-au', 'en-ca', 'en-nz', 'en-ie'):
        return 'en'
    # 繁体中文（含港澳地区常用码）
    if lang_norm in ('zh-tw', 'zh-hant', 'zh-hk', 'zh-mo'):
        return 'zh-hant'
    # 简体中文 + 无简繁信息的「zh」（检测器/ASR 常见返回值）
    if lang_norm in (
        'zh',
        'zh-cn',
        'zh-hans',
        'zh-sg',
        'cmn',
        'wuu',
        'yue',
    ):
        return 'zh-hans'
    return lang_norm


def language_code_for_osc_tag(lang: Optional[str]) -> str:
    """OSC 等场景的语言对标记：只保留主语言码，不区分地区/简繁/script 变体。"""
    if lang is None:
        return 'auto'
    norm = str(lang).strip().lower().replace('_', '-')
    if not norm:
        return 'auto'
    if norm == 'auto':
        return 'auto'
    if norm in (
        'zh',
        'zh-cn',
        'zh-hans',
        'zh-tw',
        'zh-hant',
        'zh-hk',
        'zh-mo',
        'zh-sg',
        'cmn',
        'wuu',
        'yue',
    ):
        return 'zh'
    return norm.split('-', 1)[0]


def has_secondary_translation_target() -> bool:
    return normalize_optional_language_code(
        getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)
    ) is not None


def resolve_output_target_language(
    source_language: str,
    requested_target_language: Optional[str],
) -> Optional[str]:
    target_language = normalize_optional_language_code(requested_target_language)
    if target_language is None:
        return None

    fallback_language = normalize_optional_language_code(
        getattr(config, 'FALLBACK_LANGUAGE', None)
    )
    if fallback_language and normalize_lang_code(source_language) == normalize_lang_code(target_language):
        return fallback_language

    return target_language


# ============ 输出格式化 ============

def _sanitize_output_line(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    return " ".join(part.strip() for part in normalized.split('\n') if part.strip())


def remove_trailing_sentence_period_if_needed(text: str) -> str:
    """Optionally remove a single trailing sentence-final period."""
    sanitized = _sanitize_output_line(text)
    if not sanitized or not getattr(config, 'REMOVE_TRAILING_PERIOD', False):
        return sanitized

    trimmed = sanitized.rstrip()
    if trimmed.endswith(("。", ".", "．")):
        return trimmed[:-1].rstrip()
    return trimmed


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
    total_chars: Optional[int] = None,
) -> tuple[int, int]:
    if total_chars is None:
        _, total_chars, _ = _get_dual_output_limits()
    if total_chars is None:
        return 0, 0
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


def limit_dual_output_text(
    text: str,
    max_chars: Optional[int] = None,
) -> str:
    sanitized = _sanitize_output_line(text)
    if max_chars is None:
        _, _, max_chars = _get_dual_output_limits()
    if max_chars is None:
        return sanitized
    if len(sanitized) <= max_chars:
        return sanitized
    return sanitized[:max_chars].rstrip()


def build_dual_output_display(
    primary_text: str,
    secondary_text: Optional[str],
    primary_language: Optional[str] = None,
    secondary_language: Optional[str] = None,
) -> str:
    total_max_chars, body_budget, _ = _get_dual_output_limits()
    if secondary_text is None:
        return limit_dual_output_text(primary_text)

    primary_sanitized = _sanitize_output_line(primary_text)
    secondary_sanitized = _sanitize_output_line(secondary_text)
    full_text = DUAL_OUTPUT_SEPARATOR.join([primary_sanitized, secondary_sanitized])

    if total_max_chars is None or body_budget is None:
        return full_text

    # 能完整装下时不做任何裁剪。
    if len(full_text) <= total_max_chars:
        return full_text

    primary_budget, secondary_budget = _allocate_dual_output_budgets(
        primary_language,
        secondary_language,
        total_chars=body_budget,
    )

    clipped_primary = limit_dual_output_text(primary_sanitized, max_chars=primary_budget)
    clipped_secondary = limit_dual_output_text(secondary_sanitized, max_chars=secondary_budget)
    return DUAL_OUTPUT_SEPARATOR.join([clipped_primary, clipped_secondary])


def build_streaming_output_line(text: str) -> str:
    formatted_text = remove_trailing_sentence_period_if_needed(text)
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
    """Add hiragana readings to Japanese text with kanji."""
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
    """
    if not text or not _pypinyin_available:
        return text

    if not _contains_chinese(text):
        return text

    try:
        import jieba

        full_pinyin = pinyin(text, style=Style.TONE)

        char_to_pinyin = {}
        for i, char in enumerate(text):
            if i < len(full_pinyin):
                py = full_pinyin[i][0]
                if _contains_chinese(char) and py != char:
                    char_to_pinyin[i] = py

        words = list(jieba.cut(text))

        result_parts = []
        char_index = 0

        for word in words:
            if _contains_chinese(word):
                word_pinyins = []
                for char in word:
                    if char_index in char_to_pinyin:
                        word_pinyins.append(char_to_pinyin[char_index])
                    char_index += 1

                if word_pinyins:
                    py_str = "'".join(word_pinyins)
                    result_parts.append(f"{word}{py_str}")
                else:
                    result_parts.append(word)
            else:
                result_parts.append(word)
                char_index += len(word)

        return "".join(result_parts)
    except ImportError:
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


def get_display_translation_text(translated_text: str, target_language: str) -> str:
    """为翻译结果添加假名/拼音标注。"""
    display_text = add_furigana_if_needed(translated_text, target_language)
    display_text = add_pinyin_if_needed(display_text, target_language)
    return remove_trailing_sentence_period_if_needed(display_text)
