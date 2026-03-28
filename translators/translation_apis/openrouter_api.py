"""
LLM translation API implementation using an OpenAI-compatible interface.
支持普通翻译和流式翻译两种模式

v12 smart-hybrid 策略：
- Partial: 传 draft，使用 continuation framing 保持稳定
- Final: 先用 continuation framing（传 draft）获取稳定翻译，
  然后检查内容完整性。若检测到内容丢失，则用无 draft 的 fresh
  翻译做补救，再通过 merge_with_draft 合并保留稳定前缀。
"""

import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from typing import Optional, List, Dict

from .base_translation_api import BaseTranslationAPI
import config
from openai_compat_client import OpenAICompatClientBase


def merge_with_draft(fresh_translation: str, draft: str) -> str:
    """代码级合并：保留 draft 的开头措辞，用 fresh 保证完整性。

    核心策略：
    - 用户最关注"文本不要跳动" → 开头保持一致最重要
    - 内容不能丢 → fresh 翻译保证完整性
    - 方法：找到 draft 和 fresh 的最佳分割点，之前用 draft 措辞，之后用 fresh 内容
    """
    if not draft or not fresh_translation:
        return fresh_translation

    # Case 1: fresh 已经以 draft 为前缀 → 完美
    if fresh_translation.startswith(draft):
        return fresh_translation

    # Case 2: draft 是 fresh 的前缀或超集 → 用 fresh
    if draft.startswith(fresh_translation):
        return fresh_translation

    # Case 3: 找公共前缀
    common_prefix_len = 0
    for a, b in zip(draft, fresh_translation):
        if a == b:
            common_prefix_len += 1
        else:
            break

    # 如果公共前缀 >= 40% of draft 或 >= 3 chars，在该点拼接
    if common_prefix_len >= max(len(draft) * 0.4, 3):
        return draft[:common_prefix_len] + fresh_translation[common_prefix_len:]

    # Case 4: 两者差异太大，直接用 fresh（保质量优先）
    return fresh_translation


class OpenRouterAPI(OpenAICompatClientBase, BaseTranslationAPI):
    """
    Generic LLM translation API based on an OpenAI-compatible endpoint.
    Supports both standard and streaming-optimized translation modes.
    """

    SUPPORTS_CONTEXT = True

    LANGUAGE_NAME_MAP = {
        "zh": "SIMPLIFIED CHINESE",
        "zh-cn": "SIMPLIFIED CHINESE",
        "zh-tw": "TRADITIONAL CHINESE",
        "zh-hans": "SIMPLIFIED CHINESE",
        "zh-hant": "TRADITIONAL CHINESE",
        "zh-hk": "TRADITIONAL CHINESE (Hong Kong)",
        "zh-mo": "TRADITIONAL CHINESE (Macau)",
        "zh-sg": "SIMPLIFIED CHINESE (Singapore)",
        "en": "ENGLISH",
        "en-us": "AMERICAN ENGLISH",
        "en-gb": "BRITISH ENGLISH",
        "ja": "JAPANESE",
        "ko": "KOREAN",
        "es": "SPANISH",
        "fr": "FRENCH",
        "de": "GERMAN",
        "id": "INDONESIAN",
        "ru": "RUSSIAN",
        "ar": "ARABIC",
        "pt": "PORTUGUESE",
        "th": "THAI",
        "tl": "TAGALOG (Philippines)",
        "it": "ITALIAN",
        "tr": "TURKISH",
        "fil": "FILIPINO/TAGALOG",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 30,
        max_retries: int = 3,
        streaming_mode: bool = False,
    ) -> None:
        """
        初始化 LLM 翻译客户端

        Args:
            model: 使用的模型名称
            temperature: 采样温度
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            streaming_mode: 是否使用流式翻译模式（支持 previous_translation 和 is_partial）
        """
        self.model = model or config.LLM_MODEL
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.streaming_mode = streaming_mode

        super().__init__(base_url=config.LLM_BASE_URL, model=self.model)

    @classmethod
    def _describe_language(cls, language_code: str) -> str:
        normalized = (language_code or "").strip().lower()
        if not normalized or normalized == "auto":
            return "auto-detected source language"
        language_name = cls.LANGUAGE_NAME_MAP.get(normalized, normalized.upper())
        return f"{language_name} ({language_code})"

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        context_pairs: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """
        翻译文本

        Args:
            text: 要翻译的文本
            source_language: 源语言代码
            target_language: 目标语言代码
            context: 可选的上下文信息（仅原文）
            context_pairs: 可选的上下文对列表，包含 source 和 target
            **kwargs: 其他参数
                previous_translation: 上一次的翻译结果 (str) - 仅 streaming_mode
                is_partial: 是否为部分结果 (bool) - 仅 streaming_mode

        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return ""

        if self.streaming_mode:
            return self._translate_streaming(
                text, source_language, target_language, context, context_pairs, **kwargs
            )
        return self._translate_standard(
            text, source_language, target_language, context, context_pairs, **kwargs
        )

    def _translate_standard(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
        **kwargs,
    ) -> str:
        """标准翻译模式"""
        target_descriptor = self._describe_language(target_language)
        system_prompt = self._build_system_prompt(target_descriptor)
        context_block = self._build_context_block(context, context_pairs)

        user_parts = []
        if context_block:
            user_parts.append(context_block)
        user_parts.append(f"Translate this: {text}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        return self._call_api(messages, is_partial=False)

    def _build_system_prompt(self, target_descriptor: str) -> str:
        """构建系统提示词（所有模式通用）"""
        return (
            f"You are a VRChat voice chat translator. "
            f"Translate the user's message into {target_descriptor}.\n\n"
            f"- Output ONLY in {target_descriptor}. No source-language words.\n"
            "- Translate EVERY part completely. Never skip or shorten.\n"
            "- Use casual spoken style, like friends chatting.\n"
            "- For idioms/slang: translate the meaning naturally.\n"
            "- Output the translation only. No labels, notes, or commentary."
        )

    def _build_context_block(
        self,
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
    ) -> Optional[str]:
        """构建上下文块"""
        if context_pairs:
            ctx = "Conversation so far:\n"
            for pair in context_pairs:
                ctx += f"  {pair['source']} → {pair['target']}\n"
            return ctx.strip()
        if context and context.strip():
            return f"Conversation so far:\n{context.strip()}"
        return None

    def _translate_streaming(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
        **kwargs,
    ) -> str:
        """流式翻译模式 v12: smart-hybrid

        策略：
        - Partial: 传 draft + continuation framing → 稳定的中间翻译
        - Final: 先用 continuation framing（传 draft）→ 稳定翻译
          → 检查内容完整性 → 若丢了内容则补救（fresh + merge）
        """
        previous_translation = kwargs.get("previous_translation")
        is_partial = kwargs.get("is_partial", False)
        previous_source_text = kwargs.get("previous_source_text")
        detected_source_language = kwargs.get("detected_source_language", "auto")
        target_descriptor = self._describe_language(target_language)

        system_prompt = self._build_system_prompt(target_descriptor)
        context_block = self._build_context_block(context, context_pairs)

        # ── Step 1: 构建 v11 continuation prompt（同时适用于 partial 和 final）──
        user_parts = []
        if context_block:
            user_parts.append(context_block)

        if previous_translation:
            if is_partial:
                user_parts.append(
                    f"Your previous translation: {previous_translation.strip()}\n"
                    "Source text has been updated below. Translate the full updated text. "
                    "Keep wording consistent where meaning hasn't changed."
                )
            else:
                # Final: continuation framing — 强调 "继续" 而非 "修改"
                user_parts.append(
                    f"You previously translated part of this as: {previous_translation.strip()}\n"
                    "Now the complete sentence has arrived. "
                    "Translate the COMPLETE source text below. "
                    "Start your translation the same way as your previous version, "
                    "then continue translating the rest of the sentence."
                )

        user_parts.append(f"Translate this: {text}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

        stable_translation = self._call_api(messages, is_partial=is_partial)

        # ── Partial 直接返回 ──
        if is_partial:
            return stable_translation

        # ── Step 2: Final — 内容完整性检查 ──
        if stable_translation.startswith("[ERROR]") or not previous_translation:
            return stable_translation

        needs_rescue, rescue_reason = self._check_content_completeness(
            text, stable_translation, previous_translation,
            previous_source_text=previous_source_text,
            detected_source_language=detected_source_language,
            target_language=target_language,
        )

        if not needs_rescue:
            return stable_translation

        # ── Step 3: 补救 — 无 draft 的 fresh 翻译 ──
        print(f"[LLM] v12 rescue triggered: {rescue_reason}")
        rescue_parts = []
        if context_block:
            rescue_parts.append(context_block)
        rescue_parts.append(f"Translate this: {text}")

        rescue_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(rescue_parts)},
        ]

        fresh_translation = self._call_api(rescue_messages, is_partial=False)

        if fresh_translation.startswith("[ERROR]"):
            return stable_translation

        if len(fresh_translation) <= len(stable_translation):
            return stable_translation

        # ── Step 4: 合并 — 保留 stable 的开头 + fresh 的完整内容 ──
        return merge_with_draft(fresh_translation, stable_translation)

    _CJK_LANG_PREFIXES = ("zh", "ja", "ko")

    @staticmethod
    def _is_cjk_language(lang_code: str) -> bool:
        """判断语言代码是否属于中日韩（CJK）语言。"""
        normalized = (lang_code or "").strip().lower()
        return any(
            normalized == p or normalized.startswith(p + "-")
            for p in OpenRouterAPI._CJK_LANG_PREFIXES
        )

    @staticmethod
    def _check_content_completeness(
        source_text: str,
        translation: str,
        previous_translation: str,
        previous_source_text: Optional[str] = None,
        detected_source_language: str = "auto",
        target_language: str = "",
    ) -> tuple:
        """检查翻译内容是否完整，返回 (needs_rescue, reason)。

        两个启发式检查：
        1. 翻译相对源文本太短（阈值根据 CJK 跨线情况动态调整）
        2. 源文本增长了但翻译没有增长（使用真实的上次源文本长度）
        """
        source_is_cjk = OpenRouterAPI._is_cjk_language(detected_source_language)
        target_is_cjk = OpenRouterAPI._is_cjk_language(target_language)
        cross_cjk_boundary = source_is_cjk != target_is_cjk
        ratio_threshold = 0.15 if cross_cjk_boundary else 0.5

        ratio = len(translation) / max(len(source_text), 1)
        if ratio < ratio_threshold:
            return True, f"ratio={ratio:.2f}<{ratio_threshold}"

        prev_source_len = (
            len(previous_source_text)
            if previous_source_text
            else len(previous_translation)
        )
        source_vs_prev = len(source_text) / max(prev_source_len, 1)
        trans_growth = len(translation) / max(len(previous_translation), 1)
        source_added = len(source_text) - prev_source_len

        if source_vs_prev >= 1.3 and source_added >= 3 and trans_growth < 1.05:
            return True, (
                f"source_vs_prev={source_vs_prev:.2f} (+{source_added}chars) "
                f"but trans_growth={trans_growth:.2f}"
            )

        return False, ""

    @classmethod
    def _merge_dicts(cls, base: Dict, override: Dict) -> Dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def _build_configured_extra_body(cls) -> Dict:
        raw_extra_body = (
            getattr(config, "OPENAI_COMPAT_EXTRA_BODY_JSON", "") or ""
        ).strip()
        if not raw_extra_body:
            return {}
        try:
            custom_extra_body = json.loads(raw_extra_body)
            if isinstance(custom_extra_body, dict):
                return custom_extra_body
            print("[LLM] Warning: OPENAI_COMPAT_EXTRA_BODY_JSON 不是 JSON 对象，已忽略")
        except Exception as e:
            print(
                f"[LLM] Warning: 解析 OPENAI_COMPAT_EXTRA_BODY_JSON 失败，已忽略: {e}"
            )
        return {}

    def _should_use_parallel_fastest(self, is_partial: bool) -> bool:
        mode = getattr(config, "LLM_PARALLEL_FASTEST_MODE", "off") or "off"
        if mode == "off":
            return False
        if mode == "all":
            return True
        if mode == "final_only":
            if self.streaming_mode and is_partial:
                return False
            return True
        return False

    def _execute_completion_request(self, request_kwargs: Dict[str, object]) -> str:
        try:
            completion = self.client.chat.completions.create(**request_kwargs)
            if completion.choices and completion.choices[0].message.content:
                return self.clean_response(completion.choices[0].message.content)
            return "[ERROR] Empty response from model"
        except Exception as e:
            error_msg = str(e)
            print(f"[LLM] API 调用错误: {error_msg}")
            return f"[ERROR] {error_msg}"

    def _call_api(
        self, messages: List[Dict[str, str]], is_partial: bool = False
    ) -> str:
        """调用 LLM API"""
        try:
            self._maybe_rotate_key()
            # print(f"[LLM] sending request: base_url={self.base_url}, model={self.model}, api_key_set={bool(self.api_key)}")
            request_kwargs: Dict[str, object] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "timeout": self.timeout,
            }
            extra_body = self._build_configured_extra_body()
            if self._is_openrouter_base_url():
                extra_body = self._merge_dicts(
                    extra_body, {"provider": {"sort": "latency"}}
                )
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            if self._should_use_parallel_fastest(is_partial):
                executor = ThreadPoolExecutor(max_workers=2)
                futures = [
                    executor.submit(
                        self._execute_completion_request, dict(request_kwargs)
                    ),
                    executor.submit(
                        self._execute_completion_request, dict(request_kwargs)
                    ),
                ]
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                first_completed = next(iter(done))
                result = first_completed.result()
                executor.shutdown(wait=False, cancel_futures=True)
                return result

            return self._execute_completion_request(request_kwargs)

        except Exception as e:
            error_msg = str(e)
            print(f"[LLM] API 调用错误: {error_msg}")
            return f"[ERROR] {error_msg}"


# 为了向后兼容，保留 OpenRouterStreamingAPI 这个别名类
class OpenRouterStreamingAPI(OpenRouterAPI):
    """
    Streaming-optimized LLM translation API (alias for OpenRouterAPI with streaming_mode=True).
    保留此类以向后兼容。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            streaming_mode=True,
        )
