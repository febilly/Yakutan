"""
LLM translation API implementation using an OpenAI-compatible interface.
支持普通翻译和流式翻译两种模式

流式模式：
- Partial: 传 draft，使用 continuation framing 保持稳定中间翻译
- Final: 使用 continuation framing（传 draft）得到终译，直接采用该结果
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from typing import Optional, List, Dict

from .base_translation_api import BaseTranslationAPI
import config
from openai_compat_client import OpenAICompatClientBase


class OpenRouterAPI(OpenAICompatClientBase, BaseTranslationAPI):
    """
    Generic LLM translation API based on an OpenAI-compatible endpoint.
    Supports both standard and streaming-optimized translation modes.
    """

    SUPPORTS_CONTEXT = True

    LANGUAGE_NAME_MAP = {
        "zh": "CHINESE",
        "zh-cn": "SIMPLIFIED CHINESE",
        "zh-tw": "TRADITIONAL CHINESE",
        "zh-hans": "SIMPLIFIED CHINESE",
        "zh-hant": "TRADITIONAL CHINESE",
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
        """流式翻译：partial 与 final 均通过 continuation prompt 调用一次 LLM。"""
        previous_translation = kwargs.get("previous_translation")
        is_partial = kwargs.get("is_partial", False)
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

        return self._call_api(messages, is_partial=is_partial)

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
        if not getattr(config, "ENABLE_LLM_PARALLEL_FASTEST", False):
            return False
        if self.streaming_mode and is_partial:
            return False
        return True

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
