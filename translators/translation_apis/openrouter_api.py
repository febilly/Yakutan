"""
LLM translation API implementation using an OpenAI-compatible interface.
支持普通翻译和流式翻译两种模式
"""
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
        'zh': 'Chinese',
        'zh-cn': 'Simplified Chinese',
        'zh-tw': 'Traditional Chinese',
        'zh-hans': 'Simplified Chinese',
        'zh-hant': 'Traditional Chinese',
        'en': 'English',
        'en-us': 'American English',
        'en-gb': 'British English',
        'ja': 'Japanese',
        'ko': 'Korean',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'id': 'Indonesian',
        'ru': 'Russian',
        'ar': 'Arabic',
        'pt': 'Portuguese',
        'th': 'Thai',
        'tl': 'Tagalog (Philippines)',
        'it': 'Italian',
        'tr': 'Turkish',
        'fil': 'Filipino/Tagalog',
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
        normalized = (language_code or '').strip().lower()
        if not normalized or normalized == 'auto':
            return 'auto-detected source language'
        language_name = cls.LANGUAGE_NAME_MAP.get(normalized, normalized.upper())
        return f"{language_name} ({language_code})"

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        context_pairs: Optional[List[Dict[str, str]]] = None,
        **kwargs
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
            return self._translate_streaming(text, source_language, target_language, 
                                             context, context_pairs, **kwargs)
        return self._translate_standard(text, source_language, target_language,
                                        context, context_pairs, **kwargs)

    def _translate_standard(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
        **kwargs
    ) -> str:
        """标准翻译模式"""
        if context_pairs:
            context_lines = []
            for pair in context_pairs:
                context_lines.append(f"Source: {pair['source']}")
                context_lines.append(f"Translation: {pair['target']}")
            context_block = "\n".join(context_lines)
        elif context and context.strip():
            context_block = context.strip()
        else:
            context_block = "None."
        
        source_descriptor = self._describe_language(source_language)
        target_descriptor = self._describe_language(target_language)

        system_prompt = (
            "You are a strict translation assistant. "
            f"Your output must be entirely in {target_descriptor}. "
            "Do not answer in the source language. Do not explain anything. "
            "Return only the translated text."
        )

        user_message = (
            "Previous conversation context (source text and translations):\n"
            f"{context_block}\n\n"
            "Output Format:\n"
            "Provide only the translated text without quotation marks, prefixes, or explanations.\n\n"
            "Translation Principles:\n"
            "1. Keep the tone short, breezy, and friendly.\n"
            "2. Preserve the original meaning, named entities, and essential formatting.\n"
            "3. Prefer natural phrasing that reads well for the target audience.\n"
            "4. Fix any obvious recognition errors in the source text.\n"
            "5. Maintain consistency with the previous translations shown above.\n"
            f"6. The response must be written in {target_descriptor} only.\n"
            "7. If the draft in your head is not in the requested target language, rewrite it before answering.\n\n"
            "Task:\n"
            f"Translate the text below inside <text> and </text> from **{source_descriptor}** to **{target_descriptor}**.\n\n"
            "Text To Translate:\n"
            f"<text>{text}</text>"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        return self._call_api(messages)

    def _translate_streaming(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
        **kwargs
    ) -> str:
        """流式翻译模式（支持 partial 和 previous_translation）"""
        previous_translation = kwargs.get('previous_translation')
        is_partial = kwargs.get('is_partial', False)
        source_descriptor = self._describe_language(source_language)
        target_descriptor = self._describe_language(target_language)

        system_prompt = (
            "You are a strict streaming translator assisting with VRChat conversations. "
            f"Translate all provided source text from {source_descriptor} into {target_descriptor}. "
            f"Your output must stay entirely in {target_descriptor}. "
            "Keep the style casual and friendly unless instructed otherwise. "
            "When a previous translation draft is supplied, behave like a streaming translator: reuse as much wording as possible and only make the smallest edits needed for accuracy and fluency. "
            "Maintain consistency with the previous translations in the conversation history. "
            "Never answer in the source language. Output only the translation without commentary."
        )

        user_sections = []
        if context_pairs:
            context_lines = ["Previous conversation (source and translation pairs):"]
            for pair in context_pairs:
                context_lines.append(f"Source: {pair['source']}")
                context_lines.append(f"Translation: {pair['target']}")
            user_sections.append("\n".join(context_lines))
        elif context and context.strip():
            user_sections.append(f"Scene context and conversation history:\n{context.strip()}")

        user_sections.append(f"Source text:\n{text}")

        if previous_translation:
            user_sections.append(
                "Previously streamed translation (revise minimally, keep structure where possible):\n"
                f"{previous_translation.strip()}"
            )

        if is_partial:
            user_sections.append(
                "This is an incremental streaming update. Provide an updated translation that stays compatible with potential future continuation."
            )
        else:
            user_sections.append("This is the final delivery for this utterance. Ensure the translation reads smoothly.")

        user_sections.append(f"Mandatory output language: {target_descriptor}.")
        user_sections.append("If any wording is not in the requested target language, rewrite it before responding.")
        user_sections.append("Return only the translation text.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]

        return self._call_api(messages)

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
        raw_extra_body = (getattr(config, 'OPENAI_COMPAT_EXTRA_BODY_JSON', '') or '').strip()
        if not raw_extra_body:
            return {}
        try:
            custom_extra_body = json.loads(raw_extra_body)
            if isinstance(custom_extra_body, dict):
                return custom_extra_body
            print('[LLM] Warning: OPENAI_COMPAT_EXTRA_BODY_JSON 不是 JSON 对象，已忽略')
        except Exception as e:
            print(f'[LLM] Warning: 解析 OPENAI_COMPAT_EXTRA_BODY_JSON 失败，已忽略: {e}')
        return {}

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
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
                extra_body = self._merge_dicts(extra_body, {"provider": {"sort": "latency"}})
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            completion = self.client.chat.completions.create(**request_kwargs)
            
            if completion.choices and completion.choices[0].message.content:
                return self.clean_response(completion.choices[0].message.content)
            
            return "[ERROR] Empty response from model"
            
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
