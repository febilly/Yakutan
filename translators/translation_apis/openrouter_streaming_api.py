"""
OpenRouter Streaming Translation API implementation.
"""
from typing import Optional, List, Dict

from .base_translation_api import BaseTranslationAPI
from llm_client import get_llm_client


class OpenRouterStreamingAPI(BaseTranslationAPI):
    """
    Translation API that routes requests through OpenRouter with streaming-optimized prompts.
    Supports partial results and previous translation context.
    """

    SUPPORTS_CONTEXT = True

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash-lite",
        temperature: float = 0.2,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 获取 LLM 客户端单例
        self.llm_client = get_llm_client()

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码
            target_language: 目标语言代码
            context: 可选的上下文信息
            **kwargs: 
                previous_translation: 上一次的翻译结果 (str)
                is_partial: 是否为部分结果 (bool)
        
        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return ""

        previous_translation = kwargs.get('previous_translation')
        is_partial = kwargs.get('is_partial', False)

        # 系统提示词
        system_prompt = (
            "You are a skilled bilingual translator assisting with VRChat conversations. "
            f"Translate all provided source text into {target_language} while preserving intent, tone, and register. "
            "Keep the style casual and friendly unless instructed otherwise. "
            "When a previous translation draft is supplied, behave like a streaming translator: reuse as much wording as possible and only make the smallest edits needed for accuracy and fluency. "
            "Output only the translation without commentary."
        )

        user_sections = []
        if context and context.strip():
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

        user_sections.append("Return only the translation text.")

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]

        # 调用 LLM 客户端
        result = self.llm_client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
            sort_by_latency=True,
        )

        return result
