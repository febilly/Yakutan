"""
OpenRouter translation API implementation using hosted large language models.
"""
import os
from typing import Optional

from .base_translation_api import BaseTranslationAPI
from llm_client import get_llm_client


class OpenRouterAPI(BaseTranslationAPI):
    """Translation API that routes requests through OpenRouter."""

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
        
        # 系统提示词
        self.system_prompt = (
            "You are a helpful translation assistant. Always respond with a concise, light, and friendly tone. "
            "Return only the translated text with no additional commentary."
        )

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
    ) -> str:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码
            target_language: 目标语言代码
            context: 可选的上下文信息
        
        Returns:
            翻译后的文本
        """
        if not text:
            return ""

        # 构建上下文块
        context_block = context.strip() if context and context.strip() else "None."
        
        # 源语言描述
        source_descriptor = (
            source_language
            if source_language and source_language.lower() != "auto"
            else "auto-detect the source language"
        )

        # 构建用户消息
        user_message = (
            "Context Section:\n"
            f"{context_block}\n\n"
            "Output Format:\n"
            "Provide only the translated text without quotation marks, prefixes, or explanations.\n\n"
            "Translation Principles:\n"
            "1. Keep the tone short, breezy, and friendly.\n"
            "2. Preserve the original meaning, named entities, and essential formatting.\n"
            "3. Prefer natural phrasing that reads well for the target audience.\n"
            "4. Fix any obvious recognition errors in the source text.\n\n"
            "Task:\n"
            f"Translate the text below inside <text> and </text> from **{source_descriptor}** to **{target_language}**.\n\n"
            "Text To Translate:\n"
            f"<text>{text}</text>"
        )

        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
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
