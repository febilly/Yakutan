"""
OpenRouter translation API implementation using hosted large language models.
使用 OpenAI 兼容接口
支持普通翻译和流式翻译两种模式
"""
import os
from typing import Optional, List, Dict

from .base_translation_api import BaseTranslationAPI
from proxy_detector import detect_system_proxy

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI 库未安装。请运行以下命令安装：\n"
        "pip install --upgrade openai"
    )


class OpenRouterAPI(BaseTranslationAPI):
    """
    Translation API that routes requests through OpenRouter.
    Supports both standard and streaming-optimized translation modes.
    """

    SUPPORTS_CONTEXT = True
    
    # OpenRouter 的 OpenAI 兼容端点
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash-lite",
        temperature: float = 0.2,
        timeout: int = 30,
        max_retries: int = 3,
        streaming_mode: bool = False,
    ) -> None:
        """
        初始化 OpenRouter API 客户端
        
        Args:
            model: 使用的模型名称
            temperature: 采样温度
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            streaming_mode: 是否使用流式翻译模式（支持 previous_translation 和 is_partial）
        """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.streaming_mode = streaming_mode
        
        # 获取 API Key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API Key 未设置。请在网页控制面板的 'API Keys 配置' 中填写 OpenRouter API Key。"
            )
        
        # 创建 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.BASE_URL,
        }
        
        # OpenRouter 特有的 headers
        default_headers = {}
        app_url = os.getenv("OPENROUTER_APP_URL", "")
        app_title = os.getenv("OPENROUTER_APP_TITLE", "")
        if app_url:
            default_headers["HTTP-Referer"] = app_url
        if app_title:
            default_headers["X-Title"] = app_title
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        
        # 检测系统代理
        proxies = detect_system_proxy()
        if proxies:
            import httpx
            proxy_url = proxies.get('https') or proxies.get('http')
            if proxy_url:
                client_kwargs["http_client"] = httpx.Client(proxy=proxy_url)
        
        self.client = OpenAI(**client_kwargs)

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
        else:
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
        # 构建上下文块（优先使用 context_pairs，包含原文和译文）
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
        
        # 源语言描述
        source_descriptor = (
            source_language
            if source_language and source_language.lower() != "auto"
            else "auto-detect the source language"
        )

        # 系统提示词
        system_prompt = (
            "You are a helpful translation assistant. Always respond with a concise, light, and friendly tone. "
            "Return only the translated text with no additional commentary."
        )

        # 构建用户消息
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
            "5. Maintain consistency with the previous translations shown above.\n\n"
            "Task:\n"
            f"Translate the text below inside <text> and </text> from **{source_descriptor}** to **{target_language}**.\n\n"
            "Text To Translate:\n"
            f"<text>{text}</text>"
        )

        # 构建消息列表
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

        # 系统提示词
        system_prompt = (
            "You are a skilled bilingual translator assisting with VRChat conversations. "
            f"Translate all provided source text into {target_language} while preserving intent, tone, and register. "
            "Keep the style casual and friendly unless instructed otherwise. "
            "When a previous translation draft is supplied, behave like a streaming translator: reuse as much wording as possible and only make the smallest edits needed for accuracy and fluency. "
            "Maintain consistency with the previous translations in the conversation history. "
            "Output only the translation without commentary."
        )

        user_sections = []
        
        # 构建上下文块（优先使用 context_pairs，包含原文和译文）
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

        user_sections.append("Return only the translation text.")

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]

        return self._call_api(messages)

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """调用 OpenRouter API"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                timeout=self.timeout,
                extra_body={"provider": {"sort": "latency"}},
            )
            
            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content.strip()
            
            return "[ERROR] Empty response from model"
            
        except Exception as e:
            error_msg = str(e)
            print(f"[OpenRouter] API 调用错误: {error_msg}")
            return f"[ERROR] {error_msg}"


# 为了向后兼容，保留 OpenRouterStreamingAPI 作为别名
class OpenRouterStreamingAPI(OpenRouterAPI):
    """
    Streaming-optimized translation API (alias for OpenRouterAPI with streaming_mode=True).
    保留此类以向后兼容。
    """
    
    def __init__(
        self,
        model: str = "google/gemini-2.5-flash-lite",
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
