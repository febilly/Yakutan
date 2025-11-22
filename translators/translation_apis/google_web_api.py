"""
Google Web Translator API 翻译实现
使用 googletrans 库
"""
import asyncio
from typing import Optional
from googletrans import Translator as GoogleWebTranslatorAPI
from .base_translation_api import BaseTranslationAPI
from proxy_detector import detect_system_proxy


class GoogleWebAPI(BaseTranslationAPI):
    """Google Web Translator API 封装"""
    
    # Google Web Translator 不支持原生上下文
    SUPPORTS_CONTEXT = False
    
    def __init__(self):
        """初始化 API"""
        # 检测系统代理
        proxies = detect_system_proxy()
        
        if proxies:
            # googletrans 库支持代理参数
            self.google_translator = GoogleWebTranslatorAPI(proxies=proxies)
        else:
            self.google_translator = GoogleWebTranslatorAPI()
    
    async def _translate_async(self, text: str, source_language: str, target_language: str) -> str:
        """
        异步翻译方法
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码
            target_language: 目标语言代码
        
        Returns:
            翻译后的文本
        """
        result = await self.google_translator.translate(
            text,
            src=source_language,
            dest=target_language
        )
        return result.text
    
    def translate(self, text: str, source_language: str = 'auto', 
                  target_language: str = 'zh-CN', context: Optional[str] = None, **kwargs) -> str:
        """
        同步翻译接口（包装异步调用）
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码
            target_language: 目标语言代码
            context: 上下文信息（此 API 不支持，如果提供将抛出异常）
            **kwargs: 其他参数
        
        Returns:
            翻译后的文本
        
        Raises:
            NotImplementedError: 如果提供了 context 参数
        """
        # 检查是否提供了上下文参数
        if context is not None:
            raise NotImplementedError(
                "Google Web Translator API 不支持原生上下文功能。"
                "请使用 ContextAwareTranslator 包装器来启用上下文感知翻译。"
            )
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环，创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 同步执行异步翻译
        return loop.run_until_complete(
            self._translate_async(text, source_language, target_language)
        )
