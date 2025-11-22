"""
翻译 API 抽象基类
定义所有翻译 API 的统一接口
"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseTranslationAPI(ABC):
    """
    翻译 API 抽象基类
    所有翻译 API 实现都应该继承此类
    """
    
    # 子类应该设置此属性以指示是否原生支持上下文
    SUPPORTS_CONTEXT: bool = False
    
    @abstractmethod
    def translate(self, text: str, source_language: str = 'auto', 
                  target_language: str = 'zh-CN', context: Optional[str] = None, **kwargs) -> str:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码（'auto' 表示自动检测）
            target_language: 目标语言代码
            context: 可选的上下文信息。只有支持上下文的 API 才应该接受此参数。
                     不支持上下文的 API 如果收到此参数应该抛出异常。
            **kwargs: 其他可选参数，如 previous_translation, is_partial 等
        
        Returns:
            翻译后的文本
        
        Raises:
            NotImplementedError: 如果 API 不支持上下文但收到了 context 参数
        """
        pass
