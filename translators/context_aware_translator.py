"""
基础翻译器模块
提供通用的上下文感知翻译功能，支持可插拔的翻译 API
"""
from collections import deque
from typing import Optional, Tuple, Callable, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from .translation_apis.base_translation_api import BaseTranslationAPI


class TranslationHistoryEntry:
    """翻译历史条目"""
    def __init__(self, source_text: str, translated_text: str, target_language: str):
        self.source_text = source_text
        self.translated_text = translated_text
        self.target_language = target_language


class ContextAwareTranslator:
    """
    基础翻译器类，支持上下文感知
    仿照 LiveCaptions-Translator 实现
    
    支持两种类型的 API：
    1. 原生支持上下文的 API（如 DeepL）- 直接传递 context 参数
    2. 不支持上下文的 API（如 Google Translator）- 使用标记法模拟上下文
    """
    
    def __init__(self, 
                 translation_api: 'BaseTranslationAPI',
                 api_name: str = "DefaultAPI",
                 max_context_size: int = 6, 
                 target_language: str = 'zh-CN', 
                 context_aware: bool = True):
        """
        初始化翻译器
        
        Args:
            translate_api: 实现了 BaseTranslator 接口的翻译 API 实例
            api_name: 翻译 API 的名称（用于日志和调试）
            max_context_size: 保存的最大历史记录条数（默认 6）
            target_language: 目标语言代码（默认中文简体）
            context_aware: 是否启用上下文感知功能（默认启用）
        """
        self.translation_api = translation_api
        self.api_name = api_name
        self.max_context_size = max_context_size
        self.target_language = target_language
        self.context_aware = context_aware
        
        # 检查 API 是否原生支持上下文
        self.native_context_support = getattr(translation_api, 'SUPPORTS_CONTEXT', False)
        
        # 使用 deque 存储翻译历史
        self.contexts: deque = deque(maxlen=max_context_size)
        
        # 线程锁，确保线程安全
        self._lock = threading.RLock()
    
    @property
    def display_contexts(self):
        """获取反序的上下文，用于显示或构建提示"""
        with self._lock:
            return list(reversed(list(self.contexts)))
    
    def _get_previous_caption(self, count: Optional[int] = None) -> str:
        """
        获取之前的字幕文本作为上下文前缀
        
        Args:
            count: 要包含的历史记录条数，如果为 None 则使用所有可用记录
        
        Returns:
            组合后的上下文前缀文本
        """
        if count is None:
            count = len(self.contexts)
        
        if count <= 0:
            return ""
        
        with self._lock:
            # 从最新的记录开始反向获取
            contexts_to_use = self.display_contexts[:count]
            
            if not contexts_to_use:
                return ""
            
            # 构建组合的前缀文本
            prefix_parts = []
            for entry in contexts_to_use:
                prefix_parts.append(entry.source_text)
            
            # 使用空格或句号连接
            prefix = " ".join(prefix_parts)
            
            # 确保以句号结尾（中文用。，英文用.）
            if prefix and not prefix.endswith(('.', '。', '!', '！', '?', '？')):
                # 简单判断：如果包含汉字，使用中文句号
                if any('\u4e00' <= char <= '\u9fff' for char in prefix):
                    prefix += "。"
                else:
                    prefix += "."
            
            # 末尾添加空格以分隔当前句子
            if prefix and not prefix.endswith(' '):
                prefix += " "
            
            return prefix
    
    def translate(
            self, text: str,
            source_language: str = 'auto',
            target_language: Optional[str] = None,
            context_prefix: str = "",
            **kwargs
        ) -> str:
        """
        翻译句子，支持上下文感知
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码（默认自动检测）
            target_language: 目标语言代码（如果不指定则使用初始化时的默认值）
            context_prefix: 上下文前缀
            **kwargs: 其他参数，如 previous_translation, is_partial
        
        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return ""
        
        # 使用传入的目标语言或默认值
        actual_target_language = target_language if target_language is not None else self.target_language
        
        try:
            # 根据 API 是否原生支持上下文选择不同的处理方式
            if self.native_context_support:
                # API 原生支持上下文（如 DeepL）
                if self.context_aware and (len(self.contexts) > 0 or context_prefix):
                    # 构建上下文字符串
                    context = f"{context_prefix}\n{self._get_previous_caption()}"
                    # 直接调用 API 的 translate 方法，传入 context 参数
                    translated_text = self.translation_api.translate(
                        text,
                        source_language=source_language,
                        target_language=actual_target_language,
                        context=context,
                        **kwargs
                    )
                else:
                    # 没有上下文或未启用上下文感知
                    translated_text = self.translation_api.translate(
                        text,
                        source_language=source_language,
                        target_language=actual_target_language,
                        **kwargs
                    )
            else:
                # API 不支持原生上下文，使用标记法模拟
                if self.context_aware and len(self.contexts) > 0:
                    previous_caption = self._get_previous_caption()
                    # 使用 <[text]> 标记当前文本
                    input_text = f"{previous_caption}<=={text}==>"
                else:
                    input_text = text
                
                # 调用翻译 API（不传入 context 参数）
                translated_text = self.translation_api.translate(
                    input_text,
                    source_language=source_language,
                    target_language=actual_target_language,
                    **kwargs
                )
                
                # 提取当前句子的翻译（如果有标记）
                if self.context_aware and '<==' in translated_text:
                    try:
                        # 尝试提取 <==...==> 之间的内容
                        start_idx = translated_text.rfind('<==')
                        end_idx = translated_text.rfind('==>')
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            # 提取标记内的翻译
                            extracted = translated_text[start_idx + 3:end_idx]
                            # 清理翻译结果
                            translated_text = extracted.strip()
                    except Exception:
                        # 如果提取失败，使用完整翻译结果
                        pass
                
                # 清理结果中的标记符号
                translated_text = translated_text.replace('<==', '').replace('==>', '')
            
            translated_text = translated_text.strip()
            
            # 保存到历史记录
            with self._lock:
                history_entry = TranslationHistoryEntry(
                    source_text=text,
                    translated_text=translated_text,
                    target_language=actual_target_language
                )
                self.contexts.append(history_entry)
            
            return translated_text
        
        except Exception as e:
            print(f"翻译错误: {str(e)}")
            return f"[ERROR] {str(e)}"
    
    def translate_with_context(self, text: str, source_language: str = 'auto') -> Tuple[str, dict]:
        """
        翻译句子并返回翻译结果和完整的上下文信息
        
        Returns:
            (translated_text, context_info) 元组
        """
        translated_text = self.translate(text, source_language)
        
        with self._lock:
            context_info = {
                'contexts_count': len(self.contexts),
                'previous_contexts': [
                    {
                        'source': entry.source_text,
                        'translated': entry.translated_text
                    }
                    for entry in self.display_contexts[:-1]  # 排除最新的当前项
                ]
            }
        
        return translated_text, context_info
    
    def set_context_aware(self, enabled: bool):
        """启用或禁用上下文感知功能"""
        with self._lock:
            self.context_aware = enabled
    
    def clear_contexts(self):
        """清空历史记录"""
        with self._lock:
            self.contexts.clear()
    
    def set_target_language(self, language_code: str):
        """设置目标语言"""
        with self._lock:
            self.target_language = language_code
    
    def get_contexts(self) -> list:
        """获取当前所有上下文记录"""
        with self._lock:
            return [
                {
                    'source': entry.source_text,
                    'translated': entry.translated_text,
                    'language': entry.target_language
                }
                for entry in self.display_contexts
            ]
    
    def __repr__(self):
        return (f"BaseTranslator(api={self.api_name}, "
                f"max_context_size={self.max_context_size}, "
                f"target_language='{self.target_language}', "
                f"context_aware={self.context_aware}, "
                f"current_contexts={len(self.contexts)})")
