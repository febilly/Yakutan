"""
Qwen-MT API 翻译实现
使用阿里云 DashScope 的 Qwen-MT 模型进行翻译
需要在环境变量中设置 DASHSCOPE_API_KEY
"""
import os
from typing import Optional, List, Dict
from .base_translation_api import BaseTranslationAPI
from proxy_detector import detect_system_proxy
import config

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI 库未安装。请运行以下命令安装：\n"
        "pip install --upgrade openai"
    )


class QwenMTAPI(BaseTranslationAPI):
    """Qwen-MT 翻译 API 封装（使用 OpenAI 兼容接口）"""
    
    # Qwen-MT 支持翻译记忆功能（原生上下文）
    SUPPORTS_CONTEXT = True
    
    # API 端点配置
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    BASE_URL_INTERNATIONAL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    
    # 语言代码映射（只处理需要转换的特殊情况）
    LANGUAGE_MAP = {
        'zh-cn': 'zh',
        'zh-hans': 'zh',
        'zh-hant': 'zh-tw',
        'en-us': 'en',
        'en-gb': 'en',
        'en-au': 'en',
        'pt-br': 'pt',
        'pt-pt': 'pt',
    }
    
    # 领域提示（仅支持英文）
    # 用于指定翻译的领域风格，帮助模型更好地理解上下文
    DOMAINS = "The text is casual conversation from VRChat, a social virtual reality platform. Keep translations natural, friendly and colloquial."
    
    def __init__(self, api_key: str = None, model: str = "qwen-mt-flash", 
                 use_international: bool = None):
        """
        初始化 Qwen-MT 客户端
        
        Args:
            api_key: DashScope API Key（如果为 None，从环境变量 DASHSCOPE_API_KEY 获取）
            model: 使用的模型，可选 qwen-mt-plus, qwen-mt-flash, qwen-mt-turbo, qwen-mt-lite
            use_international: 是否使用国际版端点（如果为 None，从全局配置读取）
        """
        self.api_key = api_key or os.environ.get('DASHSCOPE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "DashScope API Key 未设置。请在网页控制面板的 'API Keys 配置' 中填写阿里云 DashScope API Key。"
            )
        
        self.model = model
        
        # 从全局配置读取国际版设置
        if use_international is None:
            use_international = getattr(config, 'USE_INTERNATIONAL_ENDPOINT', False)
        
        # 选择端点
        base_url = self.BASE_URL_INTERNATIONAL if use_international else self.BASE_URL
        
        # 检测系统代理
        proxies = detect_system_proxy()
        
        # 创建 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": base_url,
        }
        
        # OpenAI 客户端支持 http_client 参数来配置代理
        if proxies:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxy=proxies.get('https') or proxies.get('http'))
        
        self.client = OpenAI(**client_kwargs)
    
    def _get_language_code(self, lang_code: str) -> str:
        """将语言代码标准化，处理特殊情况"""
        code = lang_code.lower()
        # 如果在映射表中，返回映射值；否则直接返回原代码
        return self.LANGUAGE_MAP.get(code, code)
    
    def translate(self, text: str, source_language: str = 'auto', 
                  target_language: str = 'zh-CN', context: Optional[str] = None, 
                  context_pairs: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码（'auto' 表示自动检测）
            target_language: 目标语言代码
            context: 上下文信息（Qwen-MT 使用 tm_list 替代，此参数被忽略）
            context_pairs: 翻译记忆对列表，每个元素包含 'source' 和 'target' 键
            **kwargs: 其他参数
        
        Returns:
            翻译后的文本
        """
        try:
            # 标准化语言代码
            source_lang = self._get_language_code(source_language)
            target_lang = self._get_language_code(target_language)
            
            # 构建翻译选项
            translation_options = {
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
            
            # 添加翻译记忆（tm_list）
            if context_pairs:
                tm_list = []
                for pair in context_pairs:
                    tm_list.append({
                        "source": pair['source'],
                        "target": pair['target']
                    })
                translation_options["tm_list"] = tm_list
            
            # 添加领域提示（domains）
            if self.DOMAINS:
                translation_options["domains"] = self.DOMAINS
            
            # 构建消息
            messages = [
                {"role": "user", "content": text}
            ]
            
            # 调用 API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"translation_options": translation_options},
            )
            
            # 提取翻译结果
            translated_text = completion.choices[0].message.content
            
            return translated_text.strip() if translated_text else ""
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Qwen-MT] 翻译错误: {error_msg}")
            return f"[ERROR] {error_msg}"
