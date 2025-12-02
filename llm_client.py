"""
OpenRouter LLM Client - 单例模式
统一管理所有通过 OpenRouter 的大模型调用
使用 OpenAI 兼容接口
"""
import os
from typing import Optional, List, Dict, Any
import threading
from proxy_detector import detect_system_proxy

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI 库未安装。请运行以下命令安装：\n"
        "pip install --upgrade openai"
    )


class OpenRouterClient:
    """OpenRouter 客户端单例，管理所有 LLM 调用"""
    
    _instance = None
    _lock = threading.Lock()
    
    # OpenRouter 的 OpenAI 兼容端点
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            return
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API Key 未设置。请在网页控制面板的 'API Keys 配置' 中填写 OpenRouter API Key。"
            )
        
        # HTTP 头部配置（OpenRouter 特有）
        self.app_url = os.getenv("OPENROUTER_APP_URL", "")
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "")
        
        # 创建 OpenAI 客户端
        self._create_client()
        
        self._initialized = True
    
    def _create_client(self):
        """创建或重建 OpenAI 客户端"""
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.BASE_URL,
        }
        
        # 构建默认头部
        default_headers = {}
        if self.app_url:
            default_headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            default_headers["X-Title"] = self.app_title
        
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

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "google/gemini-2.5-flash-lite",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        timeout: int = 30,
        max_retries: int = 3,
        sort_by_latency: bool = True,
        **extra_params: Any
    ) -> str:
        """
        调用 OpenRouter Chat Completion API
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
            sort_by_latency: 是否按延迟排序选择 provider
            **extra_params: 其他参数
        
        Returns:
            模型返回的文本内容
        """
        if not messages:
            return "[ERROR] No messages provided"

        try:
            # 构建请求参数
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout,
            }
            
            if max_tokens:
                request_kwargs["max_tokens"] = max_tokens
            
            # OpenRouter 特有的 extra_body 参数
            extra_body = {}
            if sort_by_latency:
                extra_body["provider"] = {"sort": "latency"}
            
            # 合并额外参数到 extra_body
            extra_body.update(extra_params)
            
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            
            # 调用 API
            completion = self.client.chat.completions.create(**request_kwargs)
            
            # 提取响应内容
            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content.strip()
            
            return "[ERROR] Empty response from model"
            
        except Exception as e:
            error_msg = str(e)
            print(f"[OpenRouter] API 调用错误: {error_msg}")
            return f"[ERROR] {error_msg}"
def get_llm_client() -> OpenRouterClient:
    """获取 LLM 客户端单例"""
    return OpenRouterClient()
