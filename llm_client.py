"""
OpenRouter LLM Client - 单例模式
统一管理所有通过 OpenRouter 的大模型调用
"""
import asyncio
import aiohttp
import json
import os
from typing import Optional, List, Dict, Any
import threading
from proxy_detector import detect_system_proxy


class OpenRouterClient:
    """OpenRouter 客户端单例，管理所有 LLM 调用"""
    
    _instance = None
    _lock = threading.Lock()
    
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
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.default_timeout = 30
        self.default_max_retries = 3
        
        # HTTP 头部配置
        self.app_url = os.getenv("OPENROUTER_APP_URL", "")
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "")
        
        # 长连接会话
        self._session: Optional[aiohttp.ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        self._initialized = True
    
    async def _get_session(self, timeout: int) -> aiohttp.ClientSession:
        """获取或创建 HTTP 长连接会话"""
        if self._session is None or self._session.closed:
            session_timeout = aiohttp.ClientTimeout(total=timeout)
            
            # 检测系统代理并配置
            proxies = detect_system_proxy()
            proxy_url = None
            if proxies:
                # aiohttp 只需要一个代理 URL，优先使用 https 代理
                proxy_url = proxies.get('https') or proxies.get('http')
            
            self._session = aiohttp.ClientSession(timeout=session_timeout)
        return self._session
    
    async def _reset_session(self, timeout: int) -> aiohttp.ClientSession:
        """重置长连接会话"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        return await self._get_session(timeout)
    
    async def close(self):
        """关闭所有连接"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
    
    async def chat_completion_async(
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
        异步调用 OpenRouter Chat Completion API
        
        Args:
            messages: 消息列表，格式: [{"role": "system/user/assistant", "content": "..."}]
            model: 模型名称，默认使用免费的 Gemini 2.0 Flash
            temperature: 温度参数 (0.0-2.0)
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            sort_by_latency: 是否按延迟排序提供商（优先选择最快的）
            **extra_params: 其他 OpenRouter API 参数
        
        Returns:
            模型生成的文本内容
        """
        if not messages:
            return "[ERROR] No messages provided"
        
        # 构建请求 payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        if sort_by_latency:
            payload["provider"] = {"sort": "latency"}
        
        # 合并额外参数
        payload.update(extra_params)
        
        # 构建请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        
        # 重试逻辑
        for attempt in range(max_retries + 1):
            try:
                session = await self._get_session(timeout)
                
                # 获取代理设置
                proxies = detect_system_proxy()
                proxy_url = None
                if proxies:
                    proxy_url = proxies.get('https') or proxies.get('http')
                
                async with session.post(
                    self.base_url, 
                    json=payload, 
                    headers=headers,
                    proxy=proxy_url
                ) as response:
                    if response.status == 200:
                        response_body = await response.text()
                        
                        try:
                            data = json.loads(response_body)
                        except json.JSONDecodeError:
                            return "[ERROR] Failed to decode OpenRouter response"
                        
                        choices = data.get("choices") or []
                        if not choices:
                            message = data.get("error", {}).get("message", "No choices returned")
                            return f"[ERROR] {message}"
                        
                        content = choices[0].get("message", {}).get("content")
                        if not content:
                            return "[ERROR] Empty response from model"
                        
                        return content.strip()
                    else:
                        error_detail = await response.text()
                        return f"[ERROR] HTTP {response.status}: {error_detail.strip()}"
            
            except (aiohttp.ClientConnectionError, aiohttp.ClientSSLError,
                    ConnectionError, BrokenPipeError) as e:
                if attempt < max_retries:
                    try:
                        await self._reset_session(timeout)
                    except Exception:
                        pass
                    continue
                else:
                    return f"[ERROR] Connection failed after {max_retries + 1} attempts: {str(e)}"
            
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    try:
                        await self._reset_session(timeout)
                    except Exception:
                        pass
                    continue
                else:
                    return f"[ERROR] Timeout (>{timeout}s) after {max_retries + 1} attempts"
            
            except Exception as e:
                return f"[ERROR] Request error: {str(e)}"
        
        return "[ERROR] Unknown error after all retry attempts"
    
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
        同步调用 OpenRouter Chat Completion API（包装异步方法）
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间
            max_retries: 最大重试次数
            sort_by_latency: 是否按延迟排序
            **extra_params: 其他参数
        
        Returns:
            模型生成的文本
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            loop = self._loop
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.chat_completion_async(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                sort_by_latency=sort_by_latency,
                **extra_params
            )
        )
    
    def __del__(self):
        """清理资源"""
        try:
            if self._loop and not self._loop.is_closed():
                if self._loop.is_running():
                    self._loop.create_task(self.close())
                else:
                    self._loop.run_until_complete(self.close())
                self._loop.close()
                self._loop = None
        except Exception:
            pass


# 便捷的全局访问函数
def get_llm_client() -> OpenRouterClient:
    """获取 LLM 客户端单例"""
    return OpenRouterClient()
