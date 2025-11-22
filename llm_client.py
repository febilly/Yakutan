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
        
        self._initialized = True
    
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
        """
        if not messages:
            return "[ERROR] No messages provided"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if sort_by_latency:
            payload["provider"] = {"sort": "latency"}

        payload.update(extra_params)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title

        proxies = detect_system_proxy()
        proxy_url = None
        if proxies:
            proxy_url = proxies.get("https") or proxies.get("http")

        timeout_config = aiohttp.ClientTimeout(total=timeout)
        attempt_count = max_retries + 1

        for attempt in range(attempt_count):
            try:
                async with aiohttp.ClientSession(timeout=timeout_config) as session:
                    async with session.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        proxy=proxy_url,
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

                        error_detail = await response.text()
                        return f"[ERROR] HTTP {response.status}: {error_detail.strip()}"

            except (aiohttp.ClientConnectionError, aiohttp.ClientSSLError,
                    ConnectionError, BrokenPipeError) as error:
                if attempt < max_retries:
                    continue
                return f"[ERROR] Connection failed after {attempt_count} attempts: {str(error)}"

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    continue
                return f"[ERROR] Timeout (>{timeout}s) after {attempt_count} attempts"

            except Exception as error:
                return f"[ERROR] Request error: {str(error)}"

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
        """
        try:
            # 尝试获取当前运行的 loop
            loop = asyncio.get_running_loop()
            
            # 如果有 loop 且正在运行，我们不能使用 run_until_complete
            if loop.is_running():
                # 这是一个棘手的情况：我们在一个正在运行的 loop 中调用了同步方法。
                # 通常这意味着代码设计有问题（应该 await 异步方法）。
                # 但为了兼容性，我们可能需要抛出错误或尝试其他变通方法（如 nest_asyncio，但不建议）。
                # 在本项目的上下文中，这通常发生在 run_in_executor 的线程中，此时 get_running_loop 应该抛出 RuntimeError。
                # 如果它没抛出，说明我们在主线程或其他有 loop 的线程中。
                raise RuntimeError("Cannot call sync chat_completion from a running event loop. Use chat_completion_async instead.")
            
            # 如果 loop 存在但未运行（极少见），可以使用它
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

        except RuntimeError:
            # 没有运行中的 loop，这是预期的情况（在线程中）
            # 使用 asyncio.run() 创建一个新的临时 loop 并运行
            # 这比手动管理 loop 更安全，且能确保资源清理
            return asyncio.run(
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
    
# 便捷的全局访问函数
def get_llm_client() -> OpenRouterClient:
    """获取 LLM 客户端单例"""
    return OpenRouterClient()
