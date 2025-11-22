"""
Google Dictionary API 翻译实现
使用 Google Dictionary Extension API
"""
import asyncio
import aiohttp
import urllib.parse
import json
from typing import Optional
from .base_translation_api import BaseTranslationAPI
from proxy_detector import detect_system_proxy


class GoogleDictionaryAPI(BaseTranslationAPI):
    """Google Dictionary 翻译 API 封装"""
    
    # Google Dictionary API 不支持原生上下文
    SUPPORTS_CONTEXT = False
    
    def __init__(self, max_retries: int = 3):
        """
        初始化 API 配置
        
        Args:
            max_retries: 连接断掉后的重试次数，默认 3 次
        """
        self.api_key = "AIzaSyA6EEtrDCfBkHV8uU2lgGY-N383ZgAOo7Y"
        self.api_endpoint = "https://dictionaryextension-pa.googleapis.com/v1/dictionaryExtensionData"
        self.strategy = "2"
        self.timeout = 2  # 超时时间（秒）
        self.max_retries = max_retries  # 重试次数
        
        # 创建长连接会话
        self._session_timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = None
        self._loop = None
        
        # 提前进行一次翻译，以建立长连接
        self.translate("你好", source_language="auto", target_language="en")
    
    async def _get_session(self):
        """获取或创建 HTTP 长连接会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._session_timeout)
        return self._session
    
    async def _reset_session(self):
        """重置长连接会话（关闭旧的，创建新的）"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        # 创建新的会话
        return await self._get_session()
    
    async def close(self):
        """关闭 HTTP 长连接会话"""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
    
    async def _translate_async(self, text: str, source_language: str, target_language: str) -> str:
        """
        异步翻译方法（带重试逻辑）
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码（此 API 会自动检测，该参数保留以兼容接口）
            target_language: 目标语言代码
        
        Returns:
            翻译后的文本
        """
        # 尝试指定次数
        for attempt in range(self.max_retries + 1):
            try:
                # URL 编码文本
                encoded_text = urllib.parse.quote(text)
                
                # 构建请求 URL
                url = (f"{self.api_endpoint}?"
                       f"language={target_language}&"
                       f"key={self.api_key}&"
                       f"term={encoded_text}&"
                       f"strategy={self.strategy}")
                
                # 设置请求头（模拟 Chrome 扩展）
                headers = {
                    'x-referer': 'chrome-extension://mgijmajocgfcbeboacabfgobmjgjcoja'
                }
                
                # 获取长连接会话
                session = await self._get_session()
                
                # 获取代理设置
                proxies = detect_system_proxy()
                proxy_url = None
                if proxies:
                    proxy_url = proxies.get('https') or proxies.get('http')
                
                # 发送请求（使用长连接）
                async with session.get(url, headers=headers, proxy=proxy_url) as response:
                    if response.status == 200:
                        response_body = await response.text()
                        
                        # 解析 JSON 响应
                        data = json.loads(response_body)
                        
                        # 提取翻译结果
                        if 'translateResponse' in data:
                            translated_text = data['translateResponse'].get('translateText', '')
                            return translated_text
                        else:
                            return "[ERROR] Translation Failed: Unexpected API response format"
                    else:
                        return f"[ERROR] HTTP {response.status}: {await response.text()}"
            
            except (aiohttp.ClientConnectionError, aiohttp.ClientSSLError, 
                    ConnectionError, BrokenPipeError) as e:
                # 长连接断掉的错误
                if attempt < self.max_retries:
                    # 还有重试次数，重置会话后重试
                    try:
                        await self._reset_session()
                    except Exception:
                        pass
                    # 继续下一次尝试
                    continue
                else:
                    # 重试次数已用完，返回错误
                    return f"[ERROR] Connection failed after {self.max_retries + 1} attempts: {str(e)}"
            
            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    # 超时也进行重试
                    try:
                        await self._reset_session()
                    except Exception:
                        pass
                    continue
                else:
                    return f"[ERROR] Translation timeout (>{self.timeout}s) after {self.max_retries + 1} attempts"
            
            except Exception as e:
                # 其他异常直接返回错误
                return f"[ERROR] {str(e)}"
        
        # 不应该到达这里
        return "[ERROR] Unknown error after all retry attempts"
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            if self._loop and not self._loop.is_closed():
                if self._loop.is_running():
                    # 如果事件循环正在运行，使用 create_task 替代
                    self._loop.create_task(self.close())
                else:
                    # 否则直接运行关闭任务
                    self._loop.run_until_complete(self.close())
                self._loop.close()
                self._loop = None
        except Exception:
            # 忽略清理过程中的错误
            pass
    
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
                "Google Dictionary API 不支持原生上下文功能。"
                "请使用 ContextAwareTranslator 包装器来启用上下文感知翻译。"
            )
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环，复用或创建新的
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            loop = self._loop
            asyncio.set_event_loop(loop)
        
        # 同步执行异步翻译
        return loop.run_until_complete(
            self._translate_async(text, source_language, target_language)
        )
