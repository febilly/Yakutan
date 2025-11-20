"""
Qwen-MT 翻译 API 实现
使用阿里云的 Qwen-MT 机器翻译模型，支持 92 个语种互译
支持术语干预、翻译记忆和领域提示等高级功能
"""
import os
from typing import Optional, List, Dict, Any
from .base_translation_api import BaseTranslationAPI

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "OpenAI 库未安装。请运行以下命令安装：\n"
        "pip install --upgrade openai"
    )


class QwenMTAPI(BaseTranslationAPI):
    """Qwen-MT 翻译 API 封装（使用 OpenAI 兼容接口）"""
    
    # Qwen-MT 通过 tm_list (翻译记忆) 支持上下文
    SUPPORTS_CONTEXT = True
    
    # 语言代码映射（将常用代码映射到 Qwen-MT 支持的格式）
    LANGUAGE_MAP = {
        'auto': 'auto',
        'zh': 'Chinese',
        'zh-cn': 'Chinese',
        'zh-hans': 'Chinese',
        'zh-tw': 'Traditional Chinese',
        'zh-hant': 'Traditional Chinese',
        'en': 'English',
        'ja': 'Japanese',
        'ko': 'Korean',
        'es': 'Spanish',
        'fr': 'French',
        'pt': 'Portuguese',
        'de': 'German',
        'it': 'Italian',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'ms': 'Malay',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'he': 'Hebrew',
        'my': 'Burmese',
        'ta': 'Tamil',
        'ur': 'Urdu',
        'bn': 'Bengali',
        'pl': 'Polish',
        'nl': 'Dutch',
        'ro': 'Romanian',
        'tr': 'Turkish',
        'km': 'Khmer',
        'lo': 'Lao',
        'yue': 'Cantonese',
        'cs': 'Czech',
        'el': 'Greek',
        'sv': 'Swedish',
        'hu': 'Hungarian',
        'da': 'Danish',
        'fi': 'Finnish',
        'uk': 'Ukrainian',
        'bg': 'Bulgarian',
        'sr': 'Serbian',
        'te': 'Telugu',
        'af': 'Afrikaans',
        'ru': 'Russian',
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-mt-flash",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        stream: bool = False,
        terms: Optional[List[Dict[str, str]]] = None,
        domains: Optional[str] = None,
    ):
        """
        初始化 Qwen-MT 客户端
        
        Args:
            api_key: DashScope API Key（如果为 None，从环境变量 DASHSCOPE_API_KEY 获取）
            model: 模型名称，可选：qwen-mt-flash（默认）、qwen-mt-plus、qwen-mt-turbo
            base_url: API 基础 URL（北京地域或新加坡地域）
            stream: 是否启用流式输出（仅 qwen-mt-flash 支持增量流式）
            terms: 术语表，格式：[{"source": "术语", "target": "翻译"}]
            domains: 领域提示（英文）
        """
        # 从参数或环境变量获取 API Key
        auth_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        
        if not auth_key:
            raise ValueError(
                "DashScope API Key 未设置。请在网页控制面板的 'API Keys 配置' 中填写 DashScope API Key。"
            )
        
        # 创建 OpenAI 客户端
        self.client = OpenAI(
            api_key=auth_key,
            base_url=base_url,
        )
        
        self.model = model
        self.stream = stream
        self.terms = terms or []
        self.domains = domains
    
    def _normalize_language_code(self, lang_code: str) -> str:
        """
        规范化语言代码为 Qwen-MT 支持的格式
        
        Args:
            lang_code: 输入的语言代码
        
        Returns:
            Qwen-MT 支持的语言名称或代码
        """
        if not lang_code:
            return 'auto'
        
        # 转换为小写进行匹配
        lang_lower = lang_code.lower()
        
        # 如果在映射表中，返回对应的值
        if lang_lower in self.LANGUAGE_MAP:
            return self.LANGUAGE_MAP[lang_lower]
        
        # 如果不在映射表中，尝试首字母大写后返回
        return lang_code.capitalize()
    
    def translate(
        self,
        text: str,
        source_language: str = 'auto',
        target_language: str = 'zh-CN',
        context: Optional[str] = None,
    ) -> str:
        """
        翻译文本
        
        Args:
            text: 要翻译的文本
            source_language: 源语言代码（'auto' 表示自动检测）
            target_language: 目标语言代码
            context: 可选的上下文信息（将被转换为翻译记忆 tm_list）
        
        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return ""
        
        try:
            # 规范化语言代码
            source_lang = self._normalize_language_code(source_language)
            target_lang = self._normalize_language_code(target_language)
            
            # 构建翻译选项
            translation_options: Dict[str, Any] = {
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
            
            # 添加术语表（如果有）
            if self.terms:
                translation_options["terms"] = self.terms
            
            # 添加领域提示（如果有）
            if self.domains:
                translation_options["domains"] = self.domains
            
            # 处理上下文：将上下文转换为翻译记忆
            if context and context.strip():
                # 将上下文分割成句子，构建简单的翻译记忆
                # 这里我们将上下文作为"已翻译"的示例
                context_lines = [line.strip() for line in context.strip().split('\n') if line.strip()]
                if context_lines:
                    # 构建翻译记忆列表
                    tm_list = []
                    for line in context_lines[:3]:  # 最多使用前3行上下文
                        # 简单处理：将上下文句子作为源和目标（表示风格参考）
                        tm_list.append({
                            "source": line,
                            "target": line  # 实际应用中可能需要更智能的处理
                        })
                    if tm_list:
                        translation_options["tm_list"] = tm_list
            
            # 构建消息
            messages = [
                {"role": "user", "content": text}
            ]
            
            # 非流式调用
            if not self.stream:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body={"translation_options": translation_options}
                )
                
                if completion.choices and len(completion.choices) > 0:
                    result = completion.choices[0].message.content
                    return result.strip() if result else ""
                else:
                    return "[ERROR] 未收到翻译结果"
            
            # 流式调用（仅 qwen-mt-flash 支持增量流式）
            else:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": False},  # 不包含使用统计以提高速度
                    extra_body={"translation_options": translation_options}
                )
                
                # 收集流式输出
                result_parts = []
                for chunk in completion:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            result_parts.append(delta_content)
                
                result = ''.join(result_parts)
                return result.strip() if result else ""
        
        except Exception as e:
            error_msg = str(e)
            if 'authorization' in error_msg.lower() or 'api key' in error_msg.lower():
                return "[ERROR] DashScope API 认证失败，请检查 API Key"
            elif 'quota' in error_msg.lower() or 'limit' in error_msg.lower():
                return "[ERROR] DashScope API 配额已用尽或达到速率限制"
            else:
                return f"[ERROR] Qwen-MT API error: {error_msg}"
    
    def set_terms(self, terms: List[Dict[str, str]]):
        """
        设置术语表
        
        Args:
            terms: 术语表，格式：[{"source": "术语", "target": "翻译"}]
        """
        self.terms = terms
    
    def set_domains(self, domains: str):
        """
        设置领域提示
        
        Args:
            domains: 领域提示（英文）
        """
        self.domains = domains
    
    def set_stream(self, stream: bool):
        """
        设置是否启用流式输出
        
        Args:
            stream: 是否启用流式输出
        """
        self.stream = stream


# 测试代码
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    print("=" * 60)
    print("Qwen-MT API 翻译测试")
    print("=" * 60)
    
    # 检查 API Key
    if 'DASHSCOPE_API_KEY' not in os.environ:
        print("\n❌ 错误：未找到 DASHSCOPE_API_KEY 环境变量")
        print("\n请在 .env 文件中添加：")
        print("DASHSCOPE_API_KEY=your-dashscope-api-key")
        sys.exit(1)
    
    try:
        # 创建 API 实例
        api = QwenMTAPI(model="qwen-mt-flash")
        print(f"\n✓ Qwen-MT API 初始化成功")
        print(f"  模型: {api.model}")
        
        # 测试翻译
        test_cases = [
            ("我看到这个视频后没有笑", "auto", "English"),
            ("Hello, world!", "English", "Chinese"),
            ("This is a test.", "English", "Japanese"),
        ]
        
        print("\n" + "-" * 60)
        print("翻译测试")
        print("-" * 60)
        
        for text, source, target in test_cases:
            print(f"\n原文: {text}")
            print(f"语言: {source} → {target}")
            result = api.translate(text, source, target)
            print(f"译文: {result}")
        
        # 测试术语干预
        print("\n" + "-" * 60)
        print("术语干预测试")
        print("-" * 60)
        
        api.set_terms([
            {"source": "生物传感器", "target": "biological sensor"},
            {"source": "身体健康状况", "target": "health status of the body"}
        ])
        
        text = '而这套生物传感器运用了石墨烯这种新型材料，它的目标物是化学元素，敏锐的"嗅觉"让它能更深度、准确地体现身体健康状况。'
        print(f"\n原文: {text}")
        result = api.translate(text, "Chinese", "English")
        print(f"译文（带术语）: {result}")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
    
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
