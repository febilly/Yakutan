"""
Qwen-MT 翻译 API 使用示例

本文件展示如何使用 Qwen-MT 翻译模型进行翻译
Qwen-MT 支持 92 个语种互译，提供术语干预、翻译记忆、领域提示等高级功能
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 确保设置了 DASHSCOPE_API_KEY
if 'DASHSCOPE_API_KEY' not in os.environ:
    print("请设置 DASHSCOPE_API_KEY 环境变量")
    print("在 .env 文件中添加: DASHSCOPE_API_KEY=your_api_key")
    exit(1)

from translators.translation_apis.qwen_mt_api import QwenMTAPI

# ============================================================================
# 示例 1: 基础翻译
# ============================================================================
print("=" * 60)
print("示例 1: 基础翻译")
print("=" * 60)

# 创建 API 实例（使用默认的 qwen-mt-flash 模型）
api = QwenMTAPI()

# 中文翻译为英文
text_zh = "我看到这个视频后没有笑"
result_en = api.translate(text_zh, source_language="auto", target_language="en")
print(f"中文: {text_zh}")
print(f"英文: {result_en}")

print()

# 英文翻译为日文
text_en = "Hello, world!"
result_ja = api.translate(text_en, source_language="en", target_language="ja")
print(f"英文: {text_en}")
print(f"日文: {result_ja}")

# ============================================================================
# 示例 2: 使用不同的模型
# ============================================================================
print("\n" + "=" * 60)
print("示例 2: 使用不同的模型")
print("=" * 60)

# qwen-mt-flash: 高性价比，支持流式输出
api_flash = QwenMTAPI(model="qwen-mt-flash")
print("使用模型: qwen-mt-flash")

# qwen-mt-plus: 旗舰模型，多语言综合效果更好
# api_plus = QwenMTAPI(model="qwen-mt-plus")

# qwen-mt-turbo: 高性价比，轻量级
# api_turbo = QwenMTAPI(model="qwen-mt-turbo")

text = "这是一个测试"
result = api_flash.translate(text, source_language="zh", target_language="en")
print(f"翻译: {text} -> {result}")

# ============================================================================
# 示例 3: 术语干预（Terms）
# ============================================================================
print("\n" + "=" * 60)
print("示例 3: 术语干预")
print("=" * 60)

# 定义术语表
terms = [
    {"source": "生物传感器", "target": "biological sensor"},
    {"source": "身体健康状况", "target": "health status of the body"}
]

# 创建带术语表的 API 实例
api_with_terms = QwenMTAPI(terms=terms)

text = '而这套生物传感器运用了石墨烯这种新型材料，它的目标物是化学元素，敏锐的"嗅觉"让它能更深度、准确地体现身体健康状况。'
result = api_with_terms.translate(text, source_language="zh", target_language="en")

print(f"原文: {text}")
print(f"译文: {result}")
print("\n注意：术语 '生物传感器' 和 '身体健康状况' 按照术语表翻译")

# ============================================================================
# 示例 4: 领域提示（Domains）
# ============================================================================
print("\n" + "=" * 60)
print("示例 4: 领域提示")
print("=" * 60)

# IT 领域翻译
domain_hint = (
    "The sentence is from Ali Cloud IT domain. It mainly involves computer-related "
    "software development and usage methods, including many terms related to computer "
    "software and hardware. Pay attention to professional troubleshooting terminologies "
    "and sentence patterns when translating. Translate into this IT domain style."
)

api_with_domain = QwenMTAPI(domains=domain_hint)

text = "第二个SELECT语句返回一个数字，表示在没有LIMIT子句的情况下，第一个SELECT语句返回了多少行。"
result = api_with_domain.translate(text, source_language="zh", target_language="en")

print(f"原文: {text}")
print(f"译文: {result}")
print("\n注意：翻译风格符合 IT 领域的专业术语和表达习惯")

# ============================================================================
# 示例 5: 流式输出（仅 qwen-mt-flash 支持）
# ============================================================================
print("\n" + "=" * 60)
print("示例 5: 流式输出")
print("=" * 60)

# 启用流式输出
api_stream = QwenMTAPI(model="qwen-mt-flash", stream=True)

text = "这是一段较长的文本，流式输出可以实时返回翻译内容，减少用户等待时间。"
result = api_stream.translate(text, source_language="zh", target_language="en")

print(f"原文: {text}")
print(f"译文: {result}")
print("\n注意：流式输出在实际应用中可以实时显示翻译进度")

# ============================================================================
# 示例 6: 上下文感知翻译（通过 ContextAwareTranslator）
# ============================================================================
print("\n" + "=" * 60)
print("示例 6: 上下文感知翻译")
print("=" * 60)

from translators.context_aware_translator import ContextAwareTranslator

# 创建上下文感知翻译器
api = QwenMTAPI()
translator = ContextAwareTranslator(
    translation_api=api,
    max_context_size=6,
    target_language='en',
    context_aware=True
)

# 翻译多个句子，建立上下文
sentences = [
    "我今天去了超市。",
    "买了一些水果和蔬菜。",
    "回家后做了晚饭。",
]

for sentence in sentences:
    result = translator.translate(sentence, source_language="zh")
    print(f"中文: {sentence}")
    print(f"英文: {result}\n")

print("注意：后面的句子翻译会考虑前面的上下文，使翻译更连贯")

# ============================================================================
# 示例 7: 支持的语言
# ============================================================================
print("\n" + "=" * 60)
print("示例 7: 多语言支持")
print("=" * 60)

api = QwenMTAPI()

# Qwen-MT 支持 92 个语种
multilingual_examples = [
    ("你好", "zh", "en", "中文->英文"),
    ("Hello", "en", "ja", "英文->日文"),
    ("こんにちは", "ja", "ko", "日文->韩文"),
    ("Bonjour", "fr", "es", "法文->西班牙文"),
]

for text, source, target, desc in multilingual_examples:
    result = api.translate(text, source_language=source, target_language=target)
    print(f"{desc}: {text} -> {result}")

print("\n" + "=" * 60)
print("示例完成")
print("=" * 60)
print("\n更多信息请参考:")
print("- Qwen-MT 文档: https://help.aliyun.com/zh/model-studio/qwen-mt")
print("- 支持的语言列表请查看 qwen_mt_api.py 中的 LANGUAGE_MAP")
