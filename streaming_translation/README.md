# StreamingTranslation

多后端、上下文感知的流式翻译库。支持 DeepL、Google 翻译、阿里 Qwen-MT 以及任意 OpenAI 兼容的 LLM 翻译 API。

## 安装

```bash
pip install streaming-translation[all]

# 或按需安装：
pip install streaming-translation[deepl]      # DeepL
pip install streaming-translation[google]     # Google 翻译
pip install streaming-translation[llm]        # LLM 翻译（OpenAI 兼容接口）
```

## 快速开始

```python
from streaming_translation import (
    TranslationConfig,
    DeepLAPI,
    ContextAwareTranslator,
)

cfg = TranslationConfig(
    target_language="ja",
    context_prefix="VRChat conversation:",
    deepl_api_key="your-key",
)

api = DeepLAPI(api_key=cfg.deepl_api_key, proxy_url=cfg.proxy_url)

translator = ContextAwareTranslator(
    translation_api=api,
    max_context_size=6,
    target_language=cfg.target_language,
)

result = translator.translate("Hello, how are you?")
```

## 架构

```
streaming_translation/
├── __init__.py             # 公共 API 导出
├── _config.py              # TranslationConfig
├── _proxy.py               # 代理检测
├── api/
│   ├── base.py             # BaseTranslationAPI
│   ├── deepl.py            # DeepL
│   ├── google_web.py       # Google Web
│   ├── google_dictionary.py# Google Dictionary
│   ├── qwen_mt.py          # Qwen-MT
│   └── openrouter.py       # LLM (OpenAI-compatible)
├── core/
│   ├── context_aware.py    # ContextAwareTranslator
│   └── smart_language.py   # 智能目标语言选择器
└── pipeline.py             # 翻译管道、工厂、编排函数
```

## API

### TranslationConfig

所有配置集中在一个数据类中，替代全局变量模式。

```python
from streaming_translation import TranslationConfig

cfg = TranslationConfig(
    target_language="ja",
    translation_api_type="deepl",
    context_prefix="Voice chat:",
    proxy_url="http://127.0.0.1:7890",
)
```

完整字段：

| 分组 | 字段 | 默认值 | 说明 |
|---|---|---|---|
| **源/目标** | `source_language` | `"auto"` | 源语言 |
| | `target_language` | `"ja"` | 主目标语言 |
| | `secondary_target_language` | `None` | 第二目标语言 |
| | `fallback_language` | `"en"` | 源=目标时的回退语言 |
| **API 选择** | `translation_api_type` | `"qwen_mt"` | 后端类型 |
| | `translate_partial_results` | `False` | 是否翻译中间结果 |
| | `context_prefix` | `""` | 场景上下文描述 |
| | `translation_context_size` | `6` | 上下文窗口大小 |
| | `translation_context_aware` | `True` | 是否启用上下文 |
| **LLM** | `llm_base_url` | `""` | OpenAI 兼容端点 |
| | `llm_model` | `""` | 模型名称 |
| | `llm_temperature` | `0.2` | 采样温度 |
| | `llm_timeout` | `30` | 超时（秒） |
| | `llm_max_retries` | `3` | 重试次数 |
| | `llm_formality` | `"medium"` | 礼貌程度 |
| | `llm_style` | `"light"` | 语气风格 |
| | `llm_extra_body_json` | `""` | 额外请求体 |
| | `llm_parallel_fastest_mode` | `"off"` | 最快响应模式 |
| **智能目标** | `smart_target_primary_enabled` | `False` | 启用主智能选择 |
| | `smart_target_secondary_enabled` | `False` | 启用副智能选择 |
| | `smart_target_strategy` | `"most_common"` | 策略 |
| | `smart_target_window_size` | `5` | 历史窗口 |
| | `smart_target_exclude_self` | `True` | 排除自身语言 |
| | `smart_target_fallback` | `"en"` | 回退语言 |
| | `smart_target_min_samples` | `3` | 最少样本数 |
| | `smart_target_count` | `2` | 返回数量 |
| | `smart_target_manual_secondary` | `None` | 手动第二语言 |
| **Qwen-MT** | `use_international_endpoint` | `False` | 国际版端点 |
| **代理** | `proxy_url` | `None` | HTTP 代理 |
| **API keys** | `deepl_api_key` | `None` | DeepL |
| | `dashscope_api_key` | `None` | DashScope |
| | `llm_api_key` | `None` | LLM |
| | `openai_api_key` | `None` | OpenAI |
| **后端覆写** | `deepl_formality` | `"default"` | DeepL 礼貌度 |

`config_from_module(module)` 可将模块对象的属性映射为 `TranslationConfig`：

```python
from streaming_translation import config_from_module
import config  # 宿主应用的 config 模块
cfg = config_from_module(config)
```

### 翻译后端

所有后端实现 `BaseTranslationAPI` 接口：

| 类 | 后端 | 上下文支持 | 可选依赖 |
|---|---|---|---|
| `DeepLAPI` | DeepL 官方 API | ✅ | `deepl` |
| `GoogleWebAPI` | Google 网页翻译 | ❌ | `googletrans` |
| `GoogleDictionaryAPI` | Google 字典 API | ❌ | `aiohttp` |
| `QwenMTAPI` | 阿里 Qwen-MT | ✅ | `openai` |
| `OpenRouterAPI` | OpenAI 兼容接口 | ✅ | `openai` |
| `OpenRouterStreamingAPI` | 同上（流式模式） | ✅ | `openai` |

`OpenRouterAPI` 在流式模式下使用 v12 smart-hybrid 策略，
`merge_with_draft(fresh, draft)` 是其公开的合并工具函数，
用于保留 draft 的开头措辞同时用 fresh 保证内容完整性。

### ContextAwareTranslator

为翻译 API 提供上下文感知包装：

- 维护滑动窗口，保留最近 N 条翻译历史
- 原生支持上下文的 API（DeepL、Qwen-MT、LLM）：传递完整原文-译文对
- 不支持上下文的 API（Google）：使用标记法模拟上下文
- 线程安全（内部 RLock）

```python
translator = ContextAwareTranslator(
    translation_api=some_api,
    max_context_size=6,
    target_language="ja",
    context_aware=True,
)
result = translator.translate(
    "Hello",
    source_language="en",
    target_language="ja",
    context_prefix="Voice chat:",
    is_partial=False,
    record_history=True,
)
```

### 翻译管道（pipeline.py）

后端注册表、工厂方法和高阶编排函数。

```python
from streaming_translation import (
    TranslationConfig,
    reinitialize_translator,
    translate_with_backend,
    reverse_translation,
)

cfg = TranslationConfig(target_language="ja")

# 初始化主/副/反向翻译器到 state 对象
reinitialize_translator(state, cfg)

# 用指定翻译器执行翻译，可配置 DeepL 优先
result = translate_with_backend(
    translator, deepl_fallback, text, target_lang,
    previous_translation="...",
    prefer_deepl=False,
    source_language="auto",
    context_prefix=cfg.context_prefix,
    record_history=True,
)

# 反向翻译（用于显示验证）
reverse = reverse_translation(backwards_bt, translated, source_lang, target_lang)
```

可用函数：

| 函数 | 说明 |
|---|---|
| `reinitialize_translator(state, cfg)` | （重新）初始化所有翻译器 |
| `update_secondary_translator(state, cfg)` | 按配置更新第二翻译器 |
| `ensure_secondary_translator(state, lang, cfg)` | 确保第二翻译器存在 |
| `translate_with_backend(...)` | 执行翻译 |
| `reverse_translation(bt, text, src, tgt)` | 反向翻译 |
| `is_streaming_translation_mode(api_type)` | 是否为流式模式 |
| `is_streaming_deepl_hybrid_mode(api_type)` | 是否为混合模式 |

### SmartTargetLanguageSelector

根据最近接收到的外语语音历史，自动推断翻译目标语言。

```python
from streaming_translation import SmartTargetLanguageSelector, TranslationConfig

cfg = TranslationConfig(
    smart_target_primary_enabled=True,
    smart_target_strategy="most_common",
    smart_target_window_size=10,
)
selector = SmartTargetLanguageSelector(cfg)
selector.record_language("ja")
selector.record_language("ja")
selector.record_language("en")
targets = selector.select_target_language(self_language="zh-CN")
# targets -> ["ja", "en"]
```

支持三种策略：
- `most_common`：按出现频率排序（频率相同时按最近出现时间）
- `latest`：只返回最近检测到的语言
- `weighted`：按时间衰减加权（越近权重越高）

## 设计要点

| 设计决策 | 说明 |
|---|---|
| `TranslationConfig` 数据类 | 所有选项显式注入，不依赖全局变量或环境变量 |
| `proxy_url` 作为参数 | 代理配置由调用方控制，库不自行探测系统代理 |
| API key 由调用方传入 | 库不读取环境变量，密钥管理完全交给上层 |
| 可选依赖 | 每个后端都是可选安装，不会引入无用依赖 |

## 可选依赖

```bash
pip install streaming-translation[deepl]   # DeepLAPI
pip install streaming-translation[google]  # GoogleWebAPI
pip install streaming-translation[llm]     # OpenRouterAPI, QwenMTAPI
pip install streaming-translation[all]     # 所有后端
```
