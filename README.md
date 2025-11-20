# Yakutan

更适合中国 [~~计科~~](#局限性) 宝宝体质的 VRChat 语音翻译器（翻译你自己的声音）

<div align="center">
    <img src="images/screenshot.png" alt="A Screenshot of the WebUI of Yakutan" style="max-width: 100%; width: 512px; height: auto;">
</div>

- 使用阿里的 [Qwen3实时语音识别（默认）](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash) 或 [Fun-ASR](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/fun-asr-realtime) 进行语音转文本
- 支持多种翻译引擎：
    - DeepL（默认）- 原生支持上下文
    - **Qwen-MT** - 阿里云机器翻译模型，支持92个语种互译
    - Google 翻译 - 开箱即用
    - OpenRouter（大模型）- 支持多种 LLM
- 将结果通过 OSC 发送至游戏
- 有一些 [独特的功能](#特点)
- ~~其实就是把一堆 API 粘在了一起~~

## 快速开始

1. 在 [Release](https://github.com/febilly/Yakutan/releases) 中下载最新的 exe
2. 将 exe 放在一个空文件夹中并运行，WebUI 应该会自动打开
3. 根据 WebUI 中 `API Keys 配置` 面板的说明获取并填入 API Keys
4. 点击 `启动服务` 按钮

## 不是已经有人做过了吗，为什么要再做一个翻译器？？？

目前，语音翻译最大的短板在语音识别上。而现有的给 VRChat 做的翻译器识别中文及带中文口音的英语的效果并不好。
_可以说我就是为了这点醋包的这顿饺子_

### 语音识别方面
- 准确性方面：
    - 断句断不准是最致命的问题，这个解决不了的话其他都白干
    - Whisper识别汉语效果实在是一坨
    - Edge的WebSpeech面对中国人口音识别效果不好
    - VRChat里经常出现一些一般的语音识别认不出来的词，需要用热词功能提升识别效果
- 一些细节的优化问题：
    - VAD断句需要一两秒的时间来等待说话结束
    - 闭麦时可能会漏掉用户说的最后一个字

### 翻译方面
- 翻译需要上下文
    - 没有上下文的话，比如看这句话被翻译成了啥：
        - 现在总行 _(xíng)_ 了吧？
        - Is the head office now?

## 特点

### 语音识别方面

- 准确性：
    - 使用阿里的 Qwen3 或者 Fun-ASR：我试了好几个 STT 的 API，感觉阿里这个是挺好的，以及他有给免费额度
        - 但我对比的大部分都是国外的 API，感觉不是很公平...... 有更好的 API 可以跟我说一声！
    - 增加了热词词库
        - 自带一部分公共的词库
            - 部分比较， _咳咳，不太好_ 的词被我删掉了，请自行添加
        - 可自己添加私人的词库
- 断句：
    - 游戏内语音模式请使用 toggle 模式。说完一句话后，按下静音键，即视为一句话说完，马上全部进行转录。这样能提高响应速度。
        - 停止录制时会额外继续录制一小段音频（默认0.3s），防止漏掉最后一个字
    - VAD：仍然有 VAD 作为补充断句方法

### 翻译方面

- 实现了翻译的上下文，默认附带一条简短的场景说明，和最近的6条消息
- 附带备用语言选项，如果识别到的源语言和主目标语言相同，则翻译至备用语言
    - 可以实现两种语言之间的互译
- 默认使用Deepl翻译
    - 可以指定翻译的正式程度（比如对于日语来说）
    - 原生支持上下文
    - 可以自定义词库（本项目还没实现）
- 可以切换为使用 **Qwen-MT 机器翻译模型**
    - 阿里云出品，支持 92 个语种互译
    - 支持术语干预（自定义术语表）
    - 支持翻译记忆（参考历史翻译风格）
    - 支持领域提示（如法律、IT等专业领域）
    - 使用与语音识别相同的 API Key，无需额外配置
- 可以切换为开箱即用的谷歌翻译（但有网络连通性问题，及速率限制）
- 可以切换为使用大模型进行翻译，但由于延迟问题，默认不使用

## 局限性
- ~~你得会配Python环境~~ **已提供打包好的可执行文件**
- ~~目前没（懒得）写 GUI，所有配置需要在 `config.py` 文件中修改~~ **现在有Web UI了**
- 使用脚本启动时系统的默认麦克风
- 使用完毕后请**一定记得关闭程序（命令行界面，不只是重启识别服务）**，否则可能会持续使用转录 API，产生额外费用
- 需要用商业服务的API Key，有一定免费额度，但免费额度用完后需要付钱
    - 阿里云的免费额度是一次性的，但是 [大学生可以拿到每年的免费额度](https://university.aliyun.com/buycenter/)
    - DeepL的免费额度每月重置，但是怎么拿到Key需要自己想办法
        - 实在懒得折腾可以把翻译器换成谷歌的
- 目前暂不支持和其他 OSC 程序同时运行
- 语言识别默认使用一个简单的中日韩英检测器
    - 如需其他语言，请自行修改配置

## 命令行运行（高级）

一般用户不需要此步骤，直接使用打包好的可执行文件即可。

<details>

### 1. 克隆项目

```bash
git clone https://github.com/febilly/Yakutan
cd Yakutan
```

### 2. 创建虚拟环境（推荐，非必须）

```bash
python -m venv .
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录创建 `.env` 文件，添加以下内容：

```env
# 必需：阿里云百炼 API Key
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# 可选：DeepL API Key（如果使用默认的 DeepL 翻译）
DEEPL_API_KEY=your_deepl_api_key_here

# 可选：OpenRouter API Key（如果使用 OpenRouter 进行翻译）
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 5. 修改配置（可选）

编辑 `config.py` 文件，根据需求调整配置：

### 6. 运行程序

```bash
python main.py
```

使用此方法时，需要手动在 `config.py` 中修改配置。

</details>

## API Key 获取

- **阿里云百炼（必需）**：https://bailian.console.aliyun.com/?tab=model#/model-market/detail/fun-asr-realtime
  - 用于语音识别（必需）
  - **同时支持 Qwen-MT 翻译**（可选，使用相同的 API Key）
  - 注册后可获得免费额度
  - 大学生可申请每年的免费额度
  
- **DeepL（可选）**：https://www.deepl.com/en/pro-api
  - 每月有 500,000 字符的免费额度
  - 需要使用特殊方式获取 API Key
  
- **OpenRouter（可选）**：https://openrouter.ai/
  - 提供多种 LLM 模型
  - 部分模型有免费额度（如 Gemini）

## 翻译引擎配置

本项目支持多种翻译引擎，可在 WebUI 或 `config.py` 中切换。

### Qwen-MT 翻译配置（推荐）

Qwen-MT 是阿里云的机器翻译模型，**使用与语音识别相同的 API Key**，无需额外配置。

**优势：**
- 支持 92 个语种互译（中、英、日、韩、法、西、德、泰、印尼、越、阿等）
- 支持术语干预、翻译记忆、领域提示等高级功能
- 与语音识别共享免费额度，无需额外付费
- 翻译质量高，速度快

**在 `config.py` 中配置：**

```python
# 设置翻译 API 类型为 qwen_mt
TRANSLATION_API_TYPE = 'qwen_mt'

# Qwen-MT 配置（可选，使用默认值即可）
QWEN_MT_MODEL = 'qwen-mt-flash'  # 可选: qwen-mt-flash, qwen-mt-plus, qwen-mt-turbo
QWEN_MT_STREAM = False  # 是否启用流式输出（仅 qwen-mt-flash 支持）

# 高级功能（可选）
QWEN_MT_TERMS = []  # 术语表，如: [{"source": "专有名词", "target": "Proper Noun"}]
QWEN_MT_DOMAINS = None  # 领域提示，如: "IT domain translation"
```

**模型选择：**
- `qwen-mt-flash`：推荐，高性价比，支持流式输出
- `qwen-mt-plus`：旗舰模型，多语言效果更好
- `qwen-mt-turbo`：轻量级，速度快

**示例代码：** 参见 `examples/qwen_mt_usage.py`

### 其他翻译引擎

<details>
<summary>DeepL（默认）</summary>

```python
TRANSLATION_API_TYPE = 'deepl'
```

需要单独申请 DeepL API Key。
</details>

<details>
<summary>Google 翻译</summary>

```python
TRANSLATION_API_TYPE = 'google_web'  # 或 'google_dictionary'
```

开箱即用，但可能有网络连通性问题。
</details>

<details>
<summary>OpenRouter（大模型）</summary>

```python
TRANSLATION_API_TYPE = 'openrouter'
OPENROUTER_TRANSLATION_MODEL = 'google/gemini-2.5-flash:nitro'
```

需要单独申请 OpenRouter API Key，延迟较高。
</details>

## 热词配置

热词功能可以显著提高特定词汇的识别准确度，特别适合专业术语、人名、地名等。
以及某些 VRChat 的 _特殊_ 词汇

<details>

### 热词文件结构

```
可执行文件或脚本所在文件夹/
└── hot_words_private/  # 私人热词目录（不会被提交到 Git）
    ├── zh-cn.txt      # 中文私人热词
    ├── en.txt         # 英文私人热词
    └── ...
```

### 热词文件格式

每个热词文件是纯文本格式，每行一个词

**注意事项：**
- 每行一个热词，不要有多余空格
- 空行会被忽略
- 总热词数量不超过 500 个（阿里云限制）

### 如何设置私人热词

- **编辑私人热词文件**

   打开 `hot_words_private/` 目录下对应语言的文件（如不存在则请手动创建）：
   例如：

   ```
   hot_words_private/zh-cn.txt
   hot_words_private/en.txt
   ```

- **启用的语言配置**

   在 `hot_words_manager.py` 中配置要加载的语言：
   
   ```python
   # 要加载的语言列表
   ENABLED_LANGUAGES = ['zh-cn', 'en']  
   # 可添加更多：['zh-cn', 'en', 'ja', 'ko']
   ```

</details>

## VRChat OSC 配置

### 启用 OSC

1. 启动 VRChat
2. 打开快捷菜单（Action Menu）
3. 进入 Options → OSC
4. 点击 "Enable" 启用 OSC

## 常见问题

<details>

### 1. 没有任何转录

- 检查系统的默认麦克风是否为你在用的麦克风
  - 切换系统默认麦克风后，**程序需要重启（命令行界面，不只是重启识别服务）**，才能识别到新的麦克风
- 检查麦克风是否有声音输入
- 检查 `ENABLE_MIC_CONTROL` 配置：
  - 如果为 `True`，需要在 VRChat 中打开麦克风才能开始识别
  - 如果为 `False`，程序启动后会立即开始识别

### 2. VRChat 聊天框没有显示

- 确认 VRChat OSC 已启用
- 如果修改了 OSC 端口，请在 `config.py` 中同步修改 `OSC_PORT` 配置

### 3. WebSocket 连接经常断开

- 调整 `KEEPALIVE_INTERVAL` 参数（建议 30-60 秒）
- 检查网络连接稳定性
- 系统会自动尝试重连，通常不需要手动干预

### 4. 翻译延迟较高

- 如果使用 OpenRouter API，延迟会比较明显
- 建议使用 DeepL 或 Google API 以获得更快的响应速度
- 可以设置 `ENABLE_TRANSLATION = False` 禁用翻译，直接输出识别结果

### 5. 配置修改后没有生效

- 确保修改的是 `config.py` 文件
- 重启程序以使配置生效
- 检查是否有语法错误（Python 对缩进敏感）

</details>

## 附录

- 要翻译别人的声音的话建议用 [soniox](https://console.soniox.com/org/e784abf7-3ab5-4127-8823-ecfc18f68b90/projects/2b220fdd-f158-4b7a-9b12-447947b5098a/playground/speech-to-text/)，用它的网页端 Playground 就行，配合 Powertoys 的窗口裁剪器
    - 也可以试试 [LiveCaptions Translator](https://github.com/SakiRinn/LiveCaptions-Translator)
- 我还没太试过国内其他家的识别服务效果怎样，如果有更好的（并且有不少免费额度的）请告诉我谢谢

## 致谢
- 本项目部分基于阿里给的 Fun-ASR 示例代码
- 快速的 Google Translate API 来自 https://github.com/SakiRinn/LiveCaptions-Translator