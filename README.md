# Yakutan

更适合中国宝宝体质的 VRChat 语音翻译器。它负责识别并翻译你自己的声音，然后通过 OSC 把字幕发到 VRChat 聊天框。

<div align="center">
    <img src="images/screenshot.png" alt="A Screenshot of the WebUI of Yakutan" style="max-width: 100%; width: 512px; height: auto;">
</div>

## 这是什么

Yakutan 不是一个只会把语音识别结果丢给翻译 API 的小脚本。它现在更像一个面向 VRChat 语音交流场景定制的字幕工作台：

- **边说边翻译**：支持 LLM 流式翻译，别人不用等你整句话说完才看到意思。
- **中文和中式口音优先**：默认使用阿里 Qwen3 实时语音识别，也可切换 Fun-ASR、Soniox、豆包录音文件识别或本地识别。
- **懂 VRChat 的上下文**：可带上最近对话、场景说明、VRCX 当前世界/玩家信息、术语表和热词，减少人名、地名、梗词翻车。
- **为游戏聊天框做过专门处理**：OSCQuery、聊天框长度裁剪、双闭麦清空、发送节奏、Typing 状态、RTL 文本重排都不是临时拼的。
- **不想调参也能用**：WebUI 有简易/高级模式，常用配置自动保存，打包版开箱即用。

~~其实就是把一堆 API 粘在了一起，但粘得越来越认真了。~~

## 快速开始

1. 在 [Releases](https://github.com/febilly/Yakutan/releases/latest) 下载最新版：
   - `Yakutan-*.exe`：标准版，适合大多数在线 ASR / 在线翻译用户。
   - `Yakutan-LocalASR-*.exe`：本地识别版，适合想试 SenseVoice / Qwen3-ASR 本地模型的用户。
2. 把 exe 放进一个空文件夹后运行，WebUI 会自动打开。
3. 在 `API Keys 配置` 里填入 DashScope Key；如果要用 DeepL、LLM、Soniox、豆包等后端，再填对应 Key。
4. 在 VRChat 的圆盘菜单中进入 `Options -> OSC`，打开 OSC。
5. 回到 WebUI，选择识别/翻译配置，点击 `启动服务`。

一般建议先用默认配置跑通：`Qwen3 ASR + Qwen-MT + 游戏静音时暂停转录`。确认 VRChat 聊天框能显示后，再尝试 LLM 流式翻译、本地识别、VRCX 上下文等高级功能。

## 核心能力

### 语音识别

- **Qwen3 实时 ASR（默认推荐）**：适合中文、日语、英语等多语混合场景。
- **Fun-ASR**：大陆版 DashScope 可用，作为稳定备用选择。
- **Soniox**：对多语混讲支持较好，需要 Soniox API Key。
- **豆包录音文件识别**：关麦后返回整段识别结果，适合不追求即时中间结果的场景。
- **本地识别**：支持 SenseVoice Small 和 Qwen3-ASR，本地模型可在 WebUI 中检查/下载；标准版可能禁用本地 ASR，请使用 LocalASR 版本或源码安装相关依赖。
- **统一 VAD**：同一组 VAD 设置同时服务于在线 API 省流门控和本地识别分段；在线后端静音时可停止发送无效音频帧，本地后端仍保持连续音频分段。
- **热词词库**：内置公共热词，并支持 `hot_words_private/` 私人热词目录，适合 VRChat 人名、世界名、梗词和专业术语。

### 低延迟断句

- **闭麦即终句**：推荐 VRChat 使用 Toggle 麦克风模式；说完一句后闭麦，Yakutan 会马上收尾并触发最终识别/翻译。
- **尾音保护**：闭麦后默认额外收 0.2 秒音频，减少漏掉最后一个字。
- **VAD 兜底**：不开麦控或忘记闭麦时，仍可通过 VAD 自动判断语音段落。
- **双闭麦清空聊天框**：短时间内连续两次闭麦会发送空消息清空 VRChat 聊天框，并丢弃迟到的识别/翻译结果，避免撤回后又被刷回来。
- **Typing 状态优化**：正在说话时只在状态变化时发送 typing，最终消息发出后再延迟关闭，减少 VRChat 聊天框显示抖动。

### 翻译

- **Qwen-MT（默认）**：使用 DashScope Key，支持上下文和术语记忆。
- **DeepL**：质量高、延迟低，支持正式程度配置。
- **Google Dictionary / Google Web**：可作为免费或备用翻译选项，但受网络连通性和速率限制影响。
- **OpenAI 兼容 LLM**：可接 OpenRouter、DashScope 兼容模式、DeepSeek 或其他兼容 `/chat/completions` 的服务。
- **LLM 流式翻译**：中间结果就能开始翻译，适合减少对方等待时间。
- **LLM 流式 + DeepL 终译混合模式**：中间过程用 LLM 快速出草稿，终译可按更新次数切回 DeepL，减少最终译文大幅跳变。
- **并行双发取最快**：可对终译或所有 LLM 请求双发，取先返回的一路结果，代价是更多 token 用量。
- **上下文感知翻译**：默认携带 VRChat 场景提示和最近 6 条对话；普通后端默认只给原文历史，Qwen-MT 使用官方 `tm_list` 传递翻译记忆。
- **第二输出语言**：可同时输出两种目标语言。
- **备用语言**：当识别到的源语言和目标语言相同时，自动切到备用语言，适合中英/中日双向交流。
- **智能目标语言**：可根据最近收到的外部语音语言自动推断主目标语言或第二目标语言，支持最高频、最新、权重衰减等策略。
- **反向翻译**：可把译文再翻回源语言，在小面板中辅助检查翻译是否偏了。

### VRChat / OSC

- **OSCQuery 优先**：默认不再和其他 OSC 程序抢固定监听端口。
- **兼容模式**：可切换为固定端口监听，用于 Resonite 等兼容 OSC 的游戏；在 VRChat 中通常不需要打开。
- **启动前端口检查**：会检查 VRChat OSC 目标 UDP 端口是否被非 VRChat 程序占用。
- **VRChat 未监听提醒**：如果没检测到 VRChat 正在监听目标端口，会提示去游戏内 `Options -> OSC` 打开开关。
- **目标端口可配置**：默认发往 UDP 9000，也可在高级设置中调整。
- **错误消息不默认发进游戏**：错误会显示在小面板，是否同步发到 OSC 可单独配置。

### 文本后处理

- **VRChat 长度裁剪**：默认按 VRChat 聊天框上限保留最新内容，并尽量在标点或空白处分段裁剪。
- **显示格式控制**：可选择是否显示原文、语言标签、句尾句号。
- **日语假名 / 中文拼音**：可为日语汉字加假名，为中文加带声调拼音。
- **阿拉伯文 / 希伯来文重排**：让 RTL 文字在 VRChat 中更接近正确显示。
- **Unicode 文本风格**：支持 small caps、curly、magic 等 fancify-text 风格。

### WebUI 和小面板

- **简易/高级模式**：简易模式只放启动、Key、识别、翻译、语言、麦克风等核心控件；高级模式展开完整参数。
- **多语言界面**：WebUI 支持简体中文、English、日本語、한국어，并会记住用户选择。
- **小面板**：独立窗口显示识别、翻译、错误和反向翻译信息；可设置宽度和悬浮窗模式。
- **快捷语言按钮**：可在小面板底部配置 4 个语言切换按钮。
- **配置自动保存**：大部分 UI 设置会保存在浏览器本地，常用模板和 API Key 来源也会自动记住。
- **LLM 模板**：内置阿里 Qwen、DeepSeek、OpenRouter、LongCat、Mercury 2 等模板，并支持多个自定义模板。

### VRCX 上下文桥接

如果你使用 VRCX，可以在 WebUI 中复制上下文脚本到 VRCX 控制台运行。Yakutan 会接收当前世界、玩家名等本地上下文，用于：

- 帮助 ASR 识别玩家名、世界名和常见专有名词。
- 帮助翻译器理解当前场景，减少把名字翻译错、把梗当普通句子的情况。
- 在状态区显示上下文是否已接收、是否过期。

## 为什么要做这个

目前 VRChat 语音翻译最大的短板通常不是“有没有翻译 API”，而是识别、断句、上下文和聊天框表现：

- 普通 Whisper / WebSpeech 对中文和中式口音英语不够友好。
- VRChat 里有大量普通模型没见过的人名、世界名、梗词和混合语。
- 等整句话结束再翻译，交流会明显慢半拍。
- 自动断句太慢会卡，太快又会切碎句子。
- 游戏聊天框有长度、发送节奏、OSC 状态和显示方向等一堆边界问题。

Yakutan 的目标就是把这些 VRChat 现场问题一个个补上，而不是只做一个“语音识别 -> 翻译 -> OSC”的演示。

## API Key 获取

- **阿里云百炼 / DashScope（常用，Qwen3 ASR、Fun-ASR、Qwen-MT 需要）**：https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash
  - 注册后通常有免费额度。
  - 大学生可关注阿里云高校权益。

- **DeepL（可选，用于 DeepL 翻译）**：https://www.deepl.com/en/pro-api
  - 每月有免费字符额度，具体规则以 DeepL 官方页面为准。

- **OpenRouter / 其他 OpenAI 兼容服务（可选，用于 LLM 翻译）**：https://openrouter.ai/
  - 也可以填 DashScope 兼容模式、DeepSeek 等兼容接口地址。

- **Soniox（可选，用于 Soniox ASR）**：https://soniox.com/

- **豆包（可选，用于豆包录音文件识别）**：请在火山引擎控制台获取对应 Key。

## 热词配置

热词可以显著提升特定词汇的识别准确度，尤其适合人名、地名、VRChat 世界名、团体名、专业术语和一些普通模型不太认识的词。

<details>

### 文件结构

```text
可执行文件或脚本所在文件夹/
└── hot_words_private/
    ├── zh-cn.txt
    ├── en.txt
    └── ...
```

### 文件格式

每个热词文件是纯文本格式，每行一个词：

```text
Yakutan
某个玩家名
某个世界名
```

注意：

- 每行一个热词，不要有多余空格。
- 空行会被忽略。
- 总热词数量不要超过服务商限制；DashScope 热词表有数量上限。
- 公共热词放在 `hot_words/`，私人热词放在 `hot_words_private/`，后者不会提交到 Git。

</details>

## VRChat OSC 配置

1. 启动 VRChat。
2. 打开圆盘菜单。
3. 进入 `Options -> OSC`。
4. 点击 `Enable` 启用 OSC。

如果 Yakutan 启动后提示未检测到 VRChat 正在监听 UDP 端口，请确认游戏已经启动并打开 OSC。默认端口是 UDP 9000；只有你明确知道自己改过端口时，才需要在高级设置里调整 `OSC 发送目标端口`。

## 命令行运行（高级）

普通用户建议直接使用打包好的 exe。需要从源码运行时可参考下面步骤。

<details>

### 1. 克隆项目

```bash
git clone https://github.com/febilly/Yakutan
cd Yakutan
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果要使用本地识别：

```bash
pip install -r requirements-local-asr.txt
```

### 4. 运行

```bash
python run_ui.py
```

</details>

## 常见问题

<details>

### 1. 没有任何转录

- 检查 WebUI 中选择的麦克风是否正确。
- 检查麦克风是否有声音输入。
- 如果开启了 `游戏静音时暂停转录`，需要先在 VRChat 中从闭麦切到开麦，程序才会开始识别。
- 如果你想让程序启动后立刻识别，请关闭 `游戏静音时暂停转录`。

### 2. VRChat 聊天框没有显示

- 确认 VRChat 已启动，并且 `Options -> OSC` 已启用。
- 看 WebUI 是否提示 VRChat 没有监听 UDP 端口。
- 如果你在用加速器，请避免使用“进程模式”。
- 检查是否有除 VRChat 之外的程序占用了目标 UDP 端口。
- 如果你打开了兼容模式，确认端口设置和游戏实际监听端口一致。

### 3. 识别不到游戏内麦克风开关

- 确认 VRChat OSC 已启用。
- 重启 VRChat 或电脑通常可以恢复 OSC 状态。
- 临时解决可以关闭 `游戏静音时暂停转录`，让程序持续识别。

### 4. 翻译延迟较高

- 优先尝试 Qwen-MT、DeepL 或 Google Dictionary。
- 如果使用 LLM，换响应更快的模型，或检查是否启用了思考模式。
- 可尝试 LLM 流式翻译，但它会增加 token 用量。
- 如果终译抖动明显，可以尝试 `LLM 流式 + DeepL 终译` 混合模式。

### 5. 本地识别不可用

- 标准版可能禁用本地 ASR，请使用 `Yakutan-LocalASR` 版本。
- 首次使用本地 ASR 前，需要在 WebUI 下载对应模型和运行时。
- 源码运行时请安装 `requirements-local-asr.txt`。
- Qwen3-ASR 本地模式对显存/驱动有要求，SenseVoice Small 更适合先跑通。

### 6. 阿里语音识别报错

- 中国大陆版请确认账号已实名并开通对应模型。
- 国际版请确认使用国际端点，并满足账号绑定要求。
- Qwen 和 Fun-ASR 都需要 DashScope API Key。

### 7. 为什么使用完要停止服务

如果服务持续运行，可能持续占用麦克风和 ASR 会话，产生额外 API 用量。离开 VRChat 或不用翻译时，请点击 `停止服务`。

</details>

## 附录

- Yakutan 主要翻译你自己的声音。如果你想翻译别人说的话，可以试试 [realtime-subtitle](https://github.com/febilly/realtime-subtitle/releases)。
- 也可以试试免费的 [LiveCaptions Translator](https://github.com/SakiRinn/LiveCaptions-Translator)。
- 以及被我狠狠地嫖了代码的 [LiveTranslate](https://github.com/TheDeathDragon/LiveTranslate)。
- 如果你知道更适合 VRChat 的中文/多语 ASR 或翻译服务，欢迎提 issue 或 PR。

## 致谢

- 本项目部分基于阿里给的 Fun-ASR 示例代码。
- 快速的 Google Translate API 来自 https://github.com/SakiRinn/LiveCaptions-Translator
- 提示词少量参考了 https://github.com/kapitalismho/PuriPuly-heart
- 本地 ASR 相关代码与模型生态参考/使用了：
  - https://github.com/TheDeathDragon/LiveTranslate
  - https://github.com/lovemefan/SenseVoice-python
  - https://github.com/snakers4/silero-vad
  - https://github.com/FunAudioLLM/SenseVoice
  - https://github.com/ggml-org/llama.cpp
  - https://github.com/HaujetZhao/Qwen3-ASR-GGUF

## 许可证

本项目的代码，除下述例外以外，遵循 MIT 许可证，详见 [LICENSE.md](LICENSE.md) 文件。

- `docs/` 文件夹下的内容来自各 API 提供商的文档，授权情况请以原始来源为准。
