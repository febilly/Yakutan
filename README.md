# Yakutan

更适合中国宝宝体质的 VRChat 语音翻译器（翻译你自己的声音）

<div align="center">
    <img src="images/screenshot.png" alt="A Screenshot of the WebUI of Yakutan" style="max-width: 100%; width: 512px; height: auto;">
</div>


- 可用大模型进行**流式翻译**，实现**边说边翻译**的效果，减少对方等待时间。
    - 也有 DeepL、qwen-mt、谷歌翻译等选项
- 使用阿里的 [Qwen3实时语音识别（默认）](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash) 或 [Fun-ASR](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/fun-asr-realtime) 进行语音转文本，也有部分其他ASR可选
    - 本地语音识别正在开发中，可下载最新Pre-release版本尝试，但可能有bug
- 将结果通过 OSC 发送至游戏
- 有一些 [独特的功能](#特点)
- ~~其实就是把一堆 API 粘在了一起~~

## 快速开始

1. 在 [https://github.com/febilly/Yakutan/releases/latest](https://github.com/febilly/Yakutan/releases/latest) 中下载最新的稳定版 exe
2. 将 exe 放在一个空文件夹中并运行，WebUI 应该会自动打开
3. 根据 WebUI 中 `API Keys 配置` 面板的说明获取并填入 API Keys
4. 点击 `启动服务` 按钮
5. 有问题的话请参见 [常见问题](#常见问题) 部分

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
- 现有的翻译一般都需要等到一整句话说完后才进行翻译，别人得等半天，等你说完，再等它翻译完，才能看到你到底说了个啥

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
        - 停止录制时会额外继续录制一小段音频（默认0.2s），防止漏掉最后一个字
    - VAD：仍然有 VAD 作为补充断句方法

### 翻译方面

- 实现了翻译的上下文，默认附带一条简短的场景说明，和最近的6条消息
- 附带备用语言选项，如果识别到的源语言和主目标语言相同，则翻译至备用语言
    - 可以实现两种语言之间的互译
- 默认使用Deepl翻译
    - 可以指定翻译的正式程度（比如对于日语来说）
    - 原生支持上下文
    - 可以自定义词库（本项目还没实现）
- 可以切换为使用大模型进行流式翻译
    - 内置若干 LLM 预设，也支持自定义 LLM 配置
- 支持第二输出语言，可同时输出两种译文
- 可以切换为开箱即用的谷歌翻译（但有网络连通性问题，及速率限制）


## 局限性
- ~~你得会配Python环境~~ **已提供打包好的可执行文件**
- ~~目前没（懒得）写 GUI，所有配置需要在 `config.py` 文件中修改~~ **现在有Web UI了**
- ~~使用脚本启动时系统的默认麦克风~~ **现在可以在Web UI中选择麦克风了**
- ~~目前暂不支持和其他 OSC 程序同时运行~~ **已改用 OSCQuery，不再冲突**
- 使用完毕后请**一定记得停止服务**，否则可能会持续使用转录 API，产生额外费用
- 需要用商业服务的API Key，有一定免费额度，但免费额度用完后需要付钱
    - 阿里云的免费额度是一次性的，但是 [大学生可以拿到每年的免费额度](https://university.aliyun.com/buycenter/)
    - DeepL的免费额度每月重置，但是怎么拿到Key需要自己想办法
        - 实在懒得折腾可以把翻译器换成谷歌的
- 语言识别默认使用一个简单的中日韩英检测器
    - 已添加自动切换逻辑

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
python -m venv .  # 推荐使用uv，速度爆快
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 3. 安装依赖

```bash
pip install -r requirements.txt  # 推荐使用uv，速度爆快
```

### 4. 运行程序

```bash
python run_ui.py
```

</details>

## API Key 获取

- **阿里云百炼（必需）**：https://bailian.console.aliyun.com/?tab=model#/model-market/detail/fun-asr-realtime
  - 注册后可获得免费额度
  - 大学生可申请每年的免费额度
  
- **DeepL（可选）**：https://www.deepl.com/en/pro-api
  - 每月有 500,000 字符的免费额度
  - 需要使用特殊方式获取 API Key
  
- **OpenRouter（可选）**：https://openrouter.ai/
  - 提供多种 LLM 模型
  - 部分模型有免费额度（如 Gemini）

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
2. 打开快捷菜单（圆盘菜单）
3. 进入 Options → OSC
4. 点击 "Enable" 启用 OSC

## 常见问题

<details>

### 1. 没有任何转录

- 检查 WebUI 中选择的麦克风是否为你正在使用的麦克风
- 检查麦克风是否有声音输入
- 检查 `游戏静音时暂停转录` 配置：
  - 如果打开，需要在 VRChat 中**第一次由闭麦状态切换到开麦状态后**才能开始识别
  - 如果关闭，程序启动后会立即开始识别

### 2. VRChat 聊天框没有显示

- 确认 VRChat OSC 已启用
- 打开游戏小菜单看有没有因发送消息过快被禁言（等待30秒后自动解除）

### 3. 识别不到游戏内的麦克风的开启与关闭

- 确认 VRChat OSC 已启用
- 一般重启电脑可以解决，如果暂时懒得重启可以先暂时关闭 `游戏静音时暂停转录` 配置作为临时解决方案

### 4. 翻译延迟较高

- 如果使用 LLM 翻译，建议换用响应速度更快的模型，以及检查 LLM 是否使用了思考模式（通过自定义 extra_body 来关闭，具体方式请查阅 LLM 提供商文档）
- 可以使用 DeepL 或 Google API 以获得更快的响应速度

### 5. 语音识别报错

- 如你在使用阿里的语音识别服务：
    - （国内版）检查阿里账号是否已实名
    - （国际版）检查阿里账号是否已绑定手机号与信用卡

### 6. 重启识别时程序自动退出

- 目前是有这么个 bug 偶发出现，咋办？先凑合用着呗，我之后再去修

</details>

## 附录

- 要翻译别人的声音的话建议用 [soniox](https://github.com/febilly/realtime-subtitle/releases)，这个效果很好，不过 Soniox API 是付费的（我是一分钱没赚啊喂）
    - 也可以试试免费的 [LiveCaptions Translator](https://github.com/SakiRinn/LiveCaptions-Translator)
    - 以及被我狠狠地嫖了代码的 [LiveTranslate](https://github.com/TheDeathDragon/LiveTranslate)
- 我还没太试过国内其他家的识别服务效果怎样，如果有更好的（并且有不少免费额度的）请告诉我谢谢

## 致谢

- 本项目部分基于阿里给的 Fun-ASR 示例代码
- 快速的 Google Translate API 来自 https://github.com/SakiRinn/LiveCaptions-Translator
- 提示词少量参考了 https://github.com/kapitalismho/PuriPuly-heart
- 还未进入稳定版的本地语音识别：
    - 本地ASR代码来自 https://github.com/TheDeathDragon/LiveTranslate
    - ONNX版Sensevoice来自 https://github.com/lovemefan/SenseVoice-python
    - 用到的其他东西：
        - https://github.com/snakers4/silero-vad
        - https://github.com/FunAudioLLM/SenseVoice
        - https://github.com/ggml-org/llama.cpp
        - https://github.com/microsoft/onnxruntime
        - https://github.com/HaujetZhao/Qwen3-ASR-GGUF

## 许可证

本项目的代码，除下述例外以外，遵循 MIT 许可证，详见 [LICENSE.md](LICENSE.md) 文件
- /docs 文件夹下的内容来自各 API 提供商的文档，是啥授权我也不知道