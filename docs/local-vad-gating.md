# 统一 VAD：在线 API 门控与本地 ASR 分段

## 概述

Yakutan 的 VAD 设置现在统一由 `VAD_ENABLED` 和一组 `LOCAL_VAD_*` 参数控制：

- 在线 API 后端（Qwen/DashScope/Soniox/Doubao）：客户端使用 Silero VAD 做发送门控，静音时不向 ASR 发送音频帧以减少无效计费。
- 本地 ASR 后端：采集侧不做发送门控，继续把连续音频交给本地识别器，由本地识别器内部 VAD 做自动分段。

Web UI 中的入口为「高级设置 -> VAD」。本地音频识别卡片只保留引擎、增量识别和中间结果间隔；VAD 参数统一移到高级设置。

## 设计目标

1. **统一控制**：一个总开关同时影响在线 API 门控和本地 ASR 分段。
2. **省流**：在线 API 后端静音时停止发送音频帧，但保持识别会话在线。
3. **本地连续音频**：本地 ASR 仍接收包含静音的连续音频，避免采集侧门控破坏本地分段逻辑。
4. **状态清理**：闭麦/停止识别时重置 VAD 状态和缓存，下一次开麦从干净状态开始。

## 架构

```text
Online API:
Mic -> read_audio_data() -> VADProcessor.process_chunk() [side channel]
                         -> is_speaking?
                            -> speech: send_queue -> ASR backend
                            -> silence: drop frame

Local ASR:
Mic -> read_audio_data() -> LocalSpeechRecognizer -> internal VAD segmentation
```

VAD 侧路分析不阻塞主通道。在线门控初始化失败时会自动降级，不影响正常音频采集。

## 配置项

```python
# config.py
VAD_ENABLED = True
LOCAL_VAD_MODE = 'silero'          # 本地识别使用：'silero' 或 'energy'
LOCAL_VAD_THRESHOLD = 0.50
LOCAL_VAD_MIN_SPEECH_DURATION = 1.0
LOCAL_VAD_MAX_SPEECH_DURATION = 30.0
LOCAL_VAD_SILENCE_DURATION = 0.8
LOCAL_VAD_PRE_SPEECH_DURATION = 0.2
```

环境变量：

- `VAD_ENABLED=0`：关闭统一 VAD。在线 API 不做发送门控，本地 ASR 内部 VAD 退化为 disabled。
- `ENABLE_VAD_GATING_VERBOSE=1`：启用在线门控详细诊断日志。
- `ENABLE_LOCAL_VAD_GATING_VERBOSE=1`：旧调试变量名，仍兼容。

`ENABLE_VAD`、`VAD_THRESHOLD`、`VAD_SILENCE_DURATION_MS` 仍保留给 Qwen 服务端 VAD 使用，但不再作为 Web UI 的主要 VAD 设置面。

## 相关文件

| 文件 | 作用 |
|------|------|
| `config.py` | 统一 VAD 默认配置 |
| `main.py` | 在线 API VAD 门控初始化、停止识别时重置 VAD |
| `audio_capture.py` | 采集侧 VAD 侧路分析与在线门控发送 |
| `speech_recognizers/local_speech_recognizer.py` | 本地 ASR 内部 VAD 分段 |
| `local_asr/vad_processor.py` | VAD 状态机、`is_speaking` 与 `reset()` |
| `ui/app.py` | `/api/config` 顶层 `vad` 对象 |
| `ui/` | 高级设置中的统一 VAD 表单 |
