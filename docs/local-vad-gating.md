# 本地 VAD 发送门控（实验性）

## 概述

Yakutan 默认将所有麦克风采集到的音频（含静音）持续发送到云端 ASR 后端（Qwen/DashScope/Soniox），按音频时长计费。当用户忘记手动 mute 时，静音帧也会消耗 API 费用。

**本地 VAD 发送门控**：在客户端侧用一个轻量 VAD（Silero ONNX，无 PyTorch，CPU 推理 ~1ms/帧）旁路分析采集的音频，仅在检测到有效语音时才将音频发送到 ASR 后端。当 VAD 判定用户静音时，停止发送音频帧，但仍保持 WebSocket 连接始终在线（不执行 pause/resume）。

## 设计目标

1. **省流**：静音时不向 ASR 发送音频，减少无效 API 计费。
2. **不重连**：不调用 `recognizer.pause()` / `resume()`，避免频繁关闭/重建 WebSocket 会话。
3. **信任服务端 VAD**：Qwen 服务端自带 800ms VAD。当客户端停止发送音频后，服务端自然检测到静音→触发终句。不需要客户端额外发送 trailing silence。
4. **首字不丢**：利用 Silero VAD 内置的 pre-buffer 机制（`pre_speech_duration=0.2s`），检测到语音时回溯补齐前 200ms 音频。
5. **渐进验证**：第一版仅做日志输出，不改变音频发送行为。待确认 VAD 状态切换稳定后再接入门控。

## 架构

```
Mic → read_audio_data() → VADProcessor.process_chunk()  [侧路分析]
    │                             │
    │                       is_speaking?
    │                       ├─ SPEECH → 日志 "[VAD] ▶ SPEECH 开始"
    │                       └─ SILENCE → 日志 "[VAD] ■ SILENCE"
    │
    └─ send_queue → ASR 后端 [主通道，不改变行为]
```

VAD 侧路分析不阻塞主通道。VAD 失败（模型未下载、onnxruntime 未安装）时自动降级，不影响正常音频流。

## 关键组件

### VADProcessor（复用现有）

来自 `local_asr/vad_processor.py`，已用于 `LocalSpeechRecognizer`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `threshold` | 0.50 | Silero 语音概率阈值 |
| `min_speech_duration` | 1.0s | 最短信语句持续时间（过滤短噪声） |
| `silence_duration` | 0.8s | 触发"说话结束"的静音时长 |
| `pre_speech_duration` | 0.2s | 回溯缓冲长度，补发句首丢失的音频 |
| `chunk_duration` | 32ms (512/16000) | 每个分析帧的时长 |

### 配置项

```python
# config.py
ENABLE_LOCAL_VAD_GATING = False  # 默认关闭

# 复用已有参数（不改动）
LOCAL_VAD_MODE = 'silero'
LOCAL_VAD_THRESHOLD = 0.50
LOCAL_VAD_MIN_SPEECH_DURATION = 1.0
LOCAL_VAD_SILENCE_DURATION = 0.8
LOCAL_VAD_PRE_SPEECH_DURATION = 0.2
```

## 验证计划

### Phase 1：日志验证（当前）

`ENABLE_LOCAL_VAD_GATING = True`，但音频流不变。观察日志确认：

- [ ] VAD 状态切换稳定（不会在 speech/silence 间快速抖动）
- [ ] 首字不会被误判为静音（pre-buffer 生效）
- [ ] 背景噪声不会频繁触发 speech 事件
- [ ] 说话结束后的 silence 判定延迟可接受

### Phase 2：门控接入

确认 Phase 1 稳定后，接入发送门控：
```
if vad.is_speaking:
    send_queue.put(data)
else:
    drop  # 不发送静音帧
```

### Phase 3：断句优化（可选）

如果门控模式下降级体验有问题（如终句延迟过大），研究是否需要发 trailing silence 辅助服务端 VAD。

## 依赖

| 依赖 | 大小 | 说明 |
|------|------|------|
| `onnxruntime` (CPU) | ~5 MB | ONNX 推理运行时 |
| `silero_vad_16k_op15.onnx` | ~1.3 MB | Silero VAD 模型，首次运行时自动下载 |

模型下载逻辑：`local_asr.model_manager.download_silero()` → 从 GitHub Release 下载到用户缓存目录。

## 相关文件

| 文件 | 改动 |
|------|------|
| `config.py` | 新增 `ENABLE_LOCAL_VAD_GATING` 开关 |
| `app_state.py` | 新增 `vad_processor`、`_vad_pending_samples`、`_vad_was_speaking` |
| `main.py` | 初始化 VADProcessor，自动下载模型 |
| `audio_capture.py` | capture loop 中新增 VAD 侧路分析 + 状态日志 |
| `requirements.txt` | 新增 `onnxruntime>=1.19.0` |
| `run_ui.spec` | 新增 `local_asr`、`onnxruntime` 隐藏导入 |
