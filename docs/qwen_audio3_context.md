# Qwen-Audio 3.1 ASR 上下文

官方资料（核对日期：2026-10-03）：

- [提升识别准确率](https://help.aliyun.com/zh/model-studio/improve-asr-accuracy)
- [客户端事件](https://help.aliyun.com/zh/model-studio/qwen-audio-asr-streaming-client-events)
- [Python SDK](https://help.aliyun.com/zh/model-studio/qwen-audio-asr-streaming-python-sdk)

`qwen-audio-3.1-asr-flash-streaming` 支持领域词表和对话历史。`user` +
`input_text` 表示已识别的语音原文或领域语料；`assistant` + `text` 专指大模型的
回复，可省略。其他人说的话也是人类语音，不能伪装成 assistant；译文也不是回复。

服务端最多保留最近 5 轮，每轮所有 user/assistant 文本合计最多 400 字符。
上下文主要通过原词匹配起效，泛泛的语义描述帮助有限。

当前实现：

- 公开和私人热词走 `parameters.vocabulary`，最多 2000 个，普通权重 1–5，
  最多 50 个权重为 50 的超级热词。不再为即时热词清理服务端预编译词表。
- 领域语料与 VRCX 世界/玩家名称合并成一条最多 400 字符的 user 消息，避免
  把完整元数据切成五条挤掉发言。两种来源并存时分别预留约 200 字符。
- 自己的最终识别原文在回调入口记录；对方原文来自实时字幕 IPC 的
  `FOREIGN_SPEECH.source_text`，即使翻译关闭也会记录。不记录 partial、译文或
  被撤回流程丢弃的结果。
- 最多保留最近 5 条发言（双方合计），按接收时间排列；有领域信息时发送最近
  4 条。每条含“自己/对方”标记，文本总长不超过 400 字符。120 秒后过期，
  IPC 断开清除对方发言，新的服务实例使用新的缓存。
- start/resume 使用 `raw_input={"context": [...]}`；长连接期间内容变化时，
  在下一帧音频前用 `update_context(payload_input={"context": [...]})` 更新。
  内容不变不重复发送；过期后发送空 context 清除服务端旧历史。
- DashScope SDK 至少 1.27.5，当前依赖范围为 `>=1.27.5,<1.28.0`。
  更新与音频发送共用生命周期锁和 SDK FIFO；回调只写缓存，避免 stop 等待
  回调时发生锁互等。

本地验证覆盖历史边界、IPC 无翻译器接收/断开、最终结果筛选、真实 SDK 队列
顺序及 start/resume 更新。协议联调使用合成上下文与静音，不代表真人识别质量。
