实时语音识别-千问服务通过 WebSocket 协议，接收实时音频流并实时转写。支持[VAD 模式](#9b49887720jcw)和[Manual 模式](#ee09a3493fsuc)交互流程。

**用户指南：**模型介绍、功能特性和完整示例代码请参见[实时语音识别-千问](https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition)

## **URL**

编码时，将`<model_name>`替换为实际的[模型](https://help.aliyun.com/zh/model-studio/qwen-real-time-speech-recognition#ff8c59ef0busr)。

```
wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=<model_name>
```

## **Headers**

```
"Authorization": "bearer <your_dashscope_api_key>"
```

## VAD 模式（默认）

服务端自动检测语音的起点和终点（断句）。开发者只需持续发送音频流，服务端会在检测到一句话结束时自动返回最终识别结果。此模式适用于实时对话、会议记录等场景。

**启用方式：**配置客户端`[session.update](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#af43722339yva)`事件的`session.turn_detection`参数。

![image](https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/1613880771/CAEQaxiBgICW9b6H3RkiIGM5MDgwMTNkMjBjMDRlNTNiOGZlODNjZGJhNDQ3NGJm5812623_20251022102739.334.svg)

-   客户端通过发送`[input_audio_buffer.append](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#a42f8e9111n72)`事件将音频追加到缓冲区。
    
-   服务端在检测到语音时返回`[input_audio_buffer.speech_started](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#038d846199nbm)`事件。
    
    **注意**：如果客户端尚未收到该事件，就直接发送`[session.finish](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#147ce70052d4z)`结束会话，服务端会直接返回`[session.finished](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#6eaa77339djdv)`事件，随后客户端需主动断开连接。
    
-   客户端继续发送`[input_audio_buffer.append](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#a42f8e9111n72)`事件提交音频。
    
-   客户端在音频提交完后，发送`[session.finish](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#147ce70052d4z)`事件通知服务端结束当前会话。
    
-   服务端在检测到语音结束时返回`[input_audio_buffer.speech_stopped](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#3d73b074cak7k)`事件。
    
-   服务端返回`[input_audio_buffer.committed](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#1108a3764an0e)`事件。
    
-   服务端返回`[conversation.item.created](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#04dabbb9b6eto)`事件。
    
-   服务端返回`[conversation.item.input_audio_transcription.text](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#ba1b5cdd79fxu)`事件，其中包含语音识别实时结果。
    
-   服务端返回`[conversation.item.input_audio_transcription.completed](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#403ecacd74qqg)`事件，其中包含语音识别最终结果。
    
-   服务端返回`[session.finished](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#6eaa77339djdv)`事件，通知客户端识别结束，此时客户端需要主动断开连接。
    

## **Manual 模式**

由客户端控制断句。客户端需要发送完一整句话的音频后，再发送一个`[input_audio_buffer.commit](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#d6d5cd90f3q4c)`事件来通知服务端。此模式适用于客户端能明确判断语句边界的场景，如聊天软件中的发送语音。

**启用方式：**将客户端`[session.update](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#af43722339yva)`事件的`session.turn_detection`设为null。

![image](https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/1613880771/CAEQaxiBgMDUp8qH3RkiIGEyYTc0NTI1ZmQ1OTQ5NjliNWE0OTYwYTAwMDBlMjBm5812623_20251022102739.334.svg)

-   客户端通过发送`[input_audio_buffer.append](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#a42f8e9111n72)`事件将音频追加到缓冲区。
    
-   客户端通过发送`[input_audio_buffer.commit](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#d6d5cd90f3q4c)`事件来提交输入音频缓冲区。 该提交会在对话中创建一个新的用户消息项。
    
-   客户端发送`[session.finish](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-client-events#147ce70052d4z)`事件通知服务端结束当前会话。
    
-   服务端返回`[input_audio_buffer.committed](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#1108a3764an0e)`事件进行响应。
    
-   服务端返回`[conversation.item.input_audio_transcription.text](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#ba1b5cdd79fxu)`事件，其中包含语音识别实时结果。
    
-   服务端返回`[conversation.item.input_audio_transcription.completed](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#403ecacd74qqg)`事件，其中包含语音识别最终结果。
    
-   服务端返回`[session.finished](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-server-events#6eaa77339djdv)`事件，通知客户端识别结束，此时客户端需要主动断开连接。