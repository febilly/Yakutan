# Qwen-MT 模型调用文档

本文介绍通过 OpenAI 兼容接口 或 DashScope API 调用 Qwen-MT 模型的输入与输出参数。

相关文档：翻译能力（Qwen-MT）

-----

## OpenAI 兼容

**北京地域新加坡地域**

  * **SDK 调用配置的base\_url为：** `https://dashscope.aliyuncs.com/compatible-mode/v1`
  * **HTTP 调用配置的endpoint：** `POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`

您需要已获取API Key并配置API Key到环境变量。若通过OpenAI SDK进行调用，需要安装SDK。

### 请求体

基础使用术语干预翻译记忆领域提示
PythonNode.jscurl

```python
import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
messages = [
    {
        "role": "user",
        "content": "我看到这个视频后没有笑"
    }
]
translation_options = {
    "source_lang": "Chinese",
    "target_lang": "English"
}

completion = client.chat.completions.create(
    model="qwen-mt-plus",
    messages=messages,
    extra_body={
        "translation_options": translation_options
    }
)
print(completion.choices[0].message.content)
```

#### 参数说明

**model string （必选）**
模型名称。支持的模型：qwen-mt-plus、qwen-mt-flash、qwen-mt-lite、qwen-mt-turbo。

**messages array （必选）**
消息数组，用于向大模型传递上下文。仅支持传入 User Message。

  * **消息类型**
      * **User Message object （必选）**
        用户消息，用于传递待翻译的句子。
      * **属性**
          * **content string（必选）**
            待翻译的句子。
          * **role string （必选）**
            用户消息的角色，必须设为user。

**stream boolean （可选） 默认值为 false**
是否以流式方式输出回复。

  * **可选值：**
      * `false`：等待模型生成完整回复后一次性返回。
      * `true`：模型边生成边返回数据块。客户端需逐块读取，以还原完整回复。
  * **说明**
    当前仅qwen-mt-flash、qwen-mt-lite模型支持以增量形式返回数据，每次返回仅包含新生成的内容。qwen-mt-plus和qwen-mt-turbo模型以非增量形式返回数据，每次返回当前已经生成的整个序列，暂时无法修改。如：
    ```text
    I
    I didn
    I didn't
    I didn't laugh
    I didn't laugh after
    ...
    ```

**stream\_options object （可选）**
流式输出的配置项，仅在 stream 为 true 时生效。

  * **属性**
      * **include\_usage boolean （可选）默认值为 false**
        是否在最后一个数据块包含Token消耗信息。
          * **可选值：**
              * `true`：包含；
              * `false`：不包含。

**max\_tokens integer （可选）**
用于限制模型输出的最大 Token 数。若生成内容超过此值，响应将被截断。
默认值与最大值均为模型的最大输出长度，请参见模型选型。

**seed integer （可选）**
随机数种子。用于确保在相同输入和参数下生成结果可复现。若调用时传入相同的 seed 且其他参数不变，模型将尽可能返回相同结果。
取值范围：[0,2^31−1]。

**temperature float （可选） 默认值为0.65**
采样温度，控制模型生成文本的多样性。
temperature越高，生成的文本更多样，反之，生成的文本更确定。
取值范围： [0, 2)
temperature与top\_p均可以控制生成文本的多样性，建议只设置其中一个值。

**top\_p float （可选）默认值为0.8**
核采样的概率阈值，控制模型生成文本的多样性。
top\_p越高，生成的文本更多样。反之，生成的文本更确定。
取值范围：（0,1.0]
temperature与top\_p均可以控制生成文本的多样性，建议只设置其中一个值。

**top\_k integer （可选）默认值为1**
生成过程中采样候选集的大小。例如，取值为50时，仅将单次生成中得分最高的50个Token组成随机采样的候选集。取值越大，生成的随机性越高；取值越小，生成的确定性越高。取值为None或当top\_k大于100时，表示不启用top\_k策略，此时仅有top\_p策略生效。
取值需要大于或等于0。
该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra\_body 对象中，配置方式为：`extra_body={"top_k": xxx}`；通过 Node.js SDK 或 HTTP 方式调用时，请作为顶层参数传递。

**repetition\_penalty float （可选）默认值为1.0**
模型生成时连续序列中的重复度。提高repetition\_penalty时可以降低模型生成的重复度，1.0表示不做惩罚。没有严格的取值范围，只要大于0即可。

**translation\_options object （必选）**
需配置的翻译参数。

  * **属性**
      * **source\_lang string （必选）**
        源语言的英文全称，详情请参见支持的语言。若设为auto，模型会自动识别输入的语种。
      * **target\_lang string （必选）**
        目标语言的英文全称，详情请参见支持的语言。
      * **terms arrays （可选）**
        使用术语干预功能时需设置的术语数组。
      * **tm\_list arrays （可选）**
        使用翻译记忆功能时需设置的翻译记忆数组。
      * **domains string （可选）**
        使用领域提示功能时需设置的领域提示语句。
        领域提示语句暂时只支持英文。
      * 该参数非OpenAI标准参数。通过 Python SDK调用时，请放入 extra\_body 对象中，配置方式为：`extra_body={"translation_options": xxx}`；通过 Node.js SDK 或 HTTP 方式调用时，请作为顶层参数传递。

### chat响应对象（非流式输出）

```json
{
  "id": "chatcmpl-999a5d8a-f646-4039-968a-167743ae0f22",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "I didn't laugh after watching this video.",
        "refusal": null,
        "role": "assistant",
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": null
      }
    }
  ],
  "created": 1762346157,
  "model": "qwen-mt-plus",
  "object": "chat.completion",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 9,
    "prompt_tokens": 53,
    "total_tokens": 62,
    "completion_tokens_details": null,
    "prompt_tokens_details": null
  }
}
```

  * **id string**
    本次请求的唯一标识符。
  * **choices array**
    模型生成内容的数组。
      * **属性**
          * **finish\_reason string**
            模型停止生成的原因。
            有两种情况：
            自然停止输出时为stop；
            生成长度过长而结束为length。
          * **index integer**
            当前对象在choices数组中的索引。
          * **message object**
            模型输出的消息。
              * **属性**
                  * **content string**
                    模型翻译结果。
                  * **refusal string**
                    该参数当前固定为null。
                  * **role string**
                    消息的角色，固定为assistant。
                  * **audio object**
                    该参数当前固定为null。
                  * **function\_call object**
                    该参数当前固定为null。
                  * **tool\_calls array**
                    该参数当前固定为null。
  * **created integer**
    本次请求被创建时的时间戳。
  * **model string**
    本次请求使用的模型。
  * **object string**
    始终为chat.completion。
  * **service\_tier string**
    该参数当前固定为null。
  * **system\_fingerprint string**
    该参数当前固定为null。
  * **usage object**
    本次请求的 Token 消耗信息。
      * **属性**

### chat响应chunk对象（流式输出）

增量输出非增量输出

```json
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": "", "function_call": null, "refusal": null, "role": "assistant", "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": "I", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": " didn", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": "'t", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": " laugh", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": " after", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": " watching", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": " this", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": " video", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": ".", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": "", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": "stop", "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [{"delta": {"content": "", "function_call": null, "refusal": null, "role": null, "tool_calls": null}, "finish_reason": "stop", "index": 0, "logprobs": null}], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": null}
{"id": "chatcmpl-d8aa6596-b366-4ed0-9f6d-2e89247f554e", "choices": [], "created": 1762504029, "model": "qwen-mt-flash", "object": "chat.completion.chunk", "service_tier": null, "system_fingerprint": null, "usage": {"completion_tokens": 9, "prompt_tokens": 56, "total_tokens": 65, "completion_tokens_details": null, "prompt_tokens_details": null}}
```

  * **id string**
    本次调用的唯一标识符。每个chunk对象有相同的 id。
  * **choices array**
    模型生成内容的数组。若设置include\_usage参数为true，则在最后一个chunk中为空。
      * **属性**
          * **delta object**
            流式返回的输出内容。
              * **属性**
                  * **content string**
                    翻译结果，qwen-mt-flash和qwen-mt-lite为增量式更新，qwen-mt-plus和qwen-mt-turbo为非增量式更新。
                  * **function\_call object**
                    该参数当前固定为null。
                  * **refusal object**
                    该参数当前固定为null。
                  * **role string**
                    消息对象的角色，只在第一个chunk中有值。
          * **finish\_reason string**
            模型停止生成的原因。有三种情况：
            自然停止输出时为stop；
            生成未结束时为null；
            生成长度过长而结束为length。
          * **index integer**
            当前响应在choices数组中的索引。
  * **created integer**
    本次请求被创建时的时间戳。每个chunk有相同的时间戳。
  * **model string**
    本次请求使用的模型。
  * **object string**
    始终为chat.completion.chunk。
  * **service\_tier string**
    该参数当前固定为null。
  * **system\_fingerprintstring**
    该参数当前固定为null。
  * **usage object**
    本次请求消耗的Token。只在include\_usage为true时，在最后一个chunk返回。
      * **属性**

-----

## DashScope

**北京地域新加坡地域**

  * **HTTP 调用配置的endpoint：** `POST https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`
  * SDK 调用无需配置 base\_url。

您需要已获取API Key并配置API Key到环境变量。若通过DashScope SDK进行调用，需要安装DashScope SDK。

### 请求体

基础使用术语干预翻译记忆领域提示
PythonJavacurl

```python
import os
import dashscope

messages = [
    {
        "role": "user",
        "content": "我看到这个视频后没有笑"
    }
]
translation_options = {
    "source_lang": "auto",
    "target_lang": "English",
}
response = dashscope.Generation.call(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model="qwen-mt-plus",
    messages=messages,
    result_format='message',
    translation_options=translation_options
)
print(response.output.choices[0].message.content)
```

#### 参数说明

**model string （必选）**
模型名称。支持的模型：qwen-mt-plus、qwen-mt-flash、qwen-mt-lite、qwen-mt-turbo。

**messages array （必选）**
消息数组，用于向大模型传递上下文。仅支持传入 User Message。

  * **消息类型**
      * **User Message object （必选）**
        用户消息，用于传递待翻译的句子。
      * **属性**
          * **content string（必选）**
            待翻译的句子。
          * **role string （必选）**
            用户消息的角色，必须设为user。

**max\_tokens integer （可选）**
用于限制模型输出的最大 Token 数。若生成内容超过此值，响应将被截断。
默认值与最大值均为模型的最大输出长度，请参见模型选型。
Java SDK中为maxTokens。通过HTTP调用时，请将 max\_tokens 放入 parameters 对象中。

**seed integer （可选）**
随机数种子。用于确保在相同输入和参数下生成结果可复现。若调用时传入相同的 seed 且其他参数不变，模型将尽可能返回相同结果。
取值范围：[0,2^31−1]。
通过HTTP调用时，请将 seed 放入 parameters 对象中。

**temperature float （可选） 默认值为0.65**
采样温度，控制模型生成文本的多样性。
temperature越高，生成的文本更多样，反之，生成的文本更确定。
取值范围： [0, 2)
temperature与top\_p均可以控制生成文本的多样性，建议只设置其中一个值。
通过HTTP调用时，请将 temperature 放入 parameters 对象中。

**top\_p float （可选）默认值为0.8**
核采样的概率阈值，控制模型生成文本的多样性。
top\_p越高，生成的文本更多样。反之，生成的文本更确定。
取值范围：（0,1.0]
temperature与top\_p均可以控制生成文本的多样性，建议只设置其中一个值。
Java SDK中为topP。通过HTTP调用时，请将 top\_p 放入 parameters 对象中。

**repetition\_penalty float （可选）默认值为1.0**
模型生成时连续序列中的重复度。提高repetition\_penalty时可以降低模型生成的重复度，1.0表示不做惩罚。没有严格的取值范围，只要大于0即可。
Java SDK中为repetitionPenalty。通过HTTP调用时，请将 repetition\_penalty 放入 parameters 对象中。

**top\_k integer （可选）默认值为1**
生成过程中采样候选集的大小。例如，取值为50时，仅将单次生成中得分最高的50个Token组成随机采样的候选集。取值越大，生成的随机性越高；取值越小，生成的确定性越高。取值为None或当top\_k大于100时，表示不启用top\_k策略，此时仅有top\_p策略生效。
取值需要大于或等于0。
Java SDK中为topK。通过HTTP调用时，请将 top\_k 放入 parameters 对象中。

**stream boolean （可选）**
是否以流式方式输出回复。

  * **可选值：**
      * `false`：等待模型生成完整回复后一次性返回。
      * `true`：模型边生成边返回数据块。客户端需逐块读取，以还原完整回复。
  * **说明**
    当前仅qwen-mt-flash、qwen-mt-lite模型支持以增量形式返回数据，每次返回仅包含新生成的内容。qwen-mt-plus和qwen-mt-turbo模型以非增量形式返回数据，每次返回当前已经生成的整个序列，暂时无法修改。如：
    ```text
    I
    I didn
    I didn't
    I didn't laugh
    I didn't laugh after
    ...
    ```
    该参数仅支持Python SDK。通过Java SDK实现流式输出请通过streamCall接口调用；通过HTTP实现流式输出请在Header中指定X-DashScope-SSE为enable。

**translation\_options object （必选）**
需配置的翻译参数。

  * **属性**
      * **source\_lang string （必选）**
        源语言的英文全称，详情请参见支持的语言。若设为auto，模型会自动识别输入的语种。
      * **target\_lang string （必选）**
        目标语言的英文全称，详情请参见支持的语言。
      * **terms arrays （可选）**
        使用术语干预功能时需设置的术语数组。
      * **tm\_list arrays （可选）**
        使用翻译记忆功能时需设置的翻译记忆数组。
      * **domains string （可选）**
        使用领域提示功能时需设置的领域提示语句。
        领域提示语句暂时只支持英文。
      * Java SDK中为translationOptions。通过HTTP调用时，请将 translation\_options 放入 parameters 对象中。

### chat响应对象（流式与非流式输出格式一致）

```json
{
  "status_code": 200,
  "request_id": "9b4ec3b2-6d29-40a6-a08b-7e3c9a51c289",
  "code": "",
  "message": "",
  "output": {
    "text": null,
    "finish_reason": "stop",
    "choices": [
      {
        "finish_reason": "stop",
        "message": {
          "role": "assistant",
          "content": "I didn't laugh after watching this video."
        }
      }
    ],
    "model_name": "qwen-mt-plus"
  },
  "usage": {
    "input_tokens": 53,
    "output_tokens": 9,
    "total_tokens": 62
  }
}
```

  * **status\_code string**
    本次请求的状态码。200 表示请求成功，否则表示请求失败。
    Java SDK不会返回该参数。调用失败会抛出异常，异常信息为status\_code和message的内容。
  * **request\_id string**
    本次调用的唯一标识符。
    Java SDK返回参数为requestId。
  * **code string**
    错误码，调用成功时为空值。
    只有Python SDK返回该参数。
  * **output object**
    调用结果信息。
      * **属性**
  * **usage object**
    本次请求使用的Token信息。
      * **属性**

## 错误码

如果模型调用失败并返回报错信息，请参见错误信息进行解决。