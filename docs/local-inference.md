# 本地与远程推理

Qwen3-ASR 使用 llama.cpp 原生 mtmd 音频编码器和 GGUF 解码器。Python ctypes 绑定固定匹配 **b11985**；旧版 b8391 和旧 CUDA ABI 配置不能与新版绑定混用。

模型文件默认位于源码目录（打包后为 exe 同级）的 `local_asr_models`：

```text
local_asr_models/
  qwen3-asr-gguf/
    Qwen3-ASR-1.7B.Q6_K.gguf
    Qwen3-ASR-1.7B.mmproj-Q8_0.gguf
  llama_vulkan_b11985/
    llama.dll
    mtmd.dll
    ggml*.dll
    libomp*.dll
  hymt2/                 # 复用现有 Hy-MT2 GGUF
  sensevoice-onnx/       # 复用现有 SenseVoice
```

Qwen 权重来自 `mradermacher/Qwen3-ASR-1.7B-GGUF`，runtime 与 Silero VAD 来自 `febilly/realtime-subtitle-local-runtime`。下载固定 revision，并校验 SHA256；HF 不通时尝试镜像，runtime 与 Silero 最后可尝试上游 GitHub。Silero 的原有打包路径保持兼容。

安装 `requirements.txt` 与 `requirements-local-inference.txt`。已有环境升级时，先卸载 `onnxruntime-directml`，再安装 `onnxruntime`，避免两个发行包覆盖同一个 Python 模块。Qwen 编码器和解码器均使用 llama.cpp 后端；SenseVoice 与 Silero 使用 CPU ONNX Runtime。

ASR 与 Hy-MT2 的 context 设置 `n_outputs_max=64`；ASR 默认使用 Q8_0 KV，Hy-MT2 保持 f16 KV。共享 worker 的 KV/草稿/语言/上下文相互独立，模型权重与 mtmd 编码器共享。mtmd 编码调用串行，多会话解码仍可合批，每个解码子批最多请求 64 行 logits。

## 独立推理服务

Windows 可使用现有 `start_server.bat` 或 `remote_inference_service/launcher.py`。客户端在「模型运行方式」选择远程服务，填写同一个 `ws://服务器:18775` 地址供 ASR 和 Hy-MT2 使用；客户端保留本地 VAD。

Windows 服务直接加载同一目录下的模型。Linux 服务也使用 b11985 绑定，需自行构建该版本（启用 CUDA 等目标后端及 mtmd），将运行库放入模型根目录的 `llama_vulkan_b11985` 中；该目录名为跨平台的固定运行时路径。Linux 的外部 Hy-MT2 HTTP 后端仍使用 `--llama-url` 配置。旧版 CUDA runtime 必须先升级，不能沿用 `YAKUTAN_LLAMA_ABI_PROFILE=cuda-2026-08`。

## 清理旧文件

新版模型与 runtime 使用独立目录，下载或启动不会自动删除旧模型。确认新版识别、翻译及远程调用正常后，可在本地模型设置点击「清理旧模型/旧运行时」，确认目录及占用空间后执行。

清理仅匹配旧 `qwen3-asr` 中的三个旧权重、`qwen_llama_vulkan_bin` 的已知运行库和旧下载包。新模型、SenseVoice、Hy-MT2 与其他文件保留；正在占用的文件会保留并报告失败。未下载新版或下载进行中时，清理接口拒绝执行。清理不会访问其他项目的模型根目录。

## 来源与许可

原生 mtmd 适配移植自 realtime-subtitle 的 `codex/qwen-local-asr-translation` 分支、提交 `2ca5f7b`。llama.cpp 与 Silero VAD 为 MIT；Qwen3-ASR 与 Hy-MT2 模型为 Apache-2.0；LLVM OpenMP 为 Apache-2.0 with LLVM Exceptions；SenseVoice 以对应仓库许可为准。模型文件与原生运行库不纳入本项目 Git 提交。
