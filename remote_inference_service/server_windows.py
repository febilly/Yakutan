#!/usr/bin/env python3
"""Unified WebSocket inference service for Yakutan on Windows hosts (Foreground Console).

与 Linux 版 server.py 的有意差异（WebSocket 协议一致，便于后续抽公共协议模块）：
- ONNX 默认 DirectML（Linux 为 CUDA）；Qwen 微批池默认关闭以省显存；
- Hy-MT2 优先使用进程内引擎（本地 GGUF 存在时），外部 llama-server 仅作回退；
- health.ready = 任一 ASR 引擎就绪（Linux 版为全部模型就绪）；
- 默认仅监听 127.0.0.1；供局域网连接请显式传 --host 0.0.0.0（服务无鉴权）；
- 支持启动预热 (--warmup auto|all|none)。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保 Windows 命令行输出 UTF-8 编码，防止中文乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from websockets.asyncio.server import serve

from winutils import check_dependencies, get_local_ips

PROTOCOL_VERSION = 1
ASR_ENGINES = ("sensevoice", "qwen3-asr")
CHAT_PREFIX = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>"
CHAT_SUFFIX = "<｜hy_Assistant｜>"

LANGUAGE_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ru": "Russian", "vi": "Vietnamese",
    "yue": "Cantonese", "zh": "Chinese",
}


def log(tag: str, msg: str) -> None:
    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] [{tag}] {msg}", flush=True)


def log_err(tag: str, msg: str) -> None:
    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] [{tag}] [ERROR] {msg}", flush=True)


def get_client_str(websocket: Any) -> str:
    try:
        addr = getattr(websocket, "remote_address", None)
        if addr and isinstance(addr, tuple):
            return f"{addr[0]}:{addr[1]}"
    except Exception:
        pass
    return "client"


def check_runtime_dependencies() -> list[str]:
    _, missing = check_dependencies()
    return missing


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_runtime = os.getenv("YAKUTAN_RUNTIME_ROOT") or str(project_root)
    default_models = os.getenv("YAKUTAN_MODELS_ROOT") or str(project_root / "local_asr_models")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="监听地址 (默认 127.0.0.1 仅本机使用；供局域网连接请显式指定 0.0.0.0，注意本服务无鉴权)",
    )
    parser.add_argument("--port", type=int, default=18775, help="WebSocket 监听端口 (默认 18775)")
    parser.add_argument("--runtime-root", default=default_runtime)
    parser.add_argument("--models-root", default=default_models)
    parser.add_argument(
        "--llama-url",
        default=os.getenv("YAKUTAN_HYMT2_LLAMA_URL", "http://127.0.0.1:18776/completion"),
    )
    parser.add_argument("--max-message-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--hymt2-max-new-tokens", type=int, default=160)
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument(
        "--microbatch", action="store_true", default=False,
        help="启用 Qwen 动态微批并发池 (适用于大显存 GPU；默认关闭以节省 3GB+ 显存)",
    )
    parser.add_argument(
        "--qwen-workers", type=int,
        default=int(os.getenv("YAKUTAN_QWEN_WORKERS", "16")),
        help="maximum Qwen sequence IDs combined into one llama.cpp microbatch",
    )
    parser.add_argument(
        "--qwen-batch-wait-ms", type=float,
        default=float(os.getenv("YAKUTAN_QWEN_BATCH_WAIT_MS", "80")),
        help="maximum adaptive batching delay for the oldest Qwen request",
    )
    parser.add_argument(
        "--hymt2-device", default=os.getenv("YAKUTAN_HYMT2_DEVICE", "auto"),
        choices=["auto", "cpu", "gpu"],
        help="Hy-MT2 运行设备 (默认 auto: 优先GPU显存，显存不足自动无感降级到CPU)",
    )
    parser.add_argument(
        "--warmup", dest="warmup", choices=["auto", "all", "none"], default="auto",
        help="启动预热策略 (默认 auto：预热高频使用的 Qwen3-ASR 与 Hy-MT2；"
             "SenseVoice 按需懒加载，首次请求时才加载；all 全量预热；none 不预热)",
    )
    parser.add_argument(
        "--no-warmup", dest="warmup", action="store_const", const="none",
        help="禁用启动预热 (等价于 --warmup none；启动较快，但首句发话会有数秒冷启动)",
    )
    return parser.parse_args()


def _require_dir(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve() if value else None
    if path is None or not path.is_dir():
        raise SystemExit(f"{label} is missing or not a directory: {value!r}")
    return path


def configure_runtime(runtime_root: Path, models_root: Path) -> None:
    sys.path.insert(0, str(runtime_root))
    os.environ["YAKUTAN_QWEN_LLAMA_BIN"] = str(models_root / "qwen_llama_vulkan_bin")

    # Windows 桌面环境保留标准桌面 ABI：无条件清除可能残留的 Linux 特化配置 (cuda-2026-08)
    os.environ.pop("YAKUTAN_LLAMA_ABI_PROFILE", None)

    # 允许 Vulkan 在显存不足时借用系统共享内存，避免抛出 ErrorOutOfDeviceMemory
    os.environ["GGML_VK_ALLOW_SYSMEM_FALLBACK"] = "1"
    import config
    from local_inference import model_manager

    model_manager.MODELS_DIR = models_root
    config.LOCAL_INFERENCE_DEVICE = "auto"

    # Windows 下默认优先 DirectML，亦兼容 CUDA/CPU
    if "YAKUTAN_ONNX_PROVIDER" not in os.environ:
        os.environ["YAKUTAN_ONNX_PROVIDER"] = "directml"

    config.LOCAL_QWEN_ENCODER_DEVICE = "gpu"
    config.LOCAL_QWEN_LOG_PIPELINE_TIMING = True


def _is_llama_server_online(url: str) -> bool:
    try:
        import urllib.parse
        p = urllib.parse.urlparse(url)
        host = p.hostname or "127.0.0.1"
        port = p.port or 18776
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.15)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def _model_status(models_root: Path, llama_url: str = "") -> dict[str, dict[str, Any]]:
    primary_dll = "llama.dll"
    expected = {
        "sensevoice": [
            models_root / "sensevoice-onnx" / "am.mvn",
            models_root / "sensevoice-onnx" / "embedding.npy",
            models_root / "sensevoice-onnx" / "sense-voice-encoder-int8.onnx",
            models_root / "sensevoice-onnx" / "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ],
        "qwen3-asr": [
            models_root / "qwen3-asr" / "qwen3_asr_encoder_frontend.int4.onnx",
            models_root / "qwen3-asr" / "qwen3_asr_encoder_backend.int4.onnx",
            models_root / "qwen3-asr" / "qwen3_asr_llm.q4_k.gguf",
            models_root / "qwen_llama_vulkan_bin" / primary_dll,
        ],
    }
    result: dict[str, dict[str, Any]] = {}
    for name, paths in expected.items():
        missing = [str(path) for path in paths if not path.is_file()]
        result[name] = {"ready": not missing, "missing": missing}

    # 检查 Hy-MT2 (本地存在 .gguf 即可使用内置引擎；否则检查外部 llama-server)
    hymt2_files = list((models_root / "hymt2").glob("*.gguf"))
    if hymt2_files:
        result["hymt2"] = {"ready": True, "missing": []}
    elif llama_url and _is_llama_server_online(llama_url):
        result["hymt2"] = {"ready": True, "missing": []}
    else:
        result["hymt2"] = {"ready": False, "missing": ["缺失模型文件: *.gguf 且外部 llama-server 未运行"]}

    # 运行时依赖检查
    deps = {
        "sensevoice": ["onnxruntime", "sentencepiece", "kaldi_native_fbank"],
        "qwen3-asr": ["onnxruntime", "gguf", "kaldi_native_fbank", "soundfile"],
    }
    for engine, mod_names in deps.items():
        if engine in result and result[engine]["ready"]:
            missing_mods = []
            for mod in mod_names:
                try:
                    __import__(mod)
                except ImportError:
                    missing_mods.append(f"Python 依赖: {mod}")
            if missing_mods:
                result[engine]["ready"] = False
                result[engine]["missing"].extend(missing_mods)

    return result


@dataclass
class ASRSessionState:
    context: str = ""
    draft_tokens: list[int] = field(default_factory=list)
    # 单实例模式下该会话独占的 Qwen context worker（共享权重，独立 KV/草稿/上下文）
    qwen_worker: Any | None = None


class ASREngineManager:
    def __init__(
        self,
        models_root: Path,
        *,
        use_microbatch: bool = False,
        qwen_workers: int = 16,
        qwen_batch_wait_ms: float = 80.0,
    ) -> None:
        self.models_root = models_root
        self.use_microbatch = use_microbatch
        self._sensevoice: Any | None = None
        self._sensevoice_lock = asyncio.Lock()
        self._qwen_engine: Any | None = None
        self._qwen_lock = asyncio.Lock()
        self._qwen_workers = max(1, int(qwen_workers))
        self._qwen_batch_wait_ms = max(0.0, float(qwen_batch_wait_ms))
        self._qwen_scheduler: Any | None = None

    @staticmethod
    def _load_sensevoice():
        from local_inference.asr_sensevoice import SenseVoiceEngine
        return SenseVoiceEngine()

    @staticmethod
    def _load_qwen_base():
        from local_inference.asr_qwen3 import Qwen3ASREngine
        return Qwen3ASREngine(corpus_text=None)

    async def _ensure_qwen(self) -> None:
        if self._qwen_engine is not None:
            return
        async with self._qwen_lock:
            if self._qwen_engine is not None:
                return
            log("Qwen3-ASR", "正在初始化并加载 Qwen3-ASR 桌面轻量引擎 (仅需~1.2G显存)...")
            base = await asyncio.to_thread(self._load_qwen_base)
            if self.use_microbatch:
                from local_inference.qwen_microbatch import QwenDynamicBatchScheduler, QwenMicroBatchEngine
                n_ctx = int(getattr(__import__("config"), "LOCAL_QWEN_ASR_N_CTX", 2048))
                engine = await asyncio.to_thread(
                    QwenMicroBatchEngine,
                    base,
                    max_sequences=self._qwen_workers,
                    n_ctx_per_sequence=n_ctx,
                )
                self._qwen_scheduler = QwenDynamicBatchScheduler(
                    engine, max_wait_ms=self._qwen_batch_wait_ms,
                )
                self._qwen_engine = engine
            else:
                # 默认与客户端原生本地推理保持一致：单 Context 轻量模式，大幅节省显存给翻译和游戏
                self._qwen_engine = base
            log("Qwen3-ASR", "Qwen3-ASR 引擎就绪！")

    async def transcribe(
        self,
        name: str,
        state: ASRSessionState,
        audio: np.ndarray,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str, dict[str, float | int]]:
        queued_at = asyncio.get_running_loop().time()
        if name == "sensevoice":
            async with self._sensevoice_lock:
                if self._sensevoice is None:
                    log("SenseVoice", "首次调用，正在加载 SenseVoice 引擎...")
                    self._sensevoice = await asyncio.to_thread(self._load_sensevoice)
                    log("SenseVoice", "SenseVoice 引擎就绪！")
                self._sensevoice.set_language(str(payload.get("language") or "auto"))
                started_at = asyncio.get_running_loop().time()
                result = await asyncio.to_thread(self._sensevoice.transcribe, audio)
                return result, "", {
                    "queue_ms": round((started_at - queued_at) * 1000, 3),
                    "run_ms": round((asyncio.get_running_loop().time() - started_at) * 1000, 3),
                    "workers": 1,
                }

        if name != "qwen3-asr":
            raise ValueError(f"unknown ASR engine: {name}")
        await self._ensure_qwen()
        started_at = asyncio.get_running_loop().time()

        if self.use_microbatch and self._qwen_scheduler is not None:
            from local_inference.qwen_microbatch import QwenBatchInput
            result, context, draft, timing = await self._qwen_scheduler.submit(QwenBatchInput(
                audio=audio,
                language=str(payload.get("language") or "auto"),
                corpus_text=str(payload.get("corpus_text") or ""),
                context=str(payload.get("context") or state.context),
                draft_tokens=[] if payload.get("reset_draft") else list(state.draft_tokens),
                update_context=bool(payload.get("update_context", True)),
            ))
            state.context = context
            state.draft_tokens = draft
            total_ms = (asyncio.get_running_loop().time() - queued_at) * 1000
            timing["queue_ms"] = round(max(0.0, total_ms - float(timing.get("run_ms", 0.0))), 3)
            return result, state.context, timing
        else:
            # 桌面轻量单实例流水线：每个会话分配共享权重的独立 context worker
            # （独立 KV/草稿/上下文），避免引擎级全局状态在多客户端间串扰。
            async with self._qwen_lock:
                if state.qwen_worker is None:
                    try:
                        state.qwen_worker = await asyncio.to_thread(
                            self._qwen_engine.create_shared_context_worker
                        )
                    except Exception as exc:
                        log("Qwen3-ASR", f"[提示] 会话独立 worker 创建失败，退回共享实例（逐请求覆写状态）: {exc}")
                engine = state.qwen_worker or self._qwen_engine
                engine.set_language(str(payload.get("language") or "auto"))
                # 逐请求全量覆写（worker 路径同样保留：便宜，且使共享回退也安全）
                engine.set_corpus_text(str(payload.get("corpus_text") or ""))
                engine.set_context(str(payload.get("context") or state.context or ""))
                if payload.get("reset_draft"):
                    engine.reset_draft()

                update_ctx = bool(payload.get("update_context", True))
                def do_transcribe():
                    return engine.transcribe(audio, update_context=update_ctx)
                result = await asyncio.to_thread(do_transcribe)
                run_ms = round((asyncio.get_running_loop().time() - started_at) * 1000, 3)
                queue_ms = round(max(0.0, (started_at - queued_at) * 1000), 3)
                state.context = (getattr(engine, "context", "") or "") or state.context
                return result, state.context, {
                    "queue_ms": queue_ms,
                    "run_ms": run_ms,
                    "workers": 1,
                }


class HyMT2Backend:
    def __init__(self, models_root: Path, url: str, max_new_tokens: int, device: str = "auto") -> None:
        self.models_root = models_root
        self.url = url
        self.max_new_tokens = max_new_tokens
        self.preferred_device = device
        self._local_engine = None
        self._local_lock = asyncio.Lock()
        
        # 探测是否存在本地 GGUF 模型 (存在即可使用内置进程内引擎，无需外部 llama-server.exe)
        hymt2_dir = models_root / "hymt2"
        self.has_local_model = hymt2_dir.is_dir() and bool(list(hymt2_dir.glob("*.gguf")))

    def _get_local_engine(self):
        if self._local_engine is not None:
            return self._local_engine
        from streaming_translation.api.hymt2 import acquire_local_engine
        hymt2_path = next((self.models_root / "hymt2").glob("*.gguf"))
        
        target_device = self.preferred_device
        try:
            self._local_engine = acquire_local_engine(str(hymt2_path), device=target_device)
            return self._local_engine
        except Exception as exc:
            # 如果是显存不足 (OutOfDeviceMemory / allocate Vulkan buffer / bad alloc 等)，自动降级到 CPU
            if target_device != "cpu":
                log("Hy-MT2", f"[显存自愈] GPU 显存不足以容纳 Hy-MT2，正在自动降级至 CPU 运行...")
                try:
                    self._local_engine = acquire_local_engine(str(hymt2_path), device="cpu")
                    log("Hy-MT2", "[显存自愈] Hy-MT2 已成功在 CPU 模式下就绪，语音识别与翻译并行流畅！")
                    return self._local_engine
                except Exception as cpu_exc:
                    raise RuntimeError(f"GPU与CPU加载Hy-MT2均失败: {cpu_exc}") from exc
            raise exc

    async def complete(self, prompt: str) -> str:
        if self.has_local_model:
            # 优先使用进程内置 HyMT2 本地推理引擎 (免去 llama-server)
            async with self._local_lock:
                if self._local_engine is None:
                    log("Hy-MT2", "首次收到翻译请求，正在就地加载内置 Hy-MT2 本地模型...")
                    self._local_engine = await asyncio.to_thread(self._get_local_engine)
                    log("Hy-MT2", "内置 Hy-MT2 翻译引擎就绪！")

                full_prompt = CHAT_PREFIX + prompt + CHAT_SUFFIX
                def call_local() -> str:
                    clean, _ = self._local_engine.generate(full_prompt, max_tokens=self.max_new_tokens)
                    return clean
                text = await asyncio.to_thread(call_local)
                return _normalize_hypothesis(text)
        else:
            # 回退到外部 llama-server HTTP 调用
            body = json.dumps({
                "prompt": CHAT_PREFIX + prompt + CHAT_SUFFIX,
                "n_predict": self.max_new_tokens,
                "temperature": 0,
                "cache_prompt": True,
            }).encode("utf-8")

            def call_http() -> dict[str, Any]:
                request = urllib.request.Request(
                    self.url, data=body, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(request, timeout=600) as response:
                    return json.loads(response.read().decode("utf-8"))

            data = await asyncio.to_thread(call_http)
            return _normalize_hypothesis(str(data.get("content") or ""))


def _normalize_hypothesis(text: str) -> str:
    text = text.strip()
    for prefix in ("Translation:", "翻译：", "译文："):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return "" if text in {"<WAIT>", "<EMPTY>", "WAIT"} else text


def _source_text(payload: dict[str, Any]) -> str:
    explicit = payload.get("source")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    rows = list(payload.get("words") or []) + list((payload.get("tail") or {}).get("words") or [])
    return "".join(str(row[0]) for row in rows if isinstance(row, list) and row).strip()


def _source_history(items: Any, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    values: list[str] = []
    for item in items:
        raw = item if isinstance(item, str) else (
            item[0] if isinstance(item, (list, tuple)) and item else
            item.get("source", "") if isinstance(item, dict) else ""
        )
        text = str(raw or "").strip()
        if text:
            values.append(text)
    return values[-max(0, limit):]


def _translation_prompt(payload: dict[str, Any], source: str, history_limit: int) -> str:
    source_lang = str(payload.get("source_lang") or "en").lower()
    target_lang = str(payload.get("target_lang") or "zh").lower()
    source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    previous_source = str(payload.get("previous_source") or "")
    previous_translation = str(payload.get("previous_translation") or "")
    history = _source_history(payload.get("history"), history_limit)
    if not history and not previous_source and not previous_translation:
        return (
            f"Translate the following text from {source_name} into {target_name}. Note that you "
            "should only output the translated result without any additional explanation:\n\n"
            + source
        )
    background: list[str] = []
    if history:
        background.append("Recent source utterances:\n" + "\n".join(history))
    if previous_source:
        background.append("Previous version of the current source:\n" + previous_source)
    if previous_translation:
        background.append("Previous translation of the current source:\n" + previous_translation)
    background.append(
        "When the source meaning has not changed, preserve the still-correct prefix of the "
        "previous translation whenever possible. When content is added or corrected, accuracy "
        "and completeness take priority."
    )
    return (
        "[Background Information]\n" + "\n\n".join(background)
        + f"\n\nPlease translate the following text from {source_name} into {target_name}, "
        "taking the provided background information into consideration.\n\n[Source Text]\n"
        + source
    )


async def handle_asr(websocket, init: dict[str, Any], manager: ASREngineManager) -> None:
    client = get_client_str(websocket)
    engine = str(init.get("engine") or "")
    if engine not in ASR_ENGINES:
        log_err("ASR", f"来自 {client} 的请求指定了不支持的引擎: {engine}")
        await websocket.send(json.dumps({"type": "error", "code": "unsupported_engine"}))
        return
    state = ASRSessionState()
    await websocket.send(json.dumps({
        "type": "init_ok", "service": "asr", "protocol_version": PROTOCOL_VERSION,
        "engine": engine, "sample_rate": 16000,
    }))
    log("ASR", f"客户端 {client} 已建立 ASR 会话 (引擎: {engine})")
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if payload.get("type") != "transcribe":
                raise ValueError("transcribe_required")
            if payload.get("audio_format") != "f32le" or int(payload.get("sample_rate") or 0) != 16000:
                raise ValueError("unsupported_audio_format")
            # 大消息的 base64 解码移出事件循环，避免阻塞其他会话
            def decode_audio() -> np.ndarray:
                return np.frombuffer(base64.b64decode(payload.get("audio_base64") or ""), dtype="<f4").copy()
            audio = await asyncio.to_thread(decode_audio)
            audio_dur = round(len(audio) / 16000.0, 2)
            lang = payload.get("language") or "auto"
            req_id = payload.get("request_id", "-")
            log("ASR", f"接到来自 {client} 的转录请求 [req_id={req_id}] (引擎: {engine}, 音频: {audio_dur}s, 语种: {lang})，准备处理...")
            
            result, context, timing = await manager.transcribe(engine, state, audio, payload)
            text_preview = (result or {}).get("text", "")
            queue_ms = timing.get("queue_ms", 0.0)
            run_ms = timing.get("run_ms", 0.0)
            log("ASR", f"[{engine}] 处理完成 [req_id={req_id}] -> 结果: \"{text_preview}\" (排队: {queue_ms}ms, 推理: {run_ms}ms)")

            await websocket.send(json.dumps({
                "type": "recognition", "request_id": payload.get("request_id"),
                "engine": engine, "result": result, "context": context, "timing": timing,
            }, ensure_ascii=False))
        except Exception as exc:
            log_err("ASR", f"来自 {client} 的 ASR 请求处理失败: {exc}")
            await websocket.send(json.dumps({
                "type": "error", "code": "asr_request_failed", "message": str(exc),
            }, ensure_ascii=False))


async def handle_hymt2(
    websocket, init: dict[str, Any], backend: HyMT2Backend, history_limit: int
) -> None:
    client = get_client_str(websocket)
    source_lang = str(init.get("source_lang") or "en").lower()
    target_lang = str(init.get("target_lang") or "zh").lower()
    await websocket.send(json.dumps({
        "type": "init_ok", "service": "hymt2", "protocol_version": PROTOCOL_VERSION,
        "direction": f"{source_lang}->{target_lang}",
        "model": "Hy-MT2-1.8B-StreamRevise-v4-Q4_K_M",
        "hypothesis_mode": "sentence_revision", "source_token_join_mode": "verbatim",
        "prompt_mode": "standard",
    }))
    log("Hy-MT2", f"客户端 {client} 已建立翻译会话 ({source_lang} -> {target_lang})")
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if payload.get("type") != "update":
                raise ValueError("update_required")
            source = _source_text(payload)
            log("Hy-MT2", f"接到来自 {client} 的翻译请求 (原文: \"{source}\")，准备处理...")
            t0 = time.perf_counter()
            translation = await backend.complete(_translation_prompt(payload, source, history_limit))
            cost_ms = round((time.perf_counter() - t0) * 1000, 1)
            previous = str(payload.get("previous_translation") or "")
            is_final = bool(payload.get("is_final"))
            log("Hy-MT2", f"[{source_lang}->{target_lang}] 翻译完成 -> 译文: \"{translation or previous}\" ({cost_ms}ms)")

            await websocket.send(json.dumps({
                "type": "translation", "seq": payload.get("seq"),
                "source_lang": str(payload.get("source_lang") or source_lang).lower(),
                "target_lang": str(payload.get("target_lang") or target_lang).lower(),
                "committed_text": translation or previous, "committed_delta": "",
                "buffer_text": "", "covered_source_units": len(source),
                "stop_reason": "final" if is_final else ("revision" if translation else "wait"),
                "final": is_final,
            }, ensure_ascii=False))
        except Exception as exc:
            err_text = str(exc)
            if "10061" in err_text or "refused" in err_text.lower():
                user_msg = "服务端未开启 llama-server (18776端口未运行)，无法进行本地 Hy-MT2 翻译。请在服务端运行 run_hymt2_llama_server.bat 或在客户端设置中改用其他翻译API。"
                log_err("Hy-MT2", f"翻译失败：服务端 18776 端口未运行 llama-server！若需本地翻译请先启动后端，否则请在客户端改用云端翻译API。")
            else:
                user_msg = err_text
                log_err("Hy-MT2", f"来自 {client} 的翻译请求处理失败: {exc}")
            await websocket.send(json.dumps({
                "type": "error", "code": "hymt2_request_failed", "message": user_msg,
            }, ensure_ascii=False))


def _select_warmup_engines(
    mode: str, *, sv_ready: bool, qw_ready: bool, mt_ready: bool
) -> tuple[bool, bool, bool]:
    """返回 (warm_sensevoice, warm_qwen, warm_hymt2)。

    auto（默认）：Qwen3-ASR 是高频使用的主力引擎，就绪即预热消除首句延迟；
    SenseVoice 不预热，保持首次收到请求时才懒加载；Hy-MT2 翻译后端同步预热。
    all：全部就绪引擎都预热（含 SenseVoice）。none：不预热。
    """
    if mode == "none":
        return False, False, False
    if mode == "all":
        return sv_ready, qw_ready, mt_ready
    return False, qw_ready, mt_ready


async def main() -> None:
    args = parse_args()
    runtime_root = _require_dir(args.runtime_root, "runtime root")
    models_root = _require_dir(args.models_root, "models root")

    # 打印前台服务横幅与就绪信息
    print("=======================================================================")
    print("           Yakutan 远程推理服务端 (Windows 前台控制台)")
    print("=======================================================================")
    print(f"项目目录: {runtime_root}")
    print(f"模型目录: {models_root}")

    # 检查依赖
    missing_deps = check_runtime_dependencies()
    if missing_deps:
        print("\n[警告] 检测到缺少以下本地推理依赖库:")
        for m in missing_deps:
            print(f"  - 缺少: {m}")
        print("若模型加载失败，请执行: pip install -r requirements-local-inference.txt\n")

    configure_runtime(runtime_root, models_root)
    status = _model_status(models_root, args.llama_url)

    print("\n模型检测状态:")
    for name, s in status.items():
        state_str = "就绪 [OK]" if s.get("ready") else f"未就绪 (缺失/离线: {s.get('missing')})"
        print(f"  - {name:<12}: {state_str}")

    local_ips = get_local_ips()
    print("\n-----------------------------------------------------------------------")
    print(f"服务端监听: {args.host}:{args.port}")
    if args.host in ("127.0.0.1", "localhost", "::1"):
        print("当前仅监听本机回环地址；如需局域网连接，请使用 start_server.bat 或显式传 --host 0.0.0.0")
    else:
        print("供局域网客户端填写的 WebSocket 地址:")
        for ip in local_ips:
            print(f"   ► ws://{ip}:{args.port}")
        print("[安全提醒] 服务无鉴权，请仅在可信局域网内开放。")
    print("-----------------------------------------------------------------------")
    print("提示: 服务正在当前窗口前台运行，实时日志将打印在下方。")
    print("      如需停止服务，随时按 Ctrl+C 或 直接关闭此控制台窗口 即可完全退出！")
    print("=======================================================================\n")

    manager = ASREngineManager(
        models_root,
        use_microbatch=args.microbatch,
        qwen_workers=args.qwen_workers,
        qwen_batch_wait_ms=args.qwen_batch_wait_ms,
    )
    hymt2 = HyMT2Backend(models_root, args.llama_url, args.hymt2_max_new_tokens, device=args.hymt2_device)

    # 启动预热：Qwen3-ASR 为高频主力引擎，就绪即预热；SenseVoice 保持懒加载，
    # 首次收到对应引擎的请求时才加载（引擎加载逻辑在 ASREngineManager.transcribe 内）。
    warmup_mode = str(getattr(args, "warmup", "auto"))
    sv_ready = bool(status.get("sensevoice", {}).get("ready"))
    qw_ready = bool(status.get("qwen3-asr", {}).get("ready"))
    mt_ready = bool(hymt2.has_local_model)
    do_sv, do_qw, do_mt = _select_warmup_engines(
        warmup_mode, sv_ready=sv_ready, qw_ready=qw_ready, mt_ready=mt_ready
    )
    if warmup_mode == "auto" and sv_ready:
        log("预热", "SenseVoice 不预热，将在首次收到 SenseVoice 请求时懒加载")

    if warmup_mode != "none" and (do_sv or do_qw or do_mt):
        t_warmup = time.perf_counter()
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            if do_sv:
                log("预热", "正在预热 SenseVoice 识别引擎...")
                t0 = time.perf_counter()
                await manager.transcribe("sensevoice", ASRSessionState(), dummy_audio, {"language": "auto"})
                log("预热", f"SenseVoice 预热完毕（耗时: {round((time.perf_counter() - t0) * 1000, 1)}ms）！")
            if do_qw:
                log("预热", "正在预热 Qwen3-ASR 识别引擎...")
                t0 = time.perf_counter()
                dummy_payload = {"language": "zh", "corpus_text": "", "update_context": False, "reset_draft": True}
                await manager.transcribe("qwen3-asr", ASRSessionState(), dummy_audio, dummy_payload)
                log("预热", f"Qwen3-ASR 预热完毕（耗时: {round((time.perf_counter() - t0) * 1000, 1)}ms）！")
            if do_mt:
                log("预热", "正在预热 Hy-MT2 翻译引擎...")
                t0 = time.perf_counter()
                dummy_prompt = "Translate from Chinese to Japanese:\n\n你好"
                await hymt2.complete(dummy_prompt)
                log("预热", f"Hy-MT2 预热完毕（耗时: {round((time.perf_counter() - t0) * 1000, 1)}ms）！")

            total_cost = round((time.perf_counter() - t_warmup) * 1000, 1)
            log("预热", f"引擎预热完成（总计: {total_cost}ms）！客户端首次开麦与翻译均直接享受毫秒级响应。")
        except Exception as e:
            log_err("预热", f"预热过程异常 (不影响后续运行): {e}")

    async def connection(websocket) -> None:
        client = get_client_str(websocket)
        try:
            raw = await websocket.recv()
            message = json.loads(raw)
            if message.get("type") == "health":
                current_status = _model_status(models_root, args.llama_url)
                # ASR 就绪即认为基础服务就绪
                asr_ready = current_status.get("qwen3-asr", {}).get("ready", False) or current_status.get("sensevoice", {}).get("ready", False)
                await websocket.send(json.dumps({
                    "type": "health_ok", "protocol_version": PROTOCOL_VERSION,
                    "ready": asr_ready,
                    "models": current_status, "hostname": socket.gethostname(),
                }))
                return
            if message.get("type") != "init":
                log_err("连接", f"来自 {client} 的首条消息非 init: {message.get('type')}")
                await websocket.send(json.dumps({"type": "error", "code": "init_required"}))
                return
            service = str(message.get("service") or "hymt2")
            if service == "asr":
                await handle_asr(websocket, message, manager)
            elif service == "hymt2":
                await handle_hymt2(websocket, message, hymt2, args.history_limit)
            else:
                log_err("连接", f"来自 {client} 请求了未知服务: {service}")
                await websocket.send(json.dumps({"type": "error", "code": "unsupported_service"}))
        except Exception as exc:
            try:
                await websocket.send(json.dumps({
                    "type": "error", "code": "connection_failed", "message": str(exc),
                }, ensure_ascii=False))
            except Exception:
                pass
        finally:
            log("连接", f"客户端 {client} 会话结束或已断开")

    async with serve(
        connection, args.host, args.port, max_size=args.max_message_bytes,
        ping_interval=20, ping_timeout=20,
    ):
        log("系统", f"WebSocket 推理服务已成功监听在 ws://{args.host}:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[系统] 用户按下了 Ctrl+C，服务端已安全停止退出。")
        sys.exit(0)
