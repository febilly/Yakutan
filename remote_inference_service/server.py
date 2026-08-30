#!/usr/bin/env python3
"""Unified WebSocket inference service for Yakutan's selectable local models."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.server import serve

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18775)
    parser.add_argument("--runtime-root", default=os.getenv("YAKUTAN_RUNTIME_ROOT", ""))
    parser.add_argument("--models-root", default=os.getenv("YAKUTAN_MODELS_ROOT", ""))
    parser.add_argument(
        "--llama-url",
        default=os.getenv("YAKUTAN_HYMT2_LLAMA_URL", "http://127.0.0.1:18776/completion"),
    )
    parser.add_argument("--max-message-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--hymt2-max-new-tokens", type=int, default=160)
    parser.add_argument("--history-limit", type=int, default=10)
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
    return parser.parse_args()


def _require_dir(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve() if value else None
    if path is None or not path.is_dir():
        raise SystemExit(f"{label} is missing or not a directory: {value!r}")
    return path


def configure_runtime(runtime_root: Path, models_root: Path) -> None:
    sys.path.insert(0, str(runtime_root))
    os.environ["YAKUTAN_QWEN_LLAMA_BIN"] = str(models_root / "qwen_llama_vulkan_bin")
    os.environ["YAKUTAN_LLAMA_ABI_PROFILE"] = "cuda-2026-08"
    os.environ.pop("GGML_VK_ALLOW_SYSMEM_FALLBACK", None)
    import config
    from local_inference import model_manager

    model_manager.MODELS_DIR = models_root
    config.LOCAL_INFERENCE_DEVICE = "auto"
    # CUDA EP is preferred on the server; encoder.py safely falls back to CPU
    # until onnxruntime-gpu is present.
    os.environ["YAKUTAN_ONNX_PROVIDER"] = "cuda"
    config.LOCAL_QWEN_ENCODER_DEVICE = "gpu"
    config.LOCAL_QWEN_LOG_PIPELINE_TIMING = True


def _model_status(models_root: Path) -> dict[str, dict[str, Any]]:
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
            models_root / "qwen_llama_vulkan_bin" / "libllama.so",
        ],
        "hymt2": list((models_root / "hymt2").glob("*.gguf")),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, paths in expected.items():
        missing = [str(path) for path in paths if not path.is_file()]
        if name == "hymt2" and not paths:
            missing = [str(models_root / "hymt2" / "*.gguf")]
        result[name] = {"ready": not missing, "missing": missing}
    return result


@dataclass
class ASRSessionState:
    context: str = ""
    draft_tokens: list[int] = field(default_factory=list)


class ASREngineManager:
    def __init__(self, models_root: Path, *, qwen_workers: int, qwen_batch_wait_ms: float) -> None:
        self.models_root = models_root
        self._sensevoice: Any | None = None
        self._sensevoice_lock = asyncio.Lock()
        self._qwen_workers = max(1, int(qwen_workers))
        self._qwen_load_lock = asyncio.Lock()
        self._qwen_loaded = False
        self._qwen_scheduler: Any | None = None
        self._qwen_batch_wait_ms = max(0.0, float(qwen_batch_wait_ms))

    @staticmethod
    def _load_sensevoice():
        from local_inference.asr_sensevoice import SenseVoiceEngine
        return SenseVoiceEngine()

    @staticmethod
    def _load_qwen_base():
        from local_inference.asr_qwen3 import Qwen3ASREngine
        return Qwen3ASREngine(corpus_text=None)

    async def _ensure_qwen_pool(self) -> None:
        if self._qwen_loaded:
            return
        async with self._qwen_load_lock:
            if self._qwen_loaded:
                return
            base = await asyncio.to_thread(self._load_qwen_base)
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
            self._qwen_loaded = True

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
                    self._sensevoice = await asyncio.to_thread(self._load_sensevoice)
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
        await self._ensure_qwen_pool()
        from local_inference.qwen_microbatch import QwenBatchInput
        if self._qwen_scheduler is None:  # defensive; _ensure_qwen_pool sets it atomically
            raise RuntimeError("Qwen microbatch scheduler failed to initialize")
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


class HyMT2Backend:
    def __init__(self, url: str, max_new_tokens: int) -> None:
        self.url = url
        self.max_new_tokens = max_new_tokens

    async def complete(self, prompt: str) -> str:
        body = json.dumps({
            "prompt": CHAT_PREFIX + prompt + CHAT_SUFFIX,
            "n_predict": self.max_new_tokens,
            "temperature": 0,
            "cache_prompt": True,
        }).encode("utf-8")

        def call() -> dict[str, Any]:
            request = urllib.request.Request(
                self.url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))

        data = await asyncio.to_thread(call)
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
    engine = str(init.get("engine") or "")
    if engine not in ASR_ENGINES:
        await websocket.send(json.dumps({"type": "error", "code": "unsupported_engine"}))
        return
    state = ASRSessionState()
    await websocket.send(json.dumps({
        "type": "init_ok", "service": "asr", "protocol_version": PROTOCOL_VERSION,
        "engine": engine, "sample_rate": 16000,
    }))
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if payload.get("type") != "transcribe":
                raise ValueError("transcribe_required")
            if payload.get("audio_format") != "f32le" or int(payload.get("sample_rate") or 0) != 16000:
                raise ValueError("unsupported_audio_format")
            audio = np.frombuffer(base64.b64decode(payload.get("audio_base64") or ""), dtype="<f4").copy()
            result, context, timing = await manager.transcribe(engine, state, audio, payload)
            await websocket.send(json.dumps({
                "type": "recognition", "request_id": payload.get("request_id"),
                "engine": engine, "result": result, "context": context, "timing": timing,
            }, ensure_ascii=False))
        except Exception as exc:
            await websocket.send(json.dumps({
                "type": "error", "code": "asr_request_failed", "message": str(exc),
            }, ensure_ascii=False))


async def handle_hymt2(
    websocket, init: dict[str, Any], backend: HyMT2Backend, history_limit: int
) -> None:
    source_lang = str(init.get("source_lang") or "en").lower()
    target_lang = str(init.get("target_lang") or "zh").lower()
    await websocket.send(json.dumps({
        "type": "init_ok", "service": "hymt2", "protocol_version": PROTOCOL_VERSION,
        "direction": f"{source_lang}->{target_lang}",
        "model": "Hy-MT2-1.8B-StreamRevise-v4-Q4_K_M",
        "hypothesis_mode": "sentence_revision", "source_token_join_mode": "verbatim",
        "prompt_mode": "standard",
    }))
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if payload.get("type") != "update":
                raise ValueError("update_required")
            source = _source_text(payload)
            translation = await backend.complete(_translation_prompt(payload, source, history_limit))
            previous = str(payload.get("previous_translation") or "")
            is_final = bool(payload.get("is_final"))
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
            await websocket.send(json.dumps({
                "type": "error", "code": "hymt2_request_failed", "message": str(exc),
            }, ensure_ascii=False))


async def main() -> None:
    args = parse_args()
    runtime_root = _require_dir(args.runtime_root, "runtime root")
    models_root = _require_dir(args.models_root, "models root")
    configure_runtime(runtime_root, models_root)
    status = _model_status(models_root)
    manager = ASREngineManager(
        models_root,
        qwen_workers=args.qwen_workers,
        qwen_batch_wait_ms=args.qwen_batch_wait_ms,
    )
    hymt2 = HyMT2Backend(args.llama_url, args.hymt2_max_new_tokens)

    async def connection(websocket) -> None:
        try:
            raw = await websocket.recv()
            message = json.loads(raw)
            if message.get("type") == "health":
                await websocket.send(json.dumps({
                    "type": "health_ok", "protocol_version": PROTOCOL_VERSION,
                    "ready": all(row["ready"] for row in status.values()),
                    "models": status, "hostname": socket.gethostname(),
                }))
                return
            if message.get("type") != "init":
                await websocket.send(json.dumps({"type": "error", "code": "init_required"}))
                return
            service = str(message.get("service") or "hymt2")
            if service == "asr":
                await handle_asr(websocket, message, manager)
            elif service == "hymt2":
                await handle_hymt2(websocket, message, hymt2, args.history_limit)
            else:
                await websocket.send(json.dumps({"type": "error", "code": "unsupported_service"}))
        except Exception as exc:
            try:
                await websocket.send(json.dumps({
                    "type": "error", "code": "connection_failed", "message": str(exc),
                }, ensure_ascii=False))
            except Exception:
                pass

    async with serve(
        connection, args.host, args.port, max_size=args.max_message_bytes,
        ping_interval=20, ping_timeout=20,
    ):
        print(json.dumps({
            "ready": True, "host": args.host, "port": args.port,
            "models": status, "hostname": socket.gethostname(),
        }), flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
