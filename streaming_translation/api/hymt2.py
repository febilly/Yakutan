"""
Hy-MT2 (HunYuan-MT 1.8B) real-time streaming translation via WebSocket.

Implements the stateless context-revision protocol documented in
`Hy-MT2-1.8B 实时文本流翻译服务接入文档` (INTEGRATION.md):

    Client -> Server : init / update (UTF-8 JSON text frames, one per frame)
    Server -> Client : init_ok / translation / error

Per the protocol the server keeps **no session state**: every ``update``
message is fully self-contained and carries the client-held revision chain
(``previous_source`` / ``previous_translation`` / ``history``).  This class
owns that chain for the sentence currently being translated, exactly like the
protocol prescribes.

The WebSocket address is intentionally **not** hard-coded: it is injected by
the caller (Web UI setting / ``HYMT2_WEBSOCKET_URL`` env var) and must be
filled in by the user.
"""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Dict, List, Optional

from .base import BaseTranslationAPI

try:
    from websockets.sync.client import connect as _sync_ws_connect
except ImportError:  # pragma: no cover
    raise ImportError(
        "websockets not installed. Run: pip install websockets>=12"
    )

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
MAX_HISTORY = 8        # 服务端接受 <=10 条，取 8 与本应用上下文窗口一致
HISTORY_KEEP = 20      # 本地保留的完成句对上限（截断后再取最新一条发送）

LANGUAGE_NAMES: Dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "it": "Italian",
}


def _local_backend_available() -> bool:
    """本地推理后端是否可用（仅本地推理版打包 / 源码模式带 local_inference 运行时）。"""
    try:
        from local_inference import is_local_inference_ui_enabled
    except Exception:
        return False
    try:
        return bool(is_local_inference_ui_enabled())
    except Exception:
        return False


def _sanitize_local_device(value) -> str:
    """把运行位置收敛成 'auto' / 'cpu' / 'vulkan:N'（不触碰 GPU 运行时）。"""
    normalized = str(value or "").strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("vulkan:") and normalized[len("vulkan:"):].isdigit():
        return normalized
    return "auto"


class HyMT2LocalEngine:
    """本地 Hy-MT2 GGUF 推理引擎（基于 llama.cpp / LlamaModel）。"""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        from local_inference.gpu_devices import describe_device, resolve_device
        from local_inference.model_manager import (
            get_hymt2_model_path,
            prepare_qwen_llama_runtime_env,
        )

        prepare_qwen_llama_runtime_env()
        path = model_path or get_hymt2_model_path()
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Hy-MT2 本地模型文件未找到: {path}")

        from local_inference.vendor.qwen_asr_gguf.inference import llama as llama_mod
        from local_inference.vendor.qwen_asr_gguf.inference.llama import (
            LlamaBatch,
            LlamaContext,
            LlamaModel,
            LlamaSampler,
        )

        self._llama_mod = llama_mod
        self._LlamaBatch = LlamaBatch

        # 解析运行位置：cpu -> 不卸载任何层；auto/vulkan:N -> 只用一张 GPU
        # （load_model 已固定 split_mode = NONE，不会把层切到核显上）
        n_gpu_layers, main_gpu, self.device = resolve_device(device)
        logger.info(
            "Hy-MT2 本地引擎运行位置: %s -> %s",
            self.device,
            describe_device(self.device),
        )

        self._LlamaSampler = LlamaSampler
        self.model = LlamaModel(str(path), n_gpu_layers=n_gpu_layers, main_gpu=main_gpu)
        self._n_ctx = 2048
        self.ctx = LlamaContext(self.model, n_ctx=self._n_ctx)
        self.eos_token = self.model.token_eos()
        self.stop_ids = {self.eos_token}
        for s in (
            "<｜hy_place▁holder▁no▁2｜>",
            "<｜hy_User｜>",
            "<｜hy_Assistant｜>",
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
        ):
            tid = self.model.token_to_id(s)
            if tid >= 0:
                self.stop_ids.add(tid)
        self._engine_lock = threading.Lock()

        from local_inference.spec_decode import SpeculativeDecoder

        self._spec = SpeculativeDecoder(llama_mod, self.ctx, self.model)
        # KV 里当前装着哪串 token（prompt + 已生成），供下一次比公共前缀
        self._cached_tokens: List[int] = []

    def _prefill(self, tokens: List[int], cached: List[int]) -> int:
        """提交 prompt，返回本次真正送进模型的 token 数。

        逐 token 调用 llama_decode 时每个 token 都要走一遍完整的 kernel 提交/
        同步往返，实测 100+ token 的 prompt 比批量提交慢 30-40 倍；
        批量语义与逐个等价（位置编码相同，只对末位 token 取 logits）。

        流式修订里相邻两次 prompt 的开头（背景/历史那段）是一样的，KV 里已经算过；
        只重算从第一处不同开始的尾巴，把前面留着。
        """
        n = len(tokens)
        if n <= 0:
            return 0
        if n > self._n_ctx:
            raise ValueError(f"prompt 超过上下文长度: {n} > {self._n_ctx}")

        from local_inference.spec_decode import common_prefix_len

        reuse = 0
        if self._prompt_cache_enabled() and cached:
            reuse = common_prefix_len(cached, tokens)
            # 末位必须重算：要拿它的 logits 起头
            reuse = min(reuse, n - 1)

        if reuse > 0:
            self._spec.seq_rm(reuse, -1)
        else:
            self.ctx.clear_kv_cache()

        todo = tokens[reuse:]
        self._spec.decode_tokens(todo, reuse, logits="last")
        return len(todo)

    @staticmethod
    def _prompt_cache_enabled() -> bool:
        import config as _config

        return bool(getattr(_config, "LOCAL_MT_PROMPT_CACHE", True))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        draft: Optional[List[int]] = None,
    ) -> tuple[str, List[int]]:
        """生成译文；返回 (清洗后的文本, 原始 token)。

        ``draft`` 是上一次中间结果的 token：流式修订里新旧译文大段重合，一次前向就能
        验证掉模型认同的部分。验证是精确比对，输出与逐 token 解码一致。
        """
        import config as _config

        with self._engine_lock:
            tokens = self.model.tokenize(prompt, add_special=False, parse_special=True)
            # 中途出错时 KV 与 _cached_tokens 会对不上，先作废；成功后再写回真实内容
            cached, self._cached_tokens = self._cached_tokens, []
            n_new = self._prefill(tokens, cached)

            sampler = self._LlamaSampler(temperature=0.0)
            try:
                use_draft = bool(getattr(_config, "LOCAL_SPECULATIVE_DECODE", True))
                gen = self._spec.generate(
                    sampler,
                    len(tokens),
                    stop_ids=self.stop_ids,
                    max_tokens=max_tokens,
                    draft=draft if use_draft else None,
                )
            finally:
                del sampler

            # KV 现在正好是 prompt + 生成的 token，下一次可以拿它比前缀
            self._cached_tokens = list(tokens) + list(gen.tokens)

            logger.debug(
                "Hy-MT2 本地生成: %s token / %s 次前向 (草稿 %s/%s), prompt 复用 %s/%s",
                len(gen.tokens), gen.n_passes, gen.n_accepted, gen.n_drafted,
                len(tokens) - n_new, len(tokens),
            )

            raw = self.model.detokenize(gen.tokens)
            clean = re.split(r"<[|｜].*?[|｜]>", raw)[0].strip()
            for prefix in ("Translation:", "翻译：", "译文："):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :].strip()
            return clean, list(gen.tokens)


# ── 进程内共享本地引擎注册表（引用计数） ─────────────────────────────────
# 主/次翻译器等同一进程内的多个翻译器共享同一本地模型实例，避免重复加载
# 同一份 GGUF 占用数倍内存；引用计数归零时才真正释放模型与显存/内存。
# 加载与释放均线程安全（推理跑在 executor 线程，配置热重载/停止跑在事件循环线程）。

_local_engine_registry: Dict[tuple, Dict] = {}
_local_engine_registry_lock = threading.Lock()


def _local_engine_key(
    model_path: Optional[str], device: str
) -> tuple:
    """按 (模型文件, 运行位置) 归并共享实例：同一路径同一设备只加载一次。"""
    path = model_path
    if not path:
        try:
            from local_inference.model_manager import get_hymt2_model_path

            path = get_hymt2_model_path()
        except Exception:  # pragma: no cover - 非本地构建
            path = None
    return (str(path) if path else "", _sanitize_local_device(device))


def _new_engine_entry() -> Dict:
    return {
        "engine": None,
        "refcount": 0,
        "loading": False,
        "load_error": None,
        "load_event": threading.Event(),
    }


def acquire_local_engine(
    model_path: Optional[str] = None, device: str = "auto"
) -> HyMT2LocalEngine:
    """获取一个本地引擎引用；尚无实例时负责加载（同参数并发调用只会加载一次）。

    调用成功即持有 1 个引用，必须配对的 :func:`release_local_engine`。
    等待中的调用方会按 key 重新检查最新状态（等待期间实例可能被释放后重建）。
    """
    key = _local_engine_key(model_path, device)
    while True:
        with _local_engine_registry_lock:
            entry = _local_engine_registry.get(key)
            if entry is None:
                entry = _new_engine_entry()
                _local_engine_registry[key] = entry
            if entry["engine"] is not None:
                entry["refcount"] += 1
                return entry["engine"]
            if entry["loading"]:
                # 有加载在进行：等它结束后回到循环顶部重新按 key 检查
                # （加载完成前实例可能被全部引用释放，需继续等新实例）
                event = entry["load_event"]
                wait_inflight = True
            else:
                event = None
                wait_inflight = False
                # 首个请求（或上次失败后的重试）：由本调用方负责加载
                entry["loading"] = True
                entry["load_error"] = None
                entry["load_event"].clear()
                entry["refcount"] += 1

        if wait_inflight:
            event.wait()
            continue

        try:
            engine = HyMT2LocalEngine(model_path, device=device)
        except Exception as exc:
            with _local_engine_registry_lock:
                entry["engine"] = None
                entry["load_error"] = exc
                entry["loading"] = False
                entry["refcount"] -= 1
                entry["load_event"].set()
            raise
        with _local_engine_registry_lock:
            entry["engine"] = engine
            entry["load_event"].set()
            entry["loading"] = False
            refcount = entry["refcount"]
        logger.info(
            "Hy-MT2 本地引擎已加载（进程内共享实例，引用计数 %d）: %s",
            refcount,
            key,
        )
        return engine


def release_local_engine(engine: Optional[HyMT2LocalEngine]) -> bool:
    """释放一个本地引擎引用；引用计数归零时真正卸载模型（释放内存/显存）。"""
    if engine is None:
        return False
    with _local_engine_registry_lock:
        disposed = False
        for key, entry in _local_engine_registry.items():
            if entry["engine"] is engine:
                entry["refcount"] -= 1
                if entry["refcount"] <= 0:
                    entry["refcount"] = 0
                    # 先从注册表摘除并置空，避免等待中的 acquire 拿到已卸载实例
                    entry["engine"] = None
                    del _local_engine_registry[key]
                    disposed = True
                break
        if not disposed:
            return False
    _dispose_local_engine(engine)
    logger.info("Hy-MT2 本地引擎已卸载（引用计数归零，模型内存已释放）")
    return True


def _dispose_local_engine(engine: HyMT2LocalEngine) -> None:
    """释放 llama.cpp 模型/上下文内存（通过包装对象的 __del__）。

    先等推理锁：若此刻有请求正在推理则等它结束，避免释放正在使用的模型。
    """
    engine_lock = getattr(engine, "_engine_lock", None)
    if engine_lock is not None:
        engine_lock.acquire()
    try:
        # 释放靠的是引用计数归零后包装对象的 __del__，所以每一个持有 ctx/model 的东西
        # 都得放手——推测解码器也持有它们，漏掉它显存就要留到进程退出才还。
        engine._spec = None
        engine._cached_tokens = []
        # 先释放上下文（llama_free），再释放模型（llama_model_free）
        engine.ctx = None
        engine.model = None
    finally:
        if engine_lock is not None:
            engine_lock.release()
    gc.collect()


def get_local_engine_runtime_status() -> Dict:
    """进程内本地引擎运行时状态快照（供 Web 前端展示加载/已加载）。"""
    with _local_engine_registry_lock:
        engines = [
            {
                "model_path": model_path or None,
                "device": device,
                "loaded": entry["engine"] is not None,
                "loading": bool(entry["loading"] and entry["engine"] is None),
                "refcount": int(entry["refcount"]),
            }
            for (model_path, device), entry in _local_engine_registry.items()
        ]
    return {
        "loaded": any(e["loaded"] for e in engines),
        "loading": any(e["loading"] for e in engines),
        "engines": engines,
    }


class HyMT2API(BaseTranslationAPI):
    """Hy-MT2 实时流式修订翻译（支持 WebSocket 外部服务与本地 GGUF 推理）。"""

    SUPPORTS_CONTEXT = True

    DEFAULT_SOURCE_LANG = "en"
    DEFAULT_TARGET_LANG = "zh"

    def __init__(
        self,
        websocket_url: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        proxy_url: Optional[str] = None,
        backend: str = "api",
        model_path: Optional[str] = None,
        local_device: str = "auto",
    ):
        self.backend = (backend or "api").lower().strip()
        if self.backend == "local" and not _local_backend_available():
            logger.warning(
                "Hy-MT2 本地推理运行时不可用（标准版不含本地模型支持），已回退到 API 后端"
            )
            self.backend = "api"
        self.websocket_url = (websocket_url or "").strip()
        self.timeout = float(timeout or 30.0)
        self.max_retries = int(max_retries if max_retries is not None else 3)
        self._proxy_url = proxy_url
        self._model_path = model_path
        self.local_device = _sanitize_local_device(local_device)
        self._local_engine: Optional[HyMT2LocalEngine] = None

        self._lock = threading.Lock()
        self._ws = None
        self._boot_ms = time.time() * 1000
        self.reset_session()
        self._ws = None
        self._boot_ms = time.time() * 1000
        self.reset_session()

    # ── Session / chain state ─────────────────────────────────────────

    def reset_session(self) -> None:
        """重置进行中的句子修订链（识别会话开始 / 应用重配置时调用）。

        已完成句子的 ``history`` 保留（仍可作为上下文），仅清空当前句。
        """
        with self._lock:
            self._seq = 0
            self._utterance_id = -1  # 首次调用自增后从 0 开始
            self._prev_source: Optional[str] = None
            self._prev_translation: Optional[str] = None
            # 上一次中间译文的 token，作为下一次修订的推测解码草稿（仅 local 后端用）
            self._prev_tokens: List[int] = []
            self._last_final = True
            self._sent_any = False
            self._last_source_lang: Optional[str] = None
            self._last_target_lang: Optional[str] = None
            self._utt_start_ms: Optional[float] = None
            self._history: List[List[str]] = []

    # ── Language helpers ───────────────────────────────────────────────

    @staticmethod
    def _normalize_lang(code: Optional[str]) -> str:
        """小写并去掉地区后缀（zh-CN -> zh），'auto'/空 -> ''。"""
        if not code:
            return ""
        norm = str(code).strip().lower()
        if not norm or norm == "auto":
            return ""
        return norm.split("-")[0]

    def _resolve_source_lang(self, source_language: str, detected: Optional[str]) -> str:
        code = self._normalize_lang(detected) or self._normalize_lang(source_language)
        if not code:
            # 兜底：沿用本连接最近一次已知的具体语言码
            code = self._last_source_lang or self.DEFAULT_SOURCE_LANG
        return code

    # ── Local engine lifecycle ─────────────────────────────────────────

    def load_local_engine(self) -> bool:
        """立即加载（或复用）本地模型；非 local 后端或已加载时直接返回。

        服务启动 / 切换为本地后端时由宿主调用，避免首次请求才加载。
        """
        if self.backend != "local" or self._local_engine is not None:
            return self._local_engine is not None
        self._local_engine = acquire_local_engine(
            self._model_path, device=self.local_device,
        )
        return True

    def unload_local_engine(self) -> bool:
        """放弃本实例对本地模型的引用；最后一个引用释放时真正卸载模型。

        返回是否成功释放了本实例持有的引用（是否真正卸载取决于引用计数）。
        """
        engine, self._local_engine = self._local_engine, None
        if engine is None:
            return False
        try:
            release_local_engine(engine)
            return True
        except Exception as e:
            logger.warning("Hy-MT2 本地引擎释放失败: %s", e)
            return False

    def close(self) -> None:
        """释放本实例持有的全部资源（本地模型引用 + WebSocket 连接）。"""
        self.unload_local_engine()
        self._close_ws()

    # ── Connection management ──────────────────────────────────────────

    def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _recv_json(self, ws) -> Dict:
        message = ws.recv(timeout=self.timeout)
        if isinstance(message, (bytes, bytearray)):
            message = bytes(message).decode("utf-8")
        data = json.loads(message)
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected non-object message: {data!r}")
        return data

    def _open_connection(self, source_lang: str, target_lang: str) -> None:
        """Open a WebSocket connection and complete the ``init`` handshake."""
        last_error: Optional[Exception] = None
        connect_kwargs: Dict = {"open_timeout": self.timeout}
        if self._proxy_url:
            connect_kwargs["proxy"] = self._proxy_url

        for attempt in range(self.max_retries + 1):
            ws = None
            try:
                ws = _sync_ws_connect(self.websocket_url, **connect_kwargs)
                init_payload = {
                    "type": "init",
                    "service": "hymt2",
                    "protocol_version": PROTOCOL_VERSION,
                    "source_lang": source_lang or self.DEFAULT_SOURCE_LANG,
                    "target_lang": target_lang or self.DEFAULT_TARGET_LANG,
                    "preset": "standard",
                    "hypothesis_mode": "sentence_revision",
                    "source_token_join_mode": "verbatim",
                }
                ws.send(json.dumps(init_payload, ensure_ascii=False))
                reply = self._recv_json(ws)
                if reply.get("type") != "init_ok":
                    raise RuntimeError(f"init failed: {reply}")
                if reply.get("hypothesis_mode") != "sentence_revision" or (
                    reply.get("source_token_join_mode") != "verbatim"
                ):
                    raise RuntimeError(
                        f"incompatible Hy-MT2 server (protocol check failed): {reply}"
                    )
                self._ws = ws
                if attempt:
                    logger.info(
                        "Hy-MT2 connection established after %d retries", attempt
                    )
                return
            except Exception as exc:  # noqa: BLE001 - surface as [ERROR]
                last_error = exc
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8.0))

        raise RuntimeError(
            f"cannot connect to Hy-MT2 service at {self.websocket_url!r}: "
            f"{last_error}"
        )

    def _request(self, payload: Dict) -> Dict:
        with self._lock:
            ws = self._ws
            if ws is None:
                self._open_connection(
                    payload.get("source_lang") or "",
                    payload.get("target_lang") or "",
                )
                ws = self._ws
            try:
                ws.send(json.dumps(payload, ensure_ascii=False))
                reply = self._recv_json(ws)
            except Exception:
                # 传输层失败：断开，让下一次请求重连（无状态，重连不丢链）
                self._close_ws()
                raise
            if reply.get("type") == "error":
                raise RuntimeError(f"Hy-MT2 server error: {reply}")
            if reply.get("type") != "translation":
                raise RuntimeError(f"unexpected Hy-MT2 message: {reply}")
            return reply

    # ── History helpers ────────────────────────────────────────────────

    @staticmethod
    def _strip_speaker_prefix(source: str) -> str:
        for prefix in ("[Me] ", "[Others] "):
            if source.startswith(prefix):
                return source[len(prefix):]
        return source

    @classmethod
    def _history_from_context_pairs(
        cls,
        context_pairs: Optional[List[Dict[str, str]]],
    ) -> List[List[str]]:
        """把 ContextAwareTranslator 的完成句对转成协议 history 格式。"""
        if not context_pairs:
            return []
        pairs: List[List[str]] = []
        for pair in context_pairs:
            source = cls._strip_speaker_prefix(str(pair.get("source") or "")).strip()
            target = str(pair.get("target") or "").strip()
            if source:
                pairs.append([source, target])
        return pairs[-MAX_HISTORY:]

    # ── Translation entry point ────────────────────────────────────────

    def _try_reuse_final(
        self, text: str, source_lang: str, target_lang: str
    ) -> Optional[str]:
        """终句原文与上一次中间结果完全一致（且语言方向一致）时，跳过这次
        重复请求/推理，直接复用上一次的译文，并按终句方式收尾修订链
        （记入历史、清空当前句 prev、标记终译），等价于正常终句响应的效果。"""
        with self._lock:
            if self._last_final or self._prev_source is None or self._prev_source != text:
                return None
            if self._last_source_lang != source_lang or self._last_target_lang != target_lang:
                return None
            display = (self._prev_translation or "").strip()
            if not display or display.startswith("[ERROR]"):
                return None
            print("[Hy-MT2] 终句原文与上次中间结果一致，复用上次译文（省一次调用）")
            if text.strip():
                self._history.append([text, display])
                self._history = self._history[-HISTORY_KEEP:]
            self._prev_source = None
            self._prev_translation = None
            self._prev_tokens = []
            self._last_final = True
            return display

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        context_pairs: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        if not text or not text.strip():
            return ""

        is_partial = bool(kwargs.get("is_partial", False))
        detected_source = kwargs.get("detected_source_language")
        source_lang = self._resolve_source_lang(source_language, detected_source)
        target_lang = self._normalize_lang(target_language) or (
            self._last_target_lang or self.DEFAULT_TARGET_LANG
        )
        history = self._history_from_context_pairs(context_pairs)

        if not is_partial:
            reused = self._try_reuse_final(text, source_lang, target_lang)
            if reused is not None:
                return reused

        if self.backend == "local":
            try:
                if self._local_engine is None:
                    # 兜底：宿主未预载时首次请求仍可用（走共享注册表）
                    self._local_engine = acquire_local_engine(
                        self._model_path, device=self.local_device,
                    )
                return self._translate_local(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    is_partial=is_partial,
                    history=history,
                )
            except Exception as e:
                logger.warning("Hy-MT2 local translate failed: %s", e)
                return f"[ERROR] Hy-MT2 本地模型错误: {e}"

        if not self.websocket_url:
            return (
                "[ERROR] Hy-MT2: WebSocket 地址未配置，请在网页「翻译API设置」"
                "中填写 Hy-MT2 WebSocket 地址（或设置 HYMT2_WEBSOCKET_URL）。"
            )

        try:
            payload, previous_translation = self._build_update(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                is_partial=is_partial,
                history=history,
            )
            reply = self._request(payload)

            committed = str(reply.get("committed_text") or "")
            buffer_text = str(reply.get("buffer_text") or "")
            display = (committed + buffer_text).strip()
            if not display:
                # stop_reason="wait"：无新内容，保持上一版显示（不返回空串）
                display = previous_translation or ""

            self._record_response(
                text=text,
                display=display,
                is_partial=is_partial,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            return display

        except Exception as e:
            logger.warning("Hy-MT2 translate failed: %s", e)
            return f"[ERROR] Hy-MT2: {e}"

    def _translate_local(
        self,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool,
        history: List[List[str]],
    ) -> str:
        with self._lock:
            if self._last_final:
                self._prev_source = None
                self._prev_translation = None
                self._prev_tokens = []

            prev_src = self._prev_source
            prev_trans = self._prev_translation
            recent_history = [p[0] for p in (self._history + history)][-MAX_HISTORY:]

            source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
            target_name = LANGUAGE_NAMES.get(target_lang, target_lang)

            if not recent_history and not prev_src and not prev_trans:
                direction = f"from {source_name} into {target_name}"
                content = (
                    f"Translate the following text {direction}. Note that you should "
                    "only output the translated result without any additional explanation:\n\n"
                    f"{text}"
                )
            else:
                background = []
                if recent_history:
                    background.append(
                        "Recent source utterances:\n" + "\n".join(recent_history)
                    )
                if prev_src:
                    background.append(
                        "Previous version of the current source:\n" + prev_src
                    )
                if prev_trans:
                    background.append(
                        "Previous translation of the current source:\n" + prev_trans
                    )
                background.append(
                    "When the source meaning has not changed, preserve the still-correct prefix "
                    "of the previous translation whenever possible. When content is added or "
                    "corrected, accuracy and completeness take priority."
                )
                direction = f"from {source_name} into {target_name}"
                content = (
                    "[Background Information]\n"
                    + "\n\n".join(background)
                    + f"\n\nPlease translate the following text {direction}, taking the provided "
                    "background information into consideration.\n\n[Source Text]\n"
                    + text
                )

            prompt = f"<｜hy_User｜>{content}<｜hy_Assistant｜>"
            if self._local_engine is None:
                display = ""
            else:
                inference_started = time.monotonic()
                display, out_tokens = self._local_engine.generate(
                    prompt, draft=self._prev_tokens or None
                )
                # 同一句的下一次修订拿这次的结果当草稿；整句定稿后这一句就翻篇了。
                self._prev_tokens = [] if not is_partial else out_tokens
                # 临时：本地模型每次推理后输出一行日志，记录本次耗时（毫秒）
                print(
                    f"[Hy-MT2 本地] 推理完成（{'最终' if not is_partial else '中间'}）: "
                    f"耗时 {(time.monotonic() - inference_started) * 1000.0:.0f}ms"
                )

            self._prev_source = text
            self._prev_translation = display
            self._last_final = not is_partial
            self._last_source_lang = source_lang
            self._last_target_lang = target_lang
            if not is_partial and text and display:
                self._history.append([text, display])
                self._history = self._history[-HISTORY_KEEP:]
            return display

    def _build_update(
        self,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool,
        history: List[List[str]],
    ) -> tuple[Dict, Optional[str]]:
        with self._lock:
            direction_changed = (
                (
                    self._last_source_lang is not None
                    and source_lang != self._last_source_lang
                )
                or (
                    self._last_target_lang is not None
                    and target_lang != self._last_target_lang
                )
            )
            new_utterance = (not self._sent_any) or self._last_final or direction_changed
            if new_utterance:
                self._utterance_id += 1
                self._prev_source = None
                self._prev_translation = None
                self._prev_tokens = []
                self._utt_start_ms = time.time() * 1000

            self._seq += 1
            self._sent_any = True
            previous_source = self._prev_source
            previous_translation = self._prev_translation
            local_history = self._history[-MAX_HISTORY:]

            self._last_source_lang = source_lang
            self._last_target_lang = target_lang
            self._last_final = not is_partial

            payload: Dict = {
                "type": "update",
                "seq": self._seq,
                "utterance_id": self._utterance_id,
                "words": [[text, None, None]],
                "clock_ms": int(
                    time.time() * 1000 - (self._utt_start_ms or self._boot_ms)
                ),
                "is_final": not is_partial,
                "source": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
            if previous_source is not None:
                payload["previous_source"] = previous_source
            if previous_translation is not None:
                payload["previous_translation"] = previous_translation
            history_send = history or local_history
            if history_send:
                payload["history"] = history_send
        return payload, previous_translation

    def _record_response(
        self,
        *,
        text: str,
        display: str,
        is_partial: bool,
        source_lang: str,
        target_lang: str,
    ) -> None:
        with self._lock:
            if not is_partial:
                # 句子已完成：记入历史，清空修订链（下一条 update 自动开新句）
                if text.strip() and display.strip():
                    self._history.append([text, display])
                    if len(self._history) > HISTORY_KEEP:
                        self._history = self._history[-HISTORY_KEEP:]
                self._prev_source = None
                self._prev_translation = None
                self._prev_tokens = []
            else:
                self._prev_source = text
                self._prev_translation = display
