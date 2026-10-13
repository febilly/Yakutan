"""
面向用户的错误汇总模块

识别服务各处（服务线程崩溃、识别后端连接失败、服务端断连等）的错误统一
写到这里：控制台打印一条带类型、可读原因的日志，同时保存"最近一次错误"，
供 /api/status 交给网页端以 toast 形式展示。

错误会按已知特征归类出一个 hint（例如 API Key 无效、模型不存在），
并尽量标明是哪个 API Key（key：dashscope / deepl / llm / soniox / doubao，
与网页端 API Keys 配置项一一对应）。网页端用 hint 和 key 查本地化的排查建议，
原始错误文本始终一并展示，方便定位。
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

__all__ = [
    "describe_exception",
    "describe_websocket_error",
    "classify_error",
    "asr_backend_key",
    "record_error",
    "get_last_error",
    "get_recent_errors",
    "clear_errors",
]

# 控制台打印用的 API Key 名称；网页端走 i18n（error.key.<key>）
KEY_NAMES_ZH = {
    "dashscope": "阿里云 DashScope API Key",
    "deepl": "DeepL API Key",
    "llm": "LLM API Key",
    "soniox": "Soniox API Key",
    "doubao": "豆包 API Key",
}

# 各识别后端用到的 API Key（本地识别不需要 Key）
_ASR_BACKEND_KEYS = {
    "qwen": "dashscope",
    "qwen_audio3": "dashscope",
    "dashscope": "dashscope",
    "soniox": "soniox",
    "doubao_file": "doubao",
}

# 控制台打印用的中文排查建议；网页端按同名 key 走 i18n（error.hint.<id>），{key} 换成 Key 名称
HINT_TEXT_ZH = {
    "invalidApiKey": "{key} 无效或已失效：请重新生成 API Key 并填入“API Keys 配置”，同时检查对应账户的余额是否充足，以及需要实名认证的账户是否已完成实名认证。",
    "modelNotExist": "服务端提示模型不存在：当前版本使用的模型可能已下线，请更新 Yakutan 到最新版本；同时确认 API Key 的地域（中国大陆/国际版）与所选模型一致。",
    "accountUnavailable": "{key} 对应的账户欠费、额度已用完或无权访问该模型：请到对应平台的控制台检查余额与模型开通状态。",
    "invalidCloseFrame": "识别服务的连接被异常断开，且返回内容无法解析。常见原因是加速器/代理干扰了连接（加速器请勿使用“进程模式”），也可能是账户欠费或无权限。",
    "network": "连接服务失败：请检查网络、代理/加速器以及防火墙设置。",
    "audioDevice": "打开麦克风失败：请检查麦克风是否被其他程序独占、是否在系统中被禁用，或在设置中重新选择麦克风。",
}

# 阿里云 DashScope 的 Key 分中国大陆/国际版，填错地域也会报 Key 无效
DASHSCOPE_REGION_NOTE_ZH = "阿里云的 API Key 分中国大陆版和国际版，请确认它与“国际版”开关一致。"

# 同一条错误在这么短时间内重复出现（例如断线重连风暴）时只累加计数，不生成新错误
_DEDUPE_WINDOW_SECONDS = 60.0

_lock = threading.Lock()
# 按去重键保存最近的几类错误（不同类错误交替出现时各自只提示一次）
_MAX_RECORDS = 10
_records: "OrderedDict[tuple, dict]" = OrderedDict()
_seq = 0

# 去重时忽略 request id 之类每次都不一样的串
_VOLATILE_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def describe_exception(error: Any, _depth: int = 0) -> str:
    """把异常变成一行可读文本：始终带异常类型；消息为空时注明，绝不返回空串。"""
    if error is None:
        return "未知错误"
    if not isinstance(error, BaseException):
        text = str(error).strip()
        return text or "未知错误"

    name = type(error).__name__
    message = str(error).strip()
    text = f"{name}: {message}" if message else f"{name}（异常未附带错误信息）"

    # 顺带带上 raise ... from 的根因（只追两层，防止异常链成环）
    cause = error.__cause__
    if cause is None and not error.__suppress_context__:
        cause = error.__context__
    if cause is not None and cause is not error and _depth < 2:
        cause_text = describe_exception(cause, _depth + 1)
        if cause_text not in text:
            text = f"{text}（原因: {cause_text}）"
    return text


def _decode_body(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", "replace")
    return str(body)


def describe_websocket_error(error: Any) -> str:
    """websocket-client 的异常补充可读信息（握手 HTTP 状态码 + 服务端返回的 code/message）。"""
    if error is None:
        return ""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        return describe_exception(error)

    body_text = _decode_body(getattr(error, "resp_body", None)).strip()
    detail = ""
    if body_text:
        try:
            payload = json.loads(body_text)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            code = payload.get("code") or payload.get("error_code")
            message = payload.get("message") or payload.get("error_msg")
            detail = " - ".join(str(part) for part in (code, message) if part)
        if not detail:
            detail = body_text[:300]
    text = f"握手失败 HTTP {status_code}"
    return f"{text}: {detail}" if detail else text


_INVALID_KEY_PATTERNS = (
    "invalidapikey",
    "invalid api-key",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "api key is invalid",
    "authentication fails",
    "authentication_error",
    "authenticationerror",
    "auth failed",
    "unauthorized",
    "http 401",
    "error code: 401",
    "status 401",
)

_ACCOUNT_PATTERNS = (
    "arrearage",
    "access denied",
    "accessdenied",
    "insufficient balance",
    "insufficient_balance",
    "insufficient_quota",
    "quota exceeded",
    "error code: 402",
    "error code: 403",
    "http 402",
    "http 403",
)


# 和具体 API Key 相关、需要点名是哪个 Key 的错误类别
_KEY_HINTS = frozenset({"invalidApiKey", "accountUnavailable"})


def asr_backend_key(backend: Optional[str]) -> Optional[str]:
    """识别后端用的是哪个 API Key；不需要 Key 的后端返回 None。"""
    return _ASR_BACKEND_KEYS.get(str(backend or "").strip())


def key_name_zh(key: Optional[str], key_detail: Optional[str] = None) -> str:
    """控制台用的 Key 名称；LLM 接的是 DeepSeek 时直接叫 DeepSeek API Key（和网页简易模式一致）。"""
    if key == "llm" and key_detail:
        if "deepseek" in key_detail.lower():
            return "DeepSeek API Key"
        return f"LLM API Key（{key_detail}）"
    return KEY_NAMES_ZH.get(key or "", "API Key")


def format_hint_zh(
    hint: Optional[str],
    key: Optional[str],
    key_detail: Optional[str] = None,
) -> str:
    """控制台用的排查建议文本（带上具体是哪个 Key）。"""
    template = HINT_TEXT_ZH.get(hint or "")
    if not template:
        return ""
    text = template.replace("{key}", key_name_zh(key, key_detail))
    if key == "dashscope" and hint == "invalidApiKey":
        text = f"{text}{DASHSCOPE_REGION_NOTE_ZH}"
    return text


def classify_error(text: str) -> Optional[str]:
    """按错误文本里的已知特征归类，返回 hint id；认不出时返回 None。"""
    lowered = (text or "").lower()
    if not lowered:
        return None
    if "打开麦克风失败" in lowered:
        return "audioDevice"
    if any(pattern in lowered for pattern in _INVALID_KEY_PATTERNS):
        return "invalidApiKey"
    if "model not exist" in lowered or "model_not_found" in lowered or "modelnotfound" in lowered:
        return "modelNotExist"
    if any(pattern in lowered for pattern in _ACCOUNT_PATTERNS):
        return "accountUnavailable"
    if "invalid close frame" in lowered:
        return "invalidCloseFrame"
    if (
        "could not established" in lowered
        or "getaddrinfo" in lowered
        or "name resolution" in lowered
        or "connection refused" in lowered
        or "connectionrefused" in lowered
        or "timed out" in lowered
        or "ssl" in lowered
        or "proxy" in lowered
    ):
        return "network"
    return None


def record_error(
    source: str,
    error: Any,
    *,
    hint: Optional[str] = None,
    key: Optional[str] = None,
    key_detail: Optional[str] = None,
) -> dict:
    """记录一条面向用户的错误并打印到控制台。返回记录（dict 副本）。

    ``error`` 可以是异常或已经整理好的文字；``hint`` 不传时按文本自动归类。
    ``key`` 标明出错的是哪个 API Key，``key_detail`` 是补充说明（例如 LLM 的接口地址）。
    """
    global _seq
    message = describe_exception(error)
    if hint is None:
        hint = classify_error(message)
    if hint not in _KEY_HINTS:
        # 只有 Key 相关的问题才点名 Key，免得把网络/麦克风问题误导成 Key 问题
        key = None
        key_detail = None
    now = time.time()

    # 能归类的错误按"来源 + Key + 类别"去重，归不了类的按原文（去掉 request id）去重
    if hint:
        dedupe_key = (str(source), key, hint)
    else:
        dedupe_key = (str(source), key, _VOLATILE_ID_RE.sub("<id>", message))

    with _lock:
        existing = _records.get(dedupe_key)
        if existing is not None and now - existing["last_at"] <= _DEDUPE_WINDOW_SECONDS:
            existing["count"] += 1
            existing["last_at"] = now
            existing["at_ms"] = int(now * 1000)
            existing["message"] = message
            _records.move_to_end(dedupe_key)
            return _public_record(existing)

        _seq += 1
        record = {
            "id": _seq,
            "source": str(source),
            "message": message,
            "hint": hint,
            "key": key,
            "key_detail": key_detail,
            "count": 1,
            "at_ms": int(now * 1000),
            "last_at": now,
        }
        _records[dedupe_key] = record
        _records.move_to_end(dedupe_key)
        while len(_records) > _MAX_RECORDS:
            _records.popitem(last=False)
        record = _public_record(record)

    key_text = key_name_zh(key, key_detail) if key else ""
    print(f"[错误] ({source}{', ' + key_text if key_text else ''}) {message}")
    hint_text = format_hint_zh(hint, key, key_detail)
    if hint_text:
        print(f"[错误] 排查建议: {hint_text}")
    return record


def _public_record(record: dict) -> dict:
    public = dict(record)
    public.pop("last_at", None)
    return public


def get_last_error() -> Optional[dict]:
    """最近一次出现（或重复出现）的错误。"""
    with _lock:
        if not _records:
            return None
        return _public_record(next(reversed(_records.values())))


def get_recent_errors() -> list:
    """最近的几类错误，按出现时间从旧到新。"""
    with _lock:
        return [_public_record(record) for record in _records.values()]


def clear_errors() -> None:
    """服务重新启动时清掉上一轮的错误，避免网页端把旧错误当成新错误。"""
    with _lock:
        _records.clear()
