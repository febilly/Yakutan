"""
LLM translation via any OpenAI-compatible endpoint.
Inlines the API-key rotation and client construction previously in
``openai_compat_client.py`` so the library is self-contained.
"""
from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List, Optional

from .base import BaseTranslationAPI
from .._proxy import detect_system_proxy

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai not installed. Run: pip install openai")


def merge_with_draft(fresh_translation: str, draft: str) -> str:
    """Merge *draft* (stable prefix) into *fresh_translation* (complete).

    Preserves the opening wording from *draft* (avoiding UI flicker)
    while using *fresh_translation* to guarantee content completeness.
    """
    if not draft or not fresh_translation:
        return fresh_translation
    if fresh_translation.startswith(draft):
        return fresh_translation
    if draft.startswith(fresh_translation):
        return fresh_translation
    common_len = 0
    for a, b in zip(draft, fresh_translation):
        if a == b:
            common_len += 1
        else:
            break
    if common_len >= max(len(draft) * 0.4, 3):
        return draft[:common_len] + fresh_translation[common_len:]
    return fresh_translation


def _parse_api_keys(raw: str) -> List[str]:
    return [k.strip() for k in raw.split(",") if k.strip()] if raw else []


def _resolve_raw_api_keys(
    llm_key: Optional[str] = None,
    openai_key: Optional[str] = None,
) -> str:
    if llm_key:
        return llm_key
    if openai_key:
        return openai_key
    env = os.environ.get("LLM_API_KEY", "")
    if env:
        return env
    env = os.environ.get("OPENAI_API_KEY", "")
    if env:
        return env
    return os.environ.get("OPENROUTER_API_KEY", "")


class _OpenAIClientManager:
    """Manages OpenAI client with optional API-key rotation."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        proxy_url: Optional[str] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.proxy_url = proxy_url or detect_system_proxy()

        self._key_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._api_keys = _parse_api_keys(api_key) if api_key else []
        self._key_index = 0
        self.client = self._build_client()

    def _build_client(self) -> OpenAI:
        kwargs = {"api_key": self.api_key, "base_url": self.base_url}
        if self.proxy_url:
            import httpx
            kwargs["http_client"] = httpx.Client(proxy=self.proxy_url)
        return OpenAI(**kwargs)

    def rotate_key(self) -> None:
        if len(self._api_keys) <= 1:
            return
        with self._key_lock:
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            next_key = self._api_keys[self._key_index]
        if next_key != self.api_key:
            with self._client_lock:
                if next_key != self.api_key:
                    self.api_key = next_key
                    self.client = self._build_client()


class OpenRouterAPI(BaseTranslationAPI):
    """LLM translation via an OpenAI-compatible endpoint.

    Supports both standard and streaming-optimised translation modes
    (controlled by the ``streaming_mode`` flag).
    """

    SUPPORTS_CONTEXT = True

    LANGUAGE_NAME_MAP = {
        "zh": "SIMPLIFIED CHINESE",
        "zh-cn": "SIMPLIFIED CHINESE",
        "zh-tw": "TRADITIONAL CHINESE",
        "zh-hans": "SIMPLIFIED CHINESE",
        "zh-hant": "TRADITIONAL CHINESE",
        "zh-hk": "TRADITIONAL CHINESE (Hong Kong)",
        "zh-mo": "TRADITIONAL CHINESE (Macau)",
        "zh-sg": "SIMPLIFIED CHINESE (Singapore)",
        "en": "ENGLISH",
        "en-us": "AMERICAN ENGLISH",
        "en-gb": "BRITISH ENGLISH",
        "ja": "JAPANESE",
        "ko": "KOREAN",
        "es": "SPANISH",
        "fr": "FRENCH",
        "de": "GERMAN",
        "id": "INDONESIAN",
        "ru": "RUSSIAN",
        "ar": "ARABIC",
        "pt": "PORTUGUESE",
        "th": "THAI",
        "tl": "TAGALOG (Philippines)",
        "it": "ITALIAN",
        "tr": "TURKISH",
        "fil": "FILIPINO/TAGALOG",
    }

    GENERIC_FORMALITY_STYLE_GUIDES = {
        "low": (
            "Keep the spoken style casual and relaxed, like close friends chatting. "
            "Use plain everyday wording with light courtesy, and avoid businesslike or overly polite phrasing unless the source clearly requires it."
        ),
        "medium": (
            "Keep the spoken style natural and friendly, but shift to everyday polite wording. "
            "Sound respectful and smooth for normal conversation with acquaintances or strangers, without becoming stiff or service-like."
        ),
        "high": (
            "Keep the translation natural and spoken, but use clearly polite, refined, and respectful wording instead of casual friend-chat phrasing. "
            "When the target language has politeness levels, choose the higher polite forms naturally, without becoming overly ceremonial unless the source requires it."
        ),
    }

    DEFAULT_FORMALITY = "medium"
    DEFAULT_STYLE = "light"

    JAPANESE_FORMALITY_STYLE_GUIDES = {
        "low": "For Japanese, use casual spoken Japanese in plain form.\nPrefer short, natural wording and avoid desu/masu unless needed.\nExamples: 「ちょっと待って。確認するね。わかった。」",
        "medium": "For Japanese, use standard everyday polite speech in basic desu/masu style.\nExamples: 「少し待ってください。確認します。わかりました。」",
        "high": "For Japanese, use clearly polite and refined teineigo.\nExamples: 「少々お待ちください。確認いたします。承知しました。」",
    }

    KOREAN_FORMALITY_STYLE_GUIDES = {
        "low": 'For Korean, use casual spoken Korean in close-friend tone.\nExamples: "잠깐만. 확인해볼게. 알겠어."',
        "medium": 'For Korean, use everyday polite spoken Korean in haeyo-che.\nExamples: "잠시만 기다려 주세요. 확인해 볼게요. 알겠어요."',
        "high": 'For Korean, use clearly formal polite Korean in habnida-che.\nExamples: "잠시만 기다려 주십시오. 확인하겠습니다. 알겠습니다."',
    }

    GENERIC_STYLE_GUIDES = {
        "standard": "Keep the sentence style natural, stable, and neutral for everyday conversation.",
        "light": "Keep the sentence style lively and chatty, with a bit more conversational momentum and warmth.",
    }

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 30,
        max_retries: int = 3,
        streaming_mode: bool = False,
        formality: str = "medium",
        style: str = "light",
        extra_body_json: str = "",
        parallel_fastest_mode: str = "off",
        proxy_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.streaming_mode = streaming_mode
        self.formality = formality
        self.style = style
        self.extra_body_json = extra_body_json
        self.parallel_fastest_mode = parallel_fastest_mode

        resolved_key = api_key or _resolve_raw_api_keys()
        if not resolved_key:
            raise ValueError(
                "LLM API Key not set. Pass api_key or set LLM_API_KEY / "
                "OPENAI_API_KEY / OPENROUTER_API_KEY."
            )
        self._client_mgr = _OpenAIClientManager(
            base_url=base_url,
            model=model,
            api_key=resolved_key,
            proxy_url=proxy_url,
        )

        app_url = os.environ.get("LLM_APP_URL", "") or os.environ.get("OPENROUTER_APP_URL", "")
        app_title = os.environ.get("LLM_APP_TITLE", "") or os.environ.get("OPENROUTER_APP_TITLE", "")
        self._default_headers = {}
        if app_url:
            self._default_headers["HTTP-Referer"] = app_url
        if app_title:
            self._default_headers["X-Title"] = app_title

    @property
    def client(self) -> OpenAI:
        return self._client_mgr.client

    # ── Language / style helpers ──────────────────────────────────────

    @classmethod
    def _describe_language(cls, code: str) -> str:
        norm = (code or "").strip().lower()
        if not norm or norm == "auto":
            return "auto-detected source language"
        name = cls.LANGUAGE_NAME_MAP.get(norm, norm.upper())
        return f"{name} ({code})"

    @classmethod
    def _get_formality_guide(cls, target_language: str) -> str:
        form = str(cls.DEFAULT_FORMALITY).strip().lower()
        norm_lang = (target_language or "").strip().lower()
        if norm_lang.startswith("ja"):
            guides = cls.JAPANESE_FORMALITY_STYLE_GUIDES
        elif norm_lang.startswith("ko"):
            guides = cls.KOREAN_FORMALITY_STYLE_GUIDES
        else:
            guides = cls.GENERIC_FORMALITY_STYLE_GUIDES
        if form not in guides:
            form = cls.DEFAULT_FORMALITY
        return guides.get(form, guides[cls.DEFAULT_FORMALITY])

    @classmethod
    def _get_style_guide(cls, target_language: str) -> str:
        style = str(cls.DEFAULT_STYLE).strip().lower()
        if style not in cls.GENERIC_STYLE_GUIDES:
            style = cls.DEFAULT_STYLE
        return cls.GENERIC_STYLE_GUIDES.get(style, cls.GENERIC_STYLE_GUIDES[cls.DEFAULT_STYLE])

    # ── Prompt building ───────────────────────────────────────────────

    def _build_system_prompt(self, target_descriptor: str, target_language: str) -> str:
        style_guide = self._get_formality_guide(target_language)
        sentence_guide = self._get_style_guide(target_language)
        return (
            f"You are a VRChat voice chat translator. "
            f"Translate the user's message into {target_descriptor}.\n\n"
            f"- Output ONLY in {target_descriptor}. No source-language words.\n"
            "- Translate EVERY part completely. Never skip or shorten.\n"
            "- Use friendly spoken style. Avoid robotic or textbook phrasing.\n"
            f"- Formality guide: {style_guide}\n"
            f"- Sentence style guide: {sentence_guide}\n"
            "- For idioms/slang: translate the meaning naturally.\n"
            "- Output the translation only. No labels, notes, or commentary."
        )

    @staticmethod
    def _extract_vrcx_context(context: Optional[str]) -> str:
        if not context:
            return ""
        start_marker = "<VRCHAT_CONTEXT>"
        end_marker = "</VRCHAT_CONTEXT>"
        start = context.find(start_marker)
        end = context.find(end_marker, start + len(start_marker))
        if start < 0 or end <= start:
            return ""
        return context[start + len(start_marker):end].strip()[:3000].rstrip()

    @staticmethod
    def _build_context_block(
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
    ) -> Optional[str]:
        blocks = []
        vrcx_context = OpenRouterAPI._extract_vrcx_context(context)
        if vrcx_context:
            blocks.append(
                "VRChat/VRCX local context for names, world and references. "
                "Use only for disambiguation; do not output it:\n"
                f"<VRCHAT_CONTEXT>\n{vrcx_context}\n</VRCHAT_CONTEXT>"
            )

        if context_pairs:
            parts = ["Conversation so far:"]
            for p in context_pairs:
                parts.append(f"  {p['source']} → {p['target']}")
            blocks.append("\n".join(parts))
        if context and context.strip():
            conversation_context = context
            if vrcx_context:
                start_marker = "<VRCHAT_CONTEXT>"
                end_marker = "</VRCHAT_CONTEXT>"
                start = conversation_context.find(start_marker)
                end = conversation_context.find(end_marker, start + len(start_marker))
                if start >= 0 and end > start:
                    conversation_context = (
                        conversation_context[:start]
                        + conversation_context[end + len(end_marker):]
                    )
            conversation_context = conversation_context.strip()
            if conversation_context:
                if not context_pairs:
                    blocks.append(f"Conversation so far:\n{conversation_context}")
                else:
                    blocks.append(conversation_context)
        return "\n\n".join(blocks) if blocks else None

    @staticmethod
    def _build_extra_body(raw: str) -> Dict:
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _strip_trailing_partial_ellipsis(text: str) -> str:
        if not text or text.startswith("[ERROR]"):
            return text
        return re.sub(r"(?:\.{3,}|…+)\s*$", "", text.rstrip())

    # ── Completeness check (v12 smart-hybrid) ─────────────────────────

    _CJK_PREFIXES = ("zh", "ja", "ko")

    @classmethod
    def _is_cjk(cls, code: str) -> bool:
        norm = (code or "").strip().lower()
        return any(norm == p or norm.startswith(p + "-") for p in cls._CJK_PREFIXES)

    @staticmethod
    def _check_content_completeness(
        source_text: str,
        translation: str,
        previous_translation: str,
        previous_source_text: Optional[str] = None,
        detected_source_language: str = "auto",
        target_language: str = "",
    ):
        source_is_cjk = OpenRouterAPI._is_cjk(detected_source_language)
        target_is_cjk = OpenRouterAPI._is_cjk(target_language)
        cross = source_is_cjk != target_is_cjk
        ratio_threshold = 0.15 if cross else 0.5

        ratio = len(translation) / max(len(source_text), 1)
        if ratio < ratio_threshold:
            return True, f"ratio={ratio:.2f}<{ratio_threshold}"

        prev_source_len = (
            len(previous_source_text)
            if previous_source_text
            else len(previous_translation)
        )
        source_vs_prev = len(source_text) / max(prev_source_len, 1)
        trans_growth = len(translation) / max(len(previous_translation), 1)
        source_added = len(source_text) - prev_source_len

        if source_vs_prev >= 1.3 and source_added >= 3 and trans_growth < 1.05:
            return True, (
                f"source_vs_prev={source_vs_prev:.2f} (+{source_added}chars) "
                f"but trans_growth={trans_growth:.2f}"
            )

        return False, ""

    # ── API call ──────────────────────────────────────────────────────

    def _execute_completion(self, request_kwargs: Dict) -> str:
        try:
            self._client_mgr.rotate_key()
            completion = self.client.chat.completions.create(**request_kwargs)
            if completion.choices and completion.choices[0].message.content:
                return self._clean_response(completion.choices[0].message.content)
            return "[ERROR] Empty response from model"
        except Exception as e:
            return f"[ERROR] {e}"

    @staticmethod
    def _clean_response(text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\r\n", "\n")
        while True:
            start = cleaned.find("<think>")
            if start == -1:
                break
            end = cleaned.find("</think>", start)
            if end == -1:
                cleaned = cleaned[:start]
                break
            cleaned = cleaned[:start] + cleaned[end + len("</think>"):]
        return cleaned.lstrip("\n").strip()

    def _should_parallel_fastest(self, is_partial: bool) -> bool:
        mode = self.parallel_fastest_mode
        if mode == "off":
            return False
        if mode == "all":
            return True
        if mode == "final_only":
            return not (self.streaming_mode and is_partial)
        return False

    def _call_api(self, messages: List[Dict], is_partial: bool = False) -> str:
        kwargs: Dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        extra = self._build_extra_body(self.extra_body_json)
        if "openrouter.ai" in self._client_mgr.base_url.lower():
            extra.setdefault("provider", {"sort": "latency"})
        if extra:
            kwargs["extra_body"] = extra

        if self._should_parallel_fastest(is_partial):
            executor = ThreadPoolExecutor(max_workers=2)
            futures = [
                executor.submit(self._execute_completion, dict(kwargs)),
                executor.submit(self._execute_completion, dict(kwargs)),
            ]
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            result = next(iter(done)).result()
            executor.shutdown(wait=False, cancel_futures=True)
            return result

        return self._execute_completion(kwargs)

    # ── Translation entry points ──────────────────────────────────────

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

        if self.streaming_mode:
            return self._translate_streaming(
                text, source_language, target_language, context, context_pairs, **kwargs
            )
        return self._translate_standard(
            text, source_language, target_language, context, context_pairs, **kwargs
        )

    def _translate_standard(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
        **kwargs,
    ) -> str:
        descriptor = self._describe_language(target_language)
        system_prompt = self._build_system_prompt(descriptor, target_language)
        context_block = self._build_context_block(context, context_pairs)

        parts = []
        if context_block:
            parts.append(context_block)
        parts.append(f"Translate this: {text}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        
        # print(messages)
        
        return self._call_api(messages, is_partial=False)

    def _translate_streaming(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: Optional[str],
        context_pairs: Optional[List[Dict[str, str]]],
        **kwargs,
    ) -> str:
        """v12 smart-hybrid streaming translation."""
        previous_translation = kwargs.get("previous_translation")
        is_partial = kwargs.get("is_partial", False)
        previous_source_text = kwargs.get("previous_source_text")
        detected_source_language = kwargs.get("detected_source_language", "auto")
        descriptor = self._describe_language(target_language)

        system_prompt = self._build_system_prompt(descriptor, target_language)
        context_block = self._build_context_block(context, context_pairs)

        parts = []
        if context_block:
            parts.append(context_block)

        if previous_translation:
            if is_partial:
                parts.append(
                    f"Your previous translation: {previous_translation.strip()}\n"
                    "Source text has been updated below. Translate the full updated text. "
                    "Keep wording consistent where meaning hasn't changed."
                )
            else:
                parts.append(
                    f"You previously translated part of this as: {previous_translation.strip()}\n"
                    "Now the complete sentence has arrived. "
                    "Translate the COMPLETE source text below. "
                    "Start your translation the same way as your previous version, "
                    "then continue translating the rest of the sentence."
                )

        parts.append(f"Translate this: {text}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

        # print(messages)

        stable = self._call_api(messages, is_partial=is_partial)

        if is_partial:
            return self._strip_trailing_partial_ellipsis(stable)

        if stable.startswith("[ERROR]") or not previous_translation:
            return stable

        needs_rescue, reason = self._check_content_completeness(
            text, stable, previous_translation,
            previous_source_text=previous_source_text,
            detected_source_language=detected_source_language,
            target_language=target_language,
        )

        if not needs_rescue:
            return stable

        rescue_parts = []
        if context_block:
            rescue_parts.append(context_block)
        rescue_parts.append(f"Translate this: {text}")

        rescue_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(rescue_parts)},
        ]

        fresh = self._call_api(rescue_messages, is_partial=False)

        if fresh.startswith("[ERROR]") or len(fresh) <= len(stable):
            return stable

        return merge_with_draft(fresh, stable)


class OpenRouterStreamingAPI(OpenRouterAPI):
    """Streaming-optimised alias of :class:`OpenRouterAPI` (sets ``streaming_mode=True``)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 30,
        max_retries: int = 3,
        formality: str = "medium",
        style: str = "light",
        extra_body_json: str = "",
        parallel_fastest_mode: str = "off",
        proxy_url: Optional[str] = None,
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            streaming_mode=True,
            formality=formality,
            style=style,
            extra_body_json=extra_body_json,
            parallel_fastest_mode=parallel_fastest_mode,
            proxy_url=proxy_url,
        )
