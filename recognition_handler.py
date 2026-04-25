"""
识别回调处理模块 - 负责语音识别事件的处理和翻译协调
"""
import asyncio
import logging
import threading
from typing import Optional

import config
from osc_manager import osc_manager
from speech_recognizers.base_speech_recognizer import (
    RecognitionEvent,
    SpeechRecognitionCallback,
)
from text_processor import (
    normalize_optional_language_code,
    normalize_lang_code,
    language_code_for_osc_tag,
    resolve_output_target_language,
    get_display_text,
    get_display_translation_text,
    build_streaming_output_line,
    build_dual_output_display,
)
from translation_pipeline import (
    is_streaming_deepl_hybrid_mode,
    translate_with_backend,
    reverse_translation,
)

logger = logging.getLogger(__name__)

PAUSE_RESUME_BACKENDS = {'qwen', 'soniox', 'doubao_file', 'local'}


def is_doubao_file_backend(backend: str) -> bool:
    return backend == 'doubao_file'


def is_effective_mic_control_enabled(backend: str) -> bool:
    return config.ENABLE_MIC_CONTROL or is_doubao_file_backend(backend)


def should_output_partial_results(backend: str) -> bool:
    return config.SHOW_PARTIAL_RESULTS and not is_doubao_file_backend(backend)


class VRChatRecognitionCallback(SpeechRecognitionCallback):
    """语音识别回调 — 处理部分结果和最终结果的翻译与输出。"""

    def __init__(self, state):
        """
        Args:
            state: AppState 实例，提供翻译器、executor 等运行时依赖。
        """
        self.state = state
        self.loop = None  # 将在主线程中设置
        self.last_partial_translation = None
        self.last_partial_translation_secondary = None
        self.translating_partial = False
        self.last_partial_source_segment = None
        self.pending_partial_segment = None
        self._partial_request_seq = 0
        self._latest_partial_request_id = 0
        self._partial_inflight = 0
        self._finalized_seq = 0
        self._final_output_version = 0
        self._final_translate_request_seq = 0
        self._async_result_seq = 0
        self._latest_applied_async_result_seq = 0
        self._session_generation = 0
        self._translate_ordering_lock = threading.Lock()
        self.partial_translation_update_count = 0
        self._prefer_deepl_on_next_final = False
        self._partial_debounce_handle: Optional[asyncio.TimerHandle] = None

    def mark_mute_finalization_requested(self) -> None:
        self._prefer_deepl_on_next_final = True

    def clear_mute_finalization_requested(self) -> None:
        self._prefer_deepl_on_next_final = False

    @staticmethod
    def _normalize_lang(lang):
        """标准化语言代码"""
        return normalize_lang_code(lang)

    @staticmethod
    def _should_translate(source_lang: Optional[str], target_lang: Optional[str]) -> bool:
        """当目标语言有效且与源语言不同，才需要执行翻译。"""
        normalized_target = normalize_optional_language_code(target_lang)
        if normalized_target is None:
            return False
        return normalize_lang_code(source_lang) != normalize_lang_code(normalized_target)

    @staticmethod
    def _extract_streaming_segment(text: str) -> Optional[str]:
        """从识别文本中截取可用于流式翻译的片段"""
        if not text:
            return None

        punctuation_chars = ("。", "？", "！", "，", "、", ".", "?", "!", ",")
        for idx in range(len(text) - 1, -1, -1):
            if text[idx] in punctuation_chars:
                remainder = text[idx + 1:]
                if remainder and remainder.strip():
                    segment = text[:idx + 1].strip()
                    if segment:
                        return segment
        return None

    @staticmethod
    def _should_trigger_partial_translation(segment: Optional[str]) -> bool:
        if not segment:
            return False

        min_chars = max(0, int(getattr(config, 'MIN_PARTIAL_TRANSLATION_CHARS', 2)))
        normalized_segment = segment.strip().rstrip("。？！，、.?!,… ")
        return len(normalized_segment) >= min_chars

    @staticmethod
    def _has_error_text(text: Optional[str]) -> bool:
        return bool(text) and "[ERROR]" in str(text)

    @staticmethod
    def _resolve_smart_targets(self_language: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        primary = None
        secondary = None
        
        primary_enabled = getattr(config, 'SMART_TARGET_PRIMARY_ENABLED', False)
        secondary_enabled = getattr(config, 'SMART_TARGET_SECONDARY_ENABLED', False)
        
        if not primary_enabled and not secondary_enabled:
            return None, None
        
        from translators.smart_target_language import get_smart_selector
        selector = get_smart_selector()
        min_samples = getattr(config, 'SMART_TARGET_LANGUAGE_MIN_SAMPLES', 1)
        
        if len(selector._history) < min_samples:
            return None, None
        
        smart_targets = selector.select_target_language(self_language)
        
        if primary_enabled and smart_targets:
            primary = smart_targets[0]
        
        if secondary_enabled and len(smart_targets) > 1:
            secondary = smart_targets[1]
        
        return primary, secondary

    @staticmethod
    def _filter_error_lines_for_osc(
        default_display_text: Optional[str],
        primary_raw_text: Optional[str] = None,
        primary_display_text: Optional[str] = None,
        primary_language: Optional[str] = None,
        secondary_raw_text: Optional[str] = None,
        secondary_display_text: Optional[str] = None,
        secondary_language: Optional[str] = None,
    ) -> Optional[str]:
        if not default_display_text:
            return default_display_text
        if getattr(config, 'OSC_SEND_ERROR_MESSAGES', False):
            return default_display_text

        primary_has_error = VRChatRecognitionCallback._has_error_text(primary_raw_text)
        secondary_has_error = VRChatRecognitionCallback._has_error_text(secondary_raw_text)
        if not primary_has_error and not secondary_has_error:
            return default_display_text

        visible_lines = []
        visible_languages = []
        if primary_display_text and not primary_has_error:
            visible_lines.append(primary_display_text)
            visible_languages.append(primary_language)
        if secondary_display_text and not secondary_has_error:
            visible_lines.append(secondary_display_text)
            visible_languages.append(secondary_language)

        if len(visible_lines) >= 2:
            return build_dual_output_display(
                visible_lines[0],
                visible_lines[1],
                visible_languages[0],
                visible_languages[1],
            )
        if len(visible_lines) == 1:
            return build_dual_output_display(visible_lines[0], None)
        return None

    def _cancel_partial_debounce(self) -> None:
        """取消尚未触发的流式翻译消抖定时器。"""

        def _cancel() -> None:
            if self._partial_debounce_handle is not None:
                self._partial_debounce_handle.cancel()
                self._partial_debounce_handle = None

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(_cancel)
        else:
            _cancel()

    def _next_async_result_seq(self) -> int:
        """为异步翻译结果分配一个全局递增序号。"""
        with self._translate_ordering_lock:
            self._async_result_seq += 1
            return self._async_result_seq

    def _get_session_generation(self) -> int:
        with self._translate_ordering_lock:
            return self._session_generation

    def _is_session_generation_current(self, session_generation: int) -> bool:
        with self._translate_ordering_lock:
            return session_generation == self._session_generation

    def _is_latest_partial_request(
        self,
        request_id: int,
        finalized_seq: int,
        final_output_version: int,
        session_generation: int,
    ) -> bool:
        with self._translate_ordering_lock:
            return (
                request_id == self._latest_partial_request_id
                and finalized_seq == self._finalized_seq
                and final_output_version == self._final_output_version
                and session_generation == self._session_generation
            )

    def _try_adopt_async_result(
        self, async_result_seq: int, session_generation: int,
    ) -> bool:
        with self._translate_ordering_lock:
            if session_generation != self._session_generation:
                return False
            if async_result_seq <= self._latest_applied_async_result_seq:
                return False
            self._latest_applied_async_result_seq = async_result_seq
            return True

    def _is_async_result_current(
        self, async_result_seq: int, session_generation: int,
    ) -> bool:
        with self._translate_ordering_lock:
            return (
                session_generation == self._session_generation
                and async_result_seq == self._latest_applied_async_result_seq
            )

    def _reset_partial_translation_state(self) -> None:
        self.last_partial_translation = None
        self.last_partial_translation_secondary = None
        self.last_partial_source_segment = None
        self.pending_partial_segment = None
        self._latest_partial_request_id = 0
        self.partial_translation_update_count = 0
        self._prefer_deepl_on_next_final = False

    def _schedule_partial_translation_with_debounce(
        self,
        segment,
        request_id,
        finalized_seq,
        final_output_version,
        async_result_seq,
        session_generation,
    ) -> None:
        """对流式翻译请求做消抖。"""
        if not self.loop:
            return

        debounce_ms = max(0, int(getattr(config, 'PARTIAL_TRANSLATION_DEBOUNCE_MS', 50)))
        debounce_seconds = debounce_ms / 1000.0

        def _dispatch() -> None:
            self._partial_debounce_handle = None
            if not self._is_latest_partial_request(
                request_id, finalized_seq, final_output_version, session_generation,
            ):
                return
            asyncio.create_task(
                self._translate_partial_task(
                    segment,
                    request_id,
                    finalized_seq,
                    final_output_version,
                    async_result_seq,
                    session_generation,
                )
            )

        def _schedule() -> None:
            if self._partial_debounce_handle is not None:
                self._partial_debounce_handle.cancel()
            self._partial_debounce_handle = self.loop.call_later(
                debounce_seconds, _dispatch,
            )

        self.loop.call_soon_threadsafe(_schedule)

    def _schedule_final_translation_task(
        self,
        *,
        text: str,
        source_lang: str,
        normalized_source: str,
        actual_target: Optional[str],
        actual_secondary_target: Optional[str],
        primary_should_translate: bool,
        secondary_should_translate: bool,
        use_secondary_output: bool,
        use_deepl_final: bool,
        previous_translation: Optional[str],
        previous_translation_secondary: Optional[str],
        previous_source_segment: Optional[str],
        request_id: int,
        async_result_seq: int,
        session_generation: int,
    ) -> bool:
        if not self.loop or not self.loop.is_running():
            return False

        def _dispatch() -> None:
            asyncio.create_task(
                self._translate_final_task(
                    text=text,
                    source_lang=source_lang,
                    normalized_source=normalized_source,
                    actual_target=actual_target,
                    actual_secondary_target=actual_secondary_target,
                    primary_should_translate=primary_should_translate,
                    secondary_should_translate=secondary_should_translate,
                    use_secondary_output=use_secondary_output,
                    use_deepl_final=use_deepl_final,
                    previous_translation=previous_translation,
                    previous_translation_secondary=previous_translation_secondary,
                    previous_source_segment=previous_source_segment,
                    request_id=request_id,
                    async_result_seq=async_result_seq,
                    session_generation=session_generation,
                )
            )

        self.loop.call_soon_threadsafe(_dispatch)
        return True

    def on_session_started(self) -> None:
        logger.info('Speech recognizer session opened.')
        self._cancel_partial_debounce()
        self._reset_partial_translation_state()
        self._final_output_version = 0
        self._final_translate_request_seq = 0
        with self._translate_ordering_lock:
            self._session_generation += 1

    def on_session_stopped(self) -> None:
        self._cancel_partial_debounce()
        with self._translate_ordering_lock:
            self._session_generation += 1
        logger.info('Speech recognizer session closed.')

    def on_error(self, error: Exception) -> None:
        logger.error('Speech recognizer failed: %s', error)

    async def _translate_partial_task(
        self,
        segment,
        request_id,
        finalized_seq,
        final_output_version,
        async_result_seq,
        session_generation,
    ):
        """异步流式翻译任务"""
        s = self.state
        self._partial_inflight += 1
        self.translating_partial = True
        try:
            if not self._is_session_generation_current(session_generation):
                return
            detected_lang_info = s.language_detector.detect(segment)
            detected_lang = detected_lang_info['language']
            smart_primary, smart_secondary = self._resolve_smart_targets(detected_lang)
            if smart_primary is not None:
                actual_target = resolve_output_target_language(detected_lang, smart_primary)
                actual_secondary_target = resolve_output_target_language(
                    detected_lang, normalize_optional_language_code(smart_secondary),
                )
            else:
                actual_target = resolve_output_target_language(detected_lang, config.TARGET_LANGUAGE)
                requested_secondary_target = normalize_optional_language_code(
                    getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)
                )
                actual_secondary_target = resolve_output_target_language(
                    detected_lang, requested_secondary_target,
                )
            use_secondary_output = (
                actual_secondary_target is not None and s.secondary_translator is not None
            )

            primary_should_translate = self._should_translate(detected_lang, actual_target)
            secondary_should_translate = (
                use_secondary_output
                and self._should_translate(detected_lang, actual_secondary_target)
            )

            loop = asyncio.get_running_loop()
            primary_future = None
            secondary_future = None

            translated_text = segment
            secondary_translated_text = segment if use_secondary_output else None

            if primary_should_translate:
                primary_future = loop.run_in_executor(
                    s.executor,
                    lambda: s.translator.translate(
                        segment,
                        source_language='auto',
                        target_language=actual_target,
                        context_prefix=config.CONTEXT_PREFIX,
                        is_partial=True,
                        record_history=False,
                        previous_translation=self.last_partial_translation,
                    ),
                )

            secondary_translator = s.secondary_translator
            if use_secondary_output and secondary_should_translate and secondary_translator is not None:
                secondary_future = loop.run_in_executor(
                    s.executor,
                    lambda: secondary_translator.translate(
                        segment,
                        source_language='auto',
                        target_language=actual_secondary_target,
                        context_prefix=config.CONTEXT_PREFIX,
                        is_partial=True,
                        record_history=False,
                        previous_translation=self.last_partial_translation_secondary,
                    ),
                )

            if primary_future is not None:
                translated_text = await primary_future

            if secondary_future is not None:
                secondary_translated_text = await secondary_future

            raw_translated_text = translated_text
            raw_secondary_translated_text = secondary_translated_text

            if not self._is_session_generation_current(session_generation):
                return

            if not self._try_adopt_async_result(async_result_seq, session_generation):
                return

            if primary_should_translate or secondary_should_translate:
                self.partial_translation_update_count += 1

            if primary_should_translate and translated_text and not translated_text.startswith("[ERROR]"):
                self.last_partial_translation = translated_text
            elif primary_should_translate:
                translated_text = ""
            else:
                self.last_partial_translation = None
                translated_text = segment

            if (
                use_secondary_output
                and secondary_should_translate
                and secondary_translated_text
                and not secondary_translated_text.startswith("[ERROR]")
            ):
                self.last_partial_translation_secondary = secondary_translated_text
            elif use_secondary_output and secondary_should_translate:
                secondary_translated_text = ""
            else:
                self.last_partial_translation_secondary = None
                secondary_translated_text = segment if use_secondary_output else ""

            self.last_partial_source_segment = segment

            display_translation = get_display_translation_text(
                translated_text,
                actual_target if primary_should_translate else detected_lang,
            )
            translation_display = build_streaming_output_line(display_translation)
            current_original_display = (
                s.subtitles_state.get("original", "") or build_streaming_output_line(segment)
            )

            if use_secondary_output and actual_secondary_target is not None:
                secondary_display_translation = get_display_translation_text(
                    secondary_translated_text,
                    actual_secondary_target if secondary_should_translate else detected_lang,
                )
                secondary_translation_display = build_streaming_output_line(
                    secondary_display_translation,
                )
                display_text = build_dual_output_display(
                    translation_display,
                    secondary_translation_display,
                    actual_target,
                    actual_secondary_target,
                )
            else:
                show_tag = primary_should_translate and getattr(
                    config, 'SHOW_ORIGINAL_AND_LANG_TAG', True,
                )
                if show_tag:
                    source_lang = language_code_for_osc_tag(detected_lang)
                    target_lang = language_code_for_osc_tag(actual_target)
                    display_text = (
                        f"[{source_lang}→{target_lang}] "
                        f"{translation_display} ({current_original_display})"
                    )
                    osc_text_max_length = config.get_effective_osc_text_max_length()
                    if (
                        osc_text_max_length is not None
                        and len(display_text) > osc_text_max_length
                    ):
                        display_text = f"[{source_lang}→{target_lang}] {translation_display}"
                else:
                    display_text = translation_display

            osc_text = self._filter_error_lines_for_osc(
                display_text,
                primary_raw_text=raw_translated_text if primary_should_translate else None,
                primary_display_text=translation_display,
                primary_language=actual_target if primary_should_translate else detected_lang,
                secondary_raw_text=(
                    raw_secondary_translated_text
                    if use_secondary_output and secondary_should_translate
                    else None
                ),
                secondary_display_text=(
                    secondary_translation_display
                    if use_secondary_output and actual_secondary_target is not None
                    else None
                ),
                secondary_language=(
                    actual_secondary_target if secondary_should_translate else detected_lang
                ) if use_secondary_output and actual_secondary_target is not None else None,
            )

            current_reverse_trans = s.subtitles_state.get("reverse_translated", "")

            if not self._is_async_result_current(async_result_seq, session_generation):
                return

            if use_secondary_output and actual_secondary_target is not None:
                s.update_subtitles(
                    current_original_display,
                    f"{translation_display}\n{secondary_translation_display}",
                    True,
                    current_reverse_trans,
                )
            else:
                s.update_subtitles(
                    current_original_display,
                    translation_display,
                    True,
                    current_reverse_trans,
                )

            if not self._is_async_result_current(async_result_seq, session_generation):
                return

            if osc_text:
                await osc_manager.send_text(osc_text, ongoing=True)

        except Exception:
            pass
        finally:
            self._partial_inflight = max(0, self._partial_inflight - 1)
            self.translating_partial = self._partial_inflight > 0
            if self._is_latest_partial_request(
                request_id, finalized_seq, final_output_version, session_generation,
            ):
                self.pending_partial_segment = None

    async def _translate_final_task(
        self,
        *,
        text: str,
        source_lang: str,
        normalized_source: str,
        actual_target: Optional[str],
        actual_secondary_target: Optional[str],
        primary_should_translate: bool,
        secondary_should_translate: bool,
        use_secondary_output: bool,
        use_deepl_final: bool,
        previous_translation: Optional[str],
        previous_translation_secondary: Optional[str],
        previous_source_segment: Optional[str],
        request_id: int,
        async_result_seq: int,
        session_generation: int,
    ) -> None:
        s = self.state
        try:
            if not self._is_session_generation_current(session_generation):
                return

            loop = asyncio.get_running_loop()
            translated_text = text
            secondary_translated_text = text if use_secondary_output else None

            primary_future = None
            secondary_future = None
            if primary_should_translate:
                primary_future = loop.run_in_executor(
                    s.executor,
                    lambda: translate_with_backend(
                        s.translator,
                        s.deepl_fallback_translator,
                        text,
                        actual_target,
                        previous_translation,
                        use_deepl_final,
                        previous_source_segment,
                        source_lang,
                        False,
                    ),
                )
            secondary_translator = s.secondary_translator
            secondary_deepl_fallback = s.secondary_deepl_fallback_translator
            if use_secondary_output and secondary_should_translate and secondary_translator is not None:
                secondary_future = loop.run_in_executor(
                    s.executor,
                    lambda: translate_with_backend(
                        secondary_translator,
                        secondary_deepl_fallback,
                        text,
                        actual_secondary_target,
                        previous_translation_secondary,
                        use_deepl_final,
                        previous_source_segment,
                        source_lang,
                        False,
                    ),
                )

            if primary_future is not None:
                translated_text = await primary_future
            if secondary_future is not None:
                secondary_translated_text = await secondary_future

            if not self._is_session_generation_current(session_generation):
                return

            if not self._try_adopt_async_result(async_result_seq, session_generation):
                return

            if primary_should_translate and translated_text and not str(
                translated_text,
            ).startswith("[ERROR]"):
                s.translator.append_history_entry(text, translated_text, actual_target)
            if (
                use_secondary_output
                and secondary_should_translate
                and secondary_translated_text
                and not str(secondary_translated_text).startswith("[ERROR]")
            ):
                s.secondary_translator.append_history_entry(
                    text, secondary_translated_text, actual_secondary_target,
                )

            display_source_text = get_display_text(text, source_lang)

            if not self._is_async_result_current(async_result_seq, session_generation):
                return

            is_translated = primary_should_translate or secondary_should_translate
            display_translated_text = None
            secondary_display_translated_text = None
            display_text = None

            if not is_translated:
                print(f'识别：{display_source_text}')
                display_text = display_source_text
                s.update_subtitles(display_source_text, "", False, "")
            else:
                print(f'主目标语言：{actual_target}')

                primary_display_language = (
                    actual_target if primary_should_translate else source_lang
                )
                display_translated_text = get_display_translation_text(
                    translated_text,
                    primary_display_language,
                )
                print(f'主译文：{display_translated_text}')

                if use_secondary_output and actual_secondary_target is not None:
                    print(f'第二目标语言：{actual_secondary_target}')
                    secondary_display_language = (
                        actual_secondary_target
                        if secondary_should_translate
                        else source_lang
                    )
                    secondary_display_translated_text = get_display_translation_text(
                        secondary_translated_text,
                        secondary_display_language,
                    )
                    print(f'第二译文：{secondary_display_translated_text}')

                if (
                    use_secondary_output
                    and secondary_display_translated_text is not None
                ):
                    display_text = build_dual_output_display(
                        display_translated_text,
                        secondary_display_translated_text,
                        actual_target,
                        actual_secondary_target,
                    )
                    subtitles_translated = (
                        f"{display_translated_text}\n"
                        f"{secondary_display_translated_text}"
                    )
                else:
                    show_tag = primary_should_translate and getattr(
                        config, 'SHOW_ORIGINAL_AND_LANG_TAG', True,
                    )
                    if show_tag:
                        tag_src = language_code_for_osc_tag(source_lang)
                        tag_tgt = language_code_for_osc_tag(actual_target)
                        display_text = (
                            f"[{tag_src}→{tag_tgt}] "
                            f"{display_translated_text} ({display_source_text})"
                        )
                        osc_text_max_length = config.get_effective_osc_text_max_length()
                        if (
                            osc_text_max_length is not None
                            and len(display_text) > osc_text_max_length
                        ):
                            display_text = f"[{tag_src}→{tag_tgt}] {display_translated_text}"
                    else:
                        display_text = str(display_translated_text)
                    subtitles_translated = str(display_translated_text)

                s.update_subtitles(
                    display_source_text,
                    subtitles_translated,
                    False,
                    "",
                )

            if not self._is_async_result_current(async_result_seq, session_generation):
                return

            osc_text = display_text
            if is_translated:
                osc_text = self._filter_error_lines_for_osc(
                    display_text,
                    primary_raw_text=translated_text if primary_should_translate else None,
                    primary_display_text=display_translated_text,
                    primary_language=(
                        actual_target if primary_should_translate else source_lang
                    ) if actual_target is not None else None,
                    secondary_raw_text=(
                        secondary_translated_text
                        if use_secondary_output and secondary_should_translate
                        else None
                    ),
                    secondary_display_text=secondary_display_translated_text,
                    secondary_language=(
                        actual_secondary_target if secondary_should_translate else source_lang
                    ) if use_secondary_output else None,
                )

            if osc_text:
                await osc_manager.send_text(osc_text, ongoing=False)

            if (
                primary_should_translate
                and config.ENABLE_REVERSE_TRANSLATION
                and self._is_async_result_current(async_result_seq, session_generation)
            ):
                reverse_translated_text = await loop.run_in_executor(
                    s.executor,
                    lambda: reverse_translation(
                        s.backwards_translator,
                        translated_text,
                        actual_target,
                        normalized_source,
                    ),
                )
                if (
                    reverse_translated_text is not None
                    and self._is_async_result_current(
                        async_result_seq, session_generation,
                    )
                ):
                    current_original = s.subtitles_state.get("original", "")
                    current_translated = s.subtitles_state.get("translated", "")
                    current_ongoing = s.subtitles_state.get("ongoing", False)
                    s.update_subtitles(
                        current_original,
                        current_translated,
                        current_ongoing,
                        str(reverse_translated_text),
                    )
        except Exception:
            logger.exception("Final translation task failed")

    def on_result(self, event: RecognitionEvent) -> None:
        s = self.state
        text = str(event.text or "").strip()
        if not text:
            return
        session_generation = self._get_session_generation()

        is_translated = False
        display_text = None
        is_ongoing = not event.is_final

        # 可能在翻译分支中赋值的变量
        display_translated_text = None
        secondary_display_translated_text = None
        display_source_text = None
        use_secondary_output = False
        actual_target = None
        normalized_source = None
        translated_text = None
        primary_translated = False

        if is_ongoing:
            print(f'部分：{text}', end='\r')
            display_text = get_display_text(text)
            current_trans = s.subtitles_state.get("translated", "")
            current_reverse_trans = s.subtitles_state.get("reverse_translated", "")
            s.update_subtitles(display_text, current_trans, True, current_reverse_trans)

            if config.ENABLE_TRANSLATION and getattr(config, 'TRANSLATE_PARTIAL_RESULTS', False):
                segment = self._extract_streaming_segment(text)
                if (
                    self._should_trigger_partial_translation(segment)
                    and segment != self.last_partial_source_segment
                    and segment != self.pending_partial_segment
                    and self.loop
                ):
                    with self._translate_ordering_lock:
                        self._partial_request_seq += 1
                        request_id = self._partial_request_seq
                        self._latest_partial_request_id = request_id
                    async_result_seq = self._next_async_result_seq()
                    self.pending_partial_segment = segment
                    final_output_version = self._final_output_version
                    self._schedule_partial_translation_with_debounce(
                        segment,
                        request_id,
                        self._finalized_seq,
                        final_output_version,
                        async_result_seq,
                        session_generation,
                    )

        else:
            if not config.ENABLE_TRANSLATION:
                source_lang_info = s.language_detector.detect(text)
                source_lang = source_lang_info['language']
                display_text = get_display_text(text, source_lang)
                print(f'识别：{display_text}')
                s.update_subtitles(display_text, "", is_ongoing, "")
            else:
                source_lang_info = s.language_detector.detect(text)
                source_lang = source_lang_info['language']

                normalized_source = self._normalize_lang(source_lang)
                smart_primary, smart_secondary = self._resolve_smart_targets(source_lang)
                if smart_primary is not None:
                    requested_target = smart_primary
                    requested_secondary_target = normalize_optional_language_code(smart_secondary)
                else:
                    requested_target = config.TARGET_LANGUAGE
                    requested_secondary_target = normalize_optional_language_code(
                        getattr(config, 'SECONDARY_TARGET_LANGUAGE', None)
                    )

                actual_target = resolve_output_target_language(source_lang, requested_target)
                actual_secondary_target = resolve_output_target_language(
                    source_lang, requested_secondary_target,
                )

                print(f'原文：{text} [{source_lang_info["language"]}]')
                if actual_target != normalize_optional_language_code(requested_target):
                    print(f'检测到主输出语言与源语言相同，使用备用语言: {config.FALLBACK_LANGUAGE}')
                if requested_secondary_target and actual_secondary_target != requested_secondary_target:
                    print(f'检测到第二输出语言与源语言相同，使用备用语言: {config.FALLBACK_LANGUAGE}')

                primary_should_translate = self._should_translate(source_lang, actual_target)

                use_deepl_final = False
                max_updates = max(0, int(getattr(config, 'STREAMING_FINAL_DEEPL_MAX_UPDATES', 2)))
                if (
                    self._prefer_deepl_on_next_final
                    and is_streaming_deepl_hybrid_mode()
                    and self.partial_translation_update_count <= max_updates
                    and s.deepl_fallback_translator is not None
                ):
                    use_deepl_final = True

                secondary_translator = s.secondary_translator
                secondary_deepl_fallback = s.secondary_deepl_fallback_translator
                use_secondary_output = (
                    actual_secondary_target is not None and secondary_translator is not None
                )
                secondary_should_translate = (
                    use_secondary_output
                    and self._should_translate(source_lang, actual_secondary_target)
                )

                # 在可能阻塞的终句翻译之前就递增，让上一句未返回的句内流式任务
                # 与未触发的 debounce 立即因 finalized_seq / final_output_version 失效，
                # 避免句间迟到 partial 在终句翻译期间仍刷新 OSC/字幕。
                self._finalized_seq += 1
                self._final_output_version += 1
                self._cancel_partial_debounce()

                if not primary_should_translate:
                    print('主输出语言与源语言相同，跳过翻译，直接输出原文')
                if use_secondary_output and not secondary_should_translate:
                    print('第二输出语言与源语言相同，跳过翻译，直接输出原文')

                will_network_translate = (
                    primary_should_translate or secondary_should_translate
                )
                my_final_tid = None
                if will_network_translate:
                    with self._translate_ordering_lock:
                        self._final_translate_request_seq += 1
                        my_final_tid = self._final_translate_request_seq

                previous_translation = self.last_partial_translation
                previous_translation_secondary = self.last_partial_translation_secondary
                previous_source_segment = self.last_partial_source_segment
                async_result_seq = self._next_async_result_seq()
                self._reset_partial_translation_state()

                if my_final_tid is not None and self._schedule_final_translation_task(
                    text=text,
                    source_lang=source_lang,
                    normalized_source=normalized_source,
                    actual_target=actual_target,
                    actual_secondary_target=actual_secondary_target,
                    primary_should_translate=primary_should_translate,
                    secondary_should_translate=secondary_should_translate,
                    use_secondary_output=use_secondary_output,
                    use_deepl_final=use_deepl_final,
                    previous_translation=previous_translation,
                    previous_translation_secondary=previous_translation_secondary,
                    previous_source_segment=previous_source_segment,
                    request_id=my_final_tid,
                    async_result_seq=async_result_seq,
                    session_generation=session_generation,
                ):
                    return

                if use_secondary_output:
                    translated_text = text
                    secondary_translated_text = text
                    if primary_should_translate:
                        translated_text = translate_with_backend(
                            s.translator,
                            s.deepl_fallback_translator,
                            text,
                            actual_target,
                            previous_translation,
                            use_deepl_final,
                            previous_source_segment,
                            source_lang,
                            False,
                        )
                    if secondary_should_translate and secondary_translator is not None:
                        secondary_translated_text = translate_with_backend(
                            secondary_translator,
                            secondary_deepl_fallback,
                            text,
                            actual_secondary_target,
                            previous_translation_secondary,
                            use_deepl_final,
                            previous_source_segment,
                            source_lang,
                            False,
                        )
                else:
                    if primary_should_translate:
                        translated_text = translate_with_backend(
                            s.translator,
                            s.deepl_fallback_translator,
                            text,
                            actual_target,
                            previous_translation,
                            use_deepl_final,
                            previous_source_segment,
                            source_lang,
                            False,
                        )
                    else:
                        translated_text = text
                    secondary_translated_text = None

                if primary_should_translate and translated_text and not str(
                    translated_text,
                ).startswith("[ERROR]"):
                    s.translator.append_history_entry(
                        text, translated_text, actual_target,
                    )
                if (
                    use_secondary_output
                    and secondary_should_translate
                    and secondary_translated_text
                    and not str(secondary_translated_text).startswith("[ERROR]")
                ):
                    secondary_translator.append_history_entry(
                        text, secondary_translated_text, actual_secondary_target,
                    )

                display_source_text = get_display_text(text, source_lang)

                primary_translated = primary_should_translate
                is_translated = primary_should_translate or secondary_should_translate

                if not is_translated:
                    print(f'识别：{display_source_text}')
                    display_text = display_source_text
                    s.update_subtitles(display_source_text, "", is_ongoing, "")
                else:
                    print(f'主目标语言：{actual_target}')

                    primary_display_language = (
                        actual_target if primary_should_translate else source_lang
                    )
                    display_translated_text = get_display_translation_text(
                        translated_text,
                        primary_display_language,
                    )
                    print(f'主译文：{display_translated_text}')

                    secondary_display_translated_text = None
                    if use_secondary_output and actual_secondary_target is not None:
                        print(f'第二目标语言：{actual_secondary_target}')
                        secondary_display_language = (
                            actual_secondary_target if secondary_should_translate else source_lang
                        )
                        secondary_display_translated_text = get_display_translation_text(
                            secondary_translated_text,
                            secondary_display_language,
                        )
                        print(f'第二译文：{secondary_display_translated_text}')

                    if use_secondary_output and secondary_display_translated_text is not None:
                        display_text = build_dual_output_display(
                            display_translated_text,
                            secondary_display_translated_text,
                            actual_target,
                            actual_secondary_target,
                        )
                    else:
                        show_tag = primary_should_translate and getattr(
                            config, 'SHOW_ORIGINAL_AND_LANG_TAG', True,
                        )
                        if show_tag:
                            tag_src = language_code_for_osc_tag(source_lang)
                            tag_tgt = language_code_for_osc_tag(actual_target)
                            display_text = (
                                f"[{tag_src}→{tag_tgt}] "
                                f"{display_translated_text} ({display_source_text})"
                            )
                            osc_text_max_length = config.get_effective_osc_text_max_length()
                            if (
                                osc_text_max_length is not None
                                and len(display_text) > osc_text_max_length
                            ):
                                display_text = (
                                    f"[{tag_src}→{tag_tgt}] {display_translated_text}"
                                )
                        else:
                            display_text = str(display_translated_text)

        if display_text is None:
            return

        osc_text = display_text
        if is_translated:
            osc_text = self._filter_error_lines_for_osc(
                display_text,
                primary_raw_text=translated_text if primary_should_translate else None,
                primary_display_text=display_translated_text,
                primary_language=(
                    actual_target if primary_should_translate else source_lang
                ) if actual_target is not None else None,
                secondary_raw_text=(
                    secondary_translated_text
                    if use_secondary_output and secondary_should_translate
                    else None
                ),
                secondary_display_text=secondary_display_translated_text,
                secondary_language=(
                    actual_secondary_target if secondary_should_translate else source_lang
                ) if use_secondary_output else None,
            )

        if is_translated:
            if use_secondary_output and secondary_display_translated_text is not None:
                s.update_subtitles(
                    display_source_text,
                    f"{display_translated_text}\n{secondary_display_translated_text}",
                    is_ongoing,
                    "",
                )
            else:
                s.update_subtitles(
                    display_source_text, str(display_translated_text), is_ongoing, "",
                )

        _should_send = (not is_ongoing) or should_output_partial_results(
            s.current_asr_backend,
        )

        if self.loop:
            if _should_send:
                if osc_text:
                    asyncio.run_coroutine_threadsafe(
                        osc_manager.send_text(osc_text, ongoing=is_ongoing),
                        self.loop,
                    )
            elif is_ongoing:
                asyncio.run_coroutine_threadsafe(
                    osc_manager.set_typing(is_ongoing),
                    self.loop,
                )
        else:
            print('[OSC] Warning: Event loop not set, cannot send OSC message.')

        if primary_translated and config.ENABLE_REVERSE_TRANSLATION:
            reverse_translated_text = reverse_translation(
                s.backwards_translator, translated_text, actual_target, normalized_source,
            )
            if reverse_translated_text is not None:
                current_original = s.subtitles_state.get("original", "")
                current_translated = s.subtitles_state.get("translated", "")
                current_ongoing = s.subtitles_state.get("ongoing", False)
                s.update_subtitles(
                    current_original,
                    current_translated,
                    current_ongoing,
                    str(reverse_translated_text),
                )
