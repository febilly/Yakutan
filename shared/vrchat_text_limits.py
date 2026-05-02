"""Shared VRChat chatbox text limits and prefix-trimming helpers."""

from __future__ import annotations

from typing import Iterable, Optional

VRCHAT_OSC_TEXT_MAX_LENGTH = 144

# Ordered longest-first where punctuation can overlap.
TEXT_PREFIX_DROP_BOUNDARIES = (
    "……",
    "...",
    "。",
    "？",
    "！",
    ".",
    "?",
    "!",
    "…",
    "‽",
    "，",
    ",",
    "、",
    "；",
    ";",
    "：",
    ":",
    "։",
    "؟",
    "،",
    "؛",
    "۔",
    "।",
    "॥",
    "።",
    "။",
    "།",
    "‚",
    "٫",
)


def normalize_osc_text_max_length(
    value: object,
    default: int = VRCHAT_OSC_TEXT_MAX_LENGTH,
) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    return max(1, normalized)


def _iter_marker_end_positions(text: str, markers: Iterable[str]):
    marker_list = tuple(markers)
    index = 0
    while index < len(text):
        matched_marker = None
        for marker in marker_list:
            if marker and text.startswith(marker, index):
                matched_marker = marker
                break
        if matched_marker is None:
            index += 1
            continue
        index += len(matched_marker)
        yield index


def _iter_whitespace_end_positions(text: str):
    index = 0
    while index < len(text):
        if not text[index].isspace():
            index += 1
            continue
        while index < len(text) and text[index].isspace():
            index += 1
        yield index


def _trim_at_prefix_boundary(
    text: str,
    max_chars: int,
    boundary_positions: Iterable[int],
) -> Optional[str]:
    for end_index in boundary_positions:
        remainder = text[end_index:].lstrip()
        if not remainder:
            continue
        if len(remainder) <= max_chars:
            return remainder.rstrip()
    return None


def trim_text_prefix_to_limit(text: str, max_chars: Optional[int]) -> str:
    """Keep the newest text while dropping old prefix at natural boundaries."""
    if max_chars is None:
        return text
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    punctuation_trimmed = _trim_at_prefix_boundary(
        text,
        max_chars,
        _iter_marker_end_positions(text, TEXT_PREFIX_DROP_BOUNDARIES),
    )
    if punctuation_trimmed is not None:
        return punctuation_trimmed

    whitespace_trimmed = _trim_at_prefix_boundary(
        text,
        max_chars,
        _iter_whitespace_end_positions(text),
    )
    if whitespace_trimmed is not None:
        return whitespace_trimmed

    return text[-max_chars:].lstrip().rstrip()
