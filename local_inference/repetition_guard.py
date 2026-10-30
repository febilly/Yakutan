"""Conservative detector for runaway repetition in Hy-MT2 output ("run, run, run, ..." until max tokens).

Text is compared as units: each CJK/kana/hangul character is one unit, each run of letters/digits is
one word unit, and punctuation and whitespace are dropped (NFKC + casefold), so "Run, run, run" is the
unit "run" three times. A block of ``p`` units repeated ``k`` times in a row is found by brute force
over ``p``: positions where ``units[i] == units[i + p]`` form runs, and a run of ``L`` such positions
means the block occurs ``L // p + 1`` times.

The thresholds are high on purpose: a false positive costs a second decode and may change a correct
translation, while a real loop runs to dozens of repeats.
"""
from __future__ import annotations

import unicodedata
from typing import Hashable, Sequence

# A translation counts as a runaway only when its tail is one block repeated at least this many times.
RUNAWAY_REPEATS = 10
# A source that itself repeats a block this many times (anywhere) may legitimately translate to a long
# repetition, so its translation is never flagged.
SOURCE_REPEATS = 5
# Penalty for the single retry after a runaway; normal decoding uses none.
RETRY_REPEAT_PENALTY = 1.15


def max_translation_tokens(source: str) -> int:
    """Output cap for one translation. Chinese to English takes about 0.65 tokens per source character,
    so 2 per character leaves any real sentence room; a loop is ended by the guard, not by this cap."""
    return max(160, 2 * len(source))


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF          # hiragana, katakana
        or 0x31F0 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x1100 <= code <= 0x11FF       # hangul jamo
        or 0x3130 <= code <= 0x318F
        or 0xAC00 <= code <= 0xD7AF
        or 0x20000 <= code <= 0x3FFFF
    )


def text_units(text: str) -> list[str]:
    units: list[str] = []
    word: list[str] = []
    for ch in unicodedata.normalize("NFKC", text or "").casefold():
        if _is_cjk(ch):
            if word:
                units.append("".join(word))
                word = []
            units.append(ch)
        elif ch.isalnum():
            word.append(ch)
        elif word:
            units.append("".join(word))
            word = []
    if word:
        units.append("".join(word))
    return units


def tail_repeats(units: Sequence[Hashable], min_count: int = RUNAWAY_REPEATS) -> int:
    """Largest ``k >= min_count`` such that the tail is one block repeated ``k`` times, else 0."""
    n = len(units)
    for p in range(1, n // min_count + 1):
        matched = 0
        i = n - p - 1
        while i >= 0 and units[i] == units[i + p]:
            matched += 1
            i -= 1
        count = matched // p + 1
        if count >= min_count:
            return count
    return 0


def has_repeated_block(units: Sequence[Hashable], min_count: int = SOURCE_REPEATS) -> bool:
    """True when some block occurs ``min_count`` times in a row anywhere in ``units``."""
    n = len(units)
    for p in range(1, n // min_count + 1):
        need = (min_count - 1) * p
        run = 0
        for i in range(n - p):
            run = run + 1 if units[i] == units[i + p] else 0
            if run >= need:
                return True
    return False


def source_is_repetitive(source: str) -> bool:
    return has_repeated_block(text_units(source), SOURCE_REPEATS)


def is_runaway(translation: str, source: str = "") -> bool:
    """A translation whose tail loops, unless the source itself repeats."""
    if not tail_repeats(text_units(translation), RUNAWAY_REPEATS):
        return False
    return not source_is_repetitive(source)
