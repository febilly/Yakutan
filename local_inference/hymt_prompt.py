"""Hy-MT2 StreamRevise v4 prompt shared by desktop and inference servers.

Keep the wording aligned with the LoRA training template (realtime-subtitle
d582ee8). Only translated history pairs were present in that training data.
"""

from __future__ import annotations

TRAINED_HISTORY_PAIRS = 5
STABILITY_RULE = (
    "When the source meaning has not changed, preserve the still-correct prefix "
    "of the previous translation whenever possible. When content is added or "
    "corrected, accuracy and completeness take priority."
)


def source_was_cut(previous_source: str | None, source: str) -> bool:
    """A substantially shorter revision must not preserve the removed content."""
    previous = (previous_source or "").strip()
    return bool(previous) and len(source.strip()) < 0.8 * len(previous)


def trained_history(items, limit: int = TRAINED_HISTORY_PAIRS) -> list[list[str]]:
    limit = max(0, min(int(limit), TRAINED_HISTORY_PAIRS))
    if not limit or not isinstance(items, (list, tuple)):
        return []
    pairs = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            source, target = item[:2]
        elif isinstance(item, dict):
            source = item.get("source")
            target = item.get("target") or item.get("translation")
        else:
            continue
        source, target = str(source or "").strip(), str(target or "").strip()
        if source and target:
            pairs.append([source, target])
    return pairs[-limit:]


def render_trained_prompt(
    *, source_name: str, target_name: str, history: list[list[str]],
    previous_source: str, previous_translation: str, current_source: str,
) -> str:
    if source_was_cut(previous_source, current_source):
        previous_source = previous_translation = ""
    if not history and not previous_source and not previous_translation:
        return (
            f"Translate the following text from {source_name} into {target_name}. Note that you "
            "should only output the translated result without any additional "
            f"explanation:\n\n{current_source}"
        )
    background = []
    if history:
        background.append(
            "Recent utterances and their translations:\n"
            + "\n".join(f"{source} → {target}" for source, target in history)
        )
    if previous_source:
        background.append("Previous version of the current source:\n" + previous_source)
    if previous_translation:
        background.append("Previous translation of the current source:\n" + previous_translation)
    background.append(STABILITY_RULE)
    return (
        "[Background Information]\n" + "\n\n".join(background)
        + f"\n\nPlease translate the following text from {source_name} into {target_name}, "
        "taking the provided background information into consideration.\n\n[Source Text]\n"
        + current_source
    )
