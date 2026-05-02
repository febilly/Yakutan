# Translation Terminology Memory

This folder contains CSV files used for keyword-based translation terminology memory.

## How it works

When translating text, the system scans the user's input for keywords defined in these CSV files. If any keyword matches, the corresponding instruction is injected into the translation context as a block like this:

```
Terminology hints (only apply these when the original text clearly matches
the term's intended meaning — do not force-fit if the context differs):
  'world': Use 'ワールド' for a VRChat world, and '世界' for the general term 'world'.
  'avatar': Use 'アバター' when referring to a VRChat avatar.
```

The system matches by simple substring — if the keyword appears anywhere in the user's text, the hint is included. No fuzzy matching. No AI processing.

## File format

Each CSV file has two columns:

| Column | Description |
|---|---|
| `keywords` | One or more aliases separated by `;`. The hint triggers if **any** of these appear in the user's text. |
| `instruction` | A full sentence in English describing how the translator should handle the term. |

- The first row is treated as a header and skipped.
- Rows where the first column starts with `#` are treated as comments and skipped.
- Empty rows are ignored.

Example (`ja.csv`):

```csv
keywords,instruction
世界; world,"Use 'ワールド' for a VRChat world, and '世界' for the general term 'world'."
# 下面是 avatar 相关术语
模型; avatar,"Use 'アバター' when referring to a VRChat avatar."
```

## File naming

Files are named by target language code:

- `ja.csv` — Used when translating to Japanese
- `en.csv` — Used when translating to English
- `zh-cn.csv` — Used when translating to Simplified Chinese
- etc.

## Private overrides

You can also create files in `terminology_private/` (next to the executable or project root). These are loaded in addition to the public files. Files in `terminology_private/` are not tracked by Git.

## Configuration

Enable or disable terminology memory in `config.py`:

```python
TERMINOLOGY_ENABLED = True
```
