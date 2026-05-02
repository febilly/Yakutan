# Translation Terminology Memory

This folder contains CSV files used for keyword-based translation terminology memory.

## How it works

When translating text, the system scans the user's input for keywords defined in these CSV files. If any keyword matches, the corresponding instruction is injected into the translation context, guiding the translator to handle specific terms correctly.

## File format

Each CSV file has two columns:

- **keywords**: One or more keywords separated by `|`. If any of these keywords appear in the user's text, the instruction is triggered.
- **instruction**: A full sentence in English explaining how the translator should handle the term. This is injected into the translation context.

Example (`ja.csv`):

```csv
keywords,instruction
世界|ワールド,"If the user mentions 'world' in the context of VRChat, translate it as 'ワールド', not '世界'."
アバター|avatar,"When referring to a VRChat character model, use 'アバター'."
```

## File naming

Files are named by target language code:

- `ja.csv` — Used when translating to Japanese
- `en.csv` — Used when translating to English
- `zh-cn.csv` — Used when translating to Simplified Chinese
- etc.

## Private overrides

You can also create files in `terminology_private/` (next to the executable or project root). These are loaded in addition to the public files and take precedence for matching. Files in `terminology_private/` are not tracked by Git.

## Configuration

Enable or disable terminology memory in `config.py`:

```python
TERMINOLOGY_ENABLED = True  # Set to False to disable
```
