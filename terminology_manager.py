import csv
import logging
import os
from typing import List, Dict, Optional, Tuple

from resource_path import get_resource_path, get_user_data_path, ensure_dir

logger = logging.getLogger(__name__)


class TerminologyEntry:
    def __init__(self, keywords: List[str], instruction: str):
        self.keywords = keywords
        self.instruction = instruction


class TerminologyManager:
    TERMINOLOGY_DIR = "terminology"
    TERMINOLOGY_PRIVATE_DIR = "terminology_private"

    def __init__(self):
        self._entries: Dict[str, List[TerminologyEntry]] = {}
        self._loaded: set = set()

    def _load_csv(self, file_path: str) -> List[TerminologyEntry]:
        entries = []
        try:
            with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    if row[0].strip().startswith("#"):
                        continue
                    keywords_raw = row[0].strip()
                    instruction = row[1].strip()
                    if not keywords_raw or not instruction:
                        continue
                    keywords = [k.strip() for k in keywords_raw.split(";") if k.strip()]
                    if keywords:
                        entries.append(TerminologyEntry(keywords, instruction))
            logger.info(f"从 {file_path} 加载了 {len(entries)} 条术语记忆")
        except FileNotFoundError:
            logger.debug(f"术语记忆文件不存在: {file_path}")
        except Exception as e:
            logger.warning(f"加载术语记忆文件失败 {file_path}: {e}")
        return entries

    def load_for_language(self, target_language: str) -> List[TerminologyEntry]:
        if target_language in self._loaded:
            return self._entries.get(target_language, [])

        base_dir = get_resource_path(self.TERMINOLOGY_DIR)
        private_dir = get_user_data_path(self.TERMINOLOGY_PRIVATE_DIR)
        ensure_dir(private_dir)

        all_entries: List[TerminologyEntry] = []
        public_path = os.path.join(base_dir, f"{target_language}.csv")
        all_entries.extend(self._load_csv(public_path))
        private_path = os.path.join(private_dir, f"{target_language}.csv")
        all_entries.extend(self._load_csv(private_path))

        self._entries[target_language] = all_entries
        self._loaded.add(target_language)
        logger.info(
            f"目标语言 '{target_language}' 共加载 {len(all_entries)} 条术语记忆"
        )
        return all_entries

    def find_matches(self, text: str, target_language: str) -> List[tuple[str, str]]:
        if not text or not text.strip():
            return []

        entries = self.load_for_language(target_language)
        if not entries:
            return []

        matched: List[tuple[str, str]] = []
        seen = set()
        text_lower = text.lower()

        for entry in entries:
            for keyword in entry.keywords:
                if keyword.lower() in text_lower:
                    if entry.instruction not in seen:
                        matched.append((keyword, entry.instruction))
                        seen.add(entry.instruction)
                    break

        return matched

    def get_terminology_hints(
        self, text: str, target_language: str
    ) -> Optional[str]:
        matched = self.find_matches(text, target_language)
        if not matched:
            return None

        lines = ["Terminology hints:"]
        for keyword, inst in matched:
            lines.append(f"  '{keyword}': {inst}")
        return "\n".join(lines)

    def reload(self) -> None:
        self._entries.clear()
        self._loaded.clear()
        logger.info("术语记忆缓存已清空，下次翻译时将重新加载")

    def list_loaded_languages(self) -> List[str]:
        return list(self._loaded)


_terminology_manager: Optional[TerminologyManager] = None


def get_terminology_manager() -> TerminologyManager:
    global _terminology_manager
    if _terminology_manager is None:
        _terminology_manager = TerminologyManager()
    return _terminology_manager


def set_terminology_manager(manager: TerminologyManager) -> None:
    global _terminology_manager
    _terminology_manager = manager
