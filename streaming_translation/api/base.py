from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class BaseTranslationAPI(ABC):
    """Abstract base class for all translation API backends.

    Subclasses must implement :meth:`translate` and set :attr:`SUPPORTS_CONTEXT`.
    """

    SUPPORTS_CONTEXT: bool = False

    @abstractmethod
    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "zh-CN",
        context: Optional[str] = None,
        context_pairs: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> str:
        """Translate *text* from *source_language* to *target_language*.

        Args:
            text: Text to translate.
            source_language: Source language code (``"auto"`` = detect).
            target_language: Target language code.
            context: Optional plain-text context (source only).  Raises
                ``NotImplementedError`` if the backend does not support it.
            context_pairs: Optional list of ``{"source": …, "target": …}``
                dicts for translation-memory-like context.
            **kwargs: Additional backend-specific options (e.g. ``is_partial``).

        Returns:
            Translated text, or a string starting with ``"[ERROR]"`` on failure.
        """
        ...
