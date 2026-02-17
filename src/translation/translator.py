"""Translation module using DeepL Free API.

Translates English text to Japanese with caching and retry logic.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

# Maximum retries for API calls
MAX_RETRIES = 3
# Base delay for exponential backoff (seconds)
BASE_RETRY_DELAY = 1.0


class Translator:
    """DeepL API based English-to-Japanese translator.

    Features:
    - LRU cache to avoid redundant API calls
    - Exponential backoff retry on failures
    - Graceful degradation when API is unavailable

    Usage:
        translator = Translator(api_key="your-deepl-api-key")
        japanese = translator.translate("Hello, world!")
    """

    def __init__(self, api_key: str = "") -> None:
        """Initialize the translator.

        Args:
            api_key: DeepL API key. If empty, translation will be unavailable.
        """
        self._api_key = api_key
        self._translator = None
        self._cache: dict[str, str] = {}
        self._cache_max_size = 1000

    def _init_client(self) -> bool:
        """Initialize DeepL client if not already done.

        Returns:
            True if client is ready, False if unavailable.
        """
        if self._translator is not None:
            return True

        if not self._api_key:
            logger.warning("DeepL API key not set, translation unavailable")
            return False

        try:
            import deepl

            self._translator = deepl.Translator(self._api_key)
            # Verify the key with a usage check
            usage = self._translator.get_usage()
            if usage.character:
                remaining = usage.character.limit - usage.character.count
                logger.info(
                    "DeepL API connected. Usage: %d/%d characters (remaining: %d)",
                    usage.character.count,
                    usage.character.limit,
                    remaining,
                )
            return True
        except ImportError:
            logger.error("deepl package not installed. Run: pip install deepl")
            return False
        except Exception:
            logger.exception("Failed to initialize DeepL translator")
            self._translator = None
            return False

    def translate(self, text: str) -> str | None:
        """Translate English text to Japanese.

        Args:
            text: English text to translate.

        Returns:
            Japanese translation, or None if translation failed/unavailable.
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Check cache
        if text in self._cache:
            logger.debug("Cache hit for: %s", text[:50])
            return self._cache[text]

        if not self._init_client():
            return None

        # Retry with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                result = self._translator.translate_text(
                    text,
                    source_lang="EN",
                    target_lang="JA",
                )
                translated = result.text

                # Cache the result
                if len(self._cache) >= self._cache_max_size:
                    # Remove oldest entries (simple strategy: clear half)
                    keys = list(self._cache.keys())
                    for key in keys[: len(keys) // 2]:
                        del self._cache[key]

                self._cache[text] = translated

                logger.debug(
                    "Translated: '%s' -> '%s'",
                    text[:50],
                    translated[:50],
                )
                return translated

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    logger.warning(
                        "Translation attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Translation failed after %d attempts: %s",
                        MAX_RETRIES,
                        e,
                    )
                    return None

    def set_api_key(self, api_key: str) -> None:
        """Update the API key and reset the client.

        Args:
            api_key: New DeepL API key.
        """
        self._api_key = api_key
        self._translator = None
        self._cache.clear()

    @property
    def is_available(self) -> bool:
        """Whether translation is available (API key is set)."""
        return bool(self._api_key)

    @property
    def cache_size(self) -> int:
        """Number of cached translations."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self._cache.clear()
