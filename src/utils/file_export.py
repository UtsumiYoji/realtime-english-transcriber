"""File export module for saving transcripts.

Supports manual save (full transcript) and auto-save (append mode).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    """A single transcript entry with timestamp and text."""

    timestamp: datetime
    text: str
    translation: str | None = None
    source: str = "output"  # "output" or "mic"
    language: str = "en"    # detected language code (e.g. "en", "ja")

    def format_text(self, include_translation: bool = True) -> str:
        """Format entry as a display/save string."""
        ts = self.timestamp.strftime("%H:%M:%S")
        source_icon = "\U0001f50a" if self.source == "output" else "\U0001f3a4"
        lang_label = self.language.upper()
        lines = [f"[{ts}] {source_icon} {lang_label}: {self.text}"]
        if include_translation and self.translation:
            lines.append(f"[{ts}] {source_icon} JA: {self.translation}")
        return "\n".join(lines)


class FileExporter:
    """Exports transcript entries to text files.

    Supports two modes:
    - Manual save: save all entries at once via save_transcript()
    - Auto-save: append entries in real-time via start_auto_save() + append_entry()
    """

    def __init__(self) -> None:
        self._auto_save_path: Path | None = None
        self._auto_save_file = None

    def save_transcript(
        self,
        entries: list[TranscriptEntry],
        filepath: str | Path,
        include_translation: bool = True,
    ) -> None:
        """Save all transcript entries to a file.

        Args:
            entries: List of transcript entries to save.
            filepath: Output file path.
            include_translation: Whether to include translations.
        """
        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Entries: {len(entries)}\n\n")
                for entry in entries:
                    f.write(entry.format_text(include_translation) + "\n\n")
            logger.info("Transcript saved to %s (%d entries)", filepath, len(entries))
        except Exception:
            logger.exception("Failed to save transcript to %s", filepath)
            raise

    def start_auto_save(self, filepath: str | Path) -> None:
        """Start auto-save mode. New entries will be appended to this file.

        Args:
            filepath: Output file path for auto-save.
        """
        self.stop_auto_save()  # Close any existing file

        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self._auto_save_file = open(filepath, "a", encoding="utf-8")
            self._auto_save_path = filepath

            # Write session header
            self._auto_save_file.write(
                f"\n# --- Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n"
            )
            self._auto_save_file.flush()
            logger.info("Auto-save started: %s", filepath)
        except Exception:
            logger.exception("Failed to start auto-save to %s", filepath)
            self._auto_save_file = None
            self._auto_save_path = None
            raise

    def append_entry(self, entry: TranscriptEntry, include_translation: bool = True) -> None:
        """Append a single entry to the auto-save file.

        Args:
            entry: Transcript entry to append.
            include_translation: Whether to include translation.
        """
        if self._auto_save_file is None:
            return

        try:
            self._auto_save_file.write(entry.format_text(include_translation) + "\n\n")
            self._auto_save_file.flush()
        except Exception:
            logger.exception("Failed to append entry to auto-save file")

    def stop_auto_save(self) -> None:
        """Stop auto-save mode and close the file."""
        if self._auto_save_file is not None:
            try:
                self._auto_save_file.write(
                    f"# --- Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )
                self._auto_save_file.close()
                logger.info("Auto-save stopped: %s", self._auto_save_path)
            except Exception:
                logger.exception("Error closing auto-save file")
            finally:
                self._auto_save_file = None
                self._auto_save_path = None

    @property
    def is_auto_saving(self) -> bool:
        """Whether auto-save mode is active."""
        return self._auto_save_file is not None

    @property
    def auto_save_filepath(self) -> Path | None:
        """Current auto-save file path, or None if not active."""
        return self._auto_save_path
