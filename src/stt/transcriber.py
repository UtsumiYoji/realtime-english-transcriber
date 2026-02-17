"""Speech-to-Text transcription module using faster-whisper.

Converts audio segments to text using the faster-whisper engine
with CTranslate2 backend for efficient CPU inference.
Supports language auto-detection to filter non-target languages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscribeResult:
    """Result of a transcription including language info."""

    text: str
    language: str  # ISO 639-1 code (e.g. "en", "ja")
    language_probability: float  # 0.0 - 1.0


class Transcriber:
    """faster-whisper based speech-to-text transcriber.

    Uses CTranslate2 backend for efficient CPU inference with INT8 quantization.
    Supports multilingual models with automatic language detection.

    Usage:
        transcriber = Transcriber(model_size="base", compute_type="int8")
        result = transcriber.transcribe(audio_array)
        if result.language == "en":
            print(result.text)
    """

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        target_language: str = "en",
        language_threshold: float = 0.5,
    ) -> None:
        """Initialize the transcriber.

        The model is lazy-loaded on first transcribe() call.
        First load may download the model from Hugging Face.

        Args:
            model_size: Whisper model size. Options: tiny, base, small, medium,
                        tiny.en, base.en, small.en (English-only).
            compute_type: Quantization type. 'int8' recommended for CPU.
            target_language: Target language code. Used only for logging/info.
            language_threshold: Min language probability to accept detection.
        """
        self._model_size = model_size
        self._compute_type = compute_type
        self._target_language = target_language
        self._language_threshold = language_threshold
        self._is_english_only = model_size.endswith(".en")
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model: %s (compute_type=%s)...",
            self._model_size,
            self._compute_type,
        )
        start = time.time()

        self._model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type=self._compute_type,
        )

        elapsed = time.time() - start
        logger.info("Whisper model loaded in %.1fs", elapsed)

    def transcribe(self, audio: np.ndarray) -> TranscribeResult:
        """Transcribe audio data to text with language detection.

        Args:
            audio: Audio data as float32 numpy array, 16kHz, mono.

        Returns:
            TranscribeResult with text, detected language, and probability.
            Empty result if no speech detected.
        """
        self._load_model()

        if len(audio) == 0:
            return TranscribeResult(text="", language="", language_probability=0.0)

        start = time.time()

        try:
            # For English-only models, force language="en"
            # For multilingual models, use language=None for auto-detection
            lang_param = "en" if self._is_english_only else None

            segments, info = self._model.transcribe(
                audio,
                language=lang_param,
                beam_size=5,
                best_of=5,
                vad_filter=False,  # We handle VAD externally
                without_timestamps=True,
            )

            # Collect all segment texts
            texts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    texts.append(text)

            result = " ".join(texts)
            elapsed = time.time() - start
            audio_duration = len(audio) / 16000

            logger.debug(
                "Transcribed %.1fs audio in %.1fs (RTF=%.2f) [lang=%s prob=%.2f]: %s",
                audio_duration,
                elapsed,
                elapsed / audio_duration if audio_duration > 0 else 0,
                info.language,
                info.language_probability,
                result[:80] + "..." if len(result) > 80 else result,
            )

            return TranscribeResult(
                text=result,
                language=info.language,
                language_probability=info.language_probability,
            )

        except Exception:
            logger.exception("Transcription failed")
            return TranscribeResult(text="", language="", language_probability=0.0)

    @property
    def model_size(self) -> str:
        """Current model size."""
        return self._model_size

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._model is not None
