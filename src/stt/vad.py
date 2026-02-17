"""Voice Activity Detection (VAD) module using Silero VAD.

Detects speech segments in an audio stream and outputs complete
utterances for downstream STT processing.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD expects 16kHz, mono, float32
VAD_SAMPLE_RATE = 16000

# Silero VAD frame sizes (in samples at 16kHz)
# Supported: 512 (32ms), 1024 (64ms), 1536 (96ms)
VAD_FRAME_SAMPLES = 512
VAD_FRAME_DURATION_MS = VAD_FRAME_SAMPLES / VAD_SAMPLE_RATE * 1000  # 32ms


class VoiceActivityDetector:
    """Silero VAD wrapper for detecting speech segments.

    Processes incoming audio chunks and outputs complete speech segments
    (from speech start to speech end) as numpy arrays.

    Usage:
        vad = VoiceActivityDetector(threshold=0.5)
        for chunk in audio_stream:
            utterances = vad.process_chunk(chunk)
            for utterance in utterances:
                text = transcriber.transcribe(utterance)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        max_speech_s: float = 30.0,
        min_silence_ms: int = 500,
        periodic_flush_s: float = 5.0,
    ) -> None:
        """Initialize the VAD.

        Args:
            threshold: Speech probability threshold (0.0-1.0).
                Higher = more strict (fewer false positives, may miss quiet speech).
            min_speech_ms: Minimum speech duration to process (milliseconds).
                Utterances shorter than this are discarded.
            max_speech_s: Maximum speech duration before forced flush (seconds).
                Prevents unbounded buffering of continuous speech.
            min_silence_ms: Minimum silence duration to end an utterance (milliseconds).
            periodic_flush_s: Flush accumulated speech every N seconds even
                if no silence is detected. Essential for continuous audio
                streams (e.g. video playback) where silence gaps are rare.
        """
        self._threshold = threshold
        self._min_speech_samples = int(VAD_SAMPLE_RATE * min_speech_ms / 1000)
        self._max_speech_samples = int(VAD_SAMPLE_RATE * max_speech_s)
        self._min_silence_frames = int(min_silence_ms / VAD_FRAME_DURATION_MS)
        self._periodic_flush_samples = int(VAD_SAMPLE_RATE * periodic_flush_s)

        # Internal state
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_frame_count = 0
        self._total_speech_samples = 0

        # Residual audio from incomplete frames
        self._residual = np.array([], dtype=np.float32)

        # Silero VAD model (lazy loaded)
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the Silero VAD model."""
        if self._model is not None:
            return

        import torch

        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model = model
        logger.info("Silero VAD model loaded")

    def process_chunk(self, audio_chunk: np.ndarray) -> list[np.ndarray]:
        """Process an audio chunk and return completed utterances.

        Args:
            audio_chunk: Audio data (float32, 16kHz, mono).

        Returns:
            List of complete utterance audio arrays. Each is a float32
            numpy array at 16kHz. May be empty if no utterance is complete.
        """
        self._load_model()

        import torch

        # Prepend any residual audio from last call
        if len(self._residual) > 0:
            audio_chunk = np.concatenate([self._residual, audio_chunk])
            self._residual = np.array([], dtype=np.float32)

        completed_utterances: list[np.ndarray] = []

        # Process in VAD_FRAME_SAMPLES-sized frames
        offset = 0
        while offset + VAD_FRAME_SAMPLES <= len(audio_chunk):
            frame = audio_chunk[offset : offset + VAD_FRAME_SAMPLES]
            offset += VAD_FRAME_SAMPLES

            # Get speech probability
            frame_tensor = torch.from_numpy(frame)
            speech_prob = self._model(frame_tensor, VAD_SAMPLE_RATE).item()

            is_speech = speech_prob >= self._threshold

            if is_speech:
                if not self._is_speaking:
                    # Speech started
                    self._is_speaking = True
                    self._silence_frame_count = 0
                    self._total_speech_samples = 0
                    logger.debug("Speech started (prob=%.2f)", speech_prob)

                self._speech_buffer.append(frame.copy())
                self._total_speech_samples += VAD_FRAME_SAMPLES
                self._silence_frame_count = 0

                # Periodic flush for continuous speech (e.g. video audio)
                if self._total_speech_samples >= self._periodic_flush_samples:
                    utterance = self._flush_buffer()
                    if utterance is not None:
                        completed_utterances.append(utterance)
                        logger.info(
                            "Periodic flush: %.1fs of speech",
                            len(utterance) / VAD_SAMPLE_RATE,
                        )
                    # Reset sample counter but stay in speaking state
                    self._total_speech_samples = 0
            else:
                if self._is_speaking:
                    # Still include silence frames in buffer for natural endings
                    self._speech_buffer.append(frame.copy())
                    self._silence_frame_count += 1

                    if self._silence_frame_count >= self._min_silence_frames:
                        # Speech ended
                        utterance = self._flush_buffer()
                        if utterance is not None:
                            completed_utterances.append(utterance)
                            logger.debug(
                                "Speech ended: %.1fs of audio",
                                len(utterance) / VAD_SAMPLE_RATE,
                            )
                        self._is_speaking = False
                        self._silence_frame_count = 0
                        self._total_speech_samples = 0

        # Save residual audio for next call
        if offset < len(audio_chunk):
            self._residual = audio_chunk[offset:].copy()

        return completed_utterances

    def _flush_buffer(self) -> np.ndarray | None:
        """Concatenate and return the speech buffer, then clear it.

        Returns:
            Concatenated audio if it meets minimum length, else None.
        """
        if not self._speech_buffer:
            return None

        utterance = np.concatenate(self._speech_buffer)
        self._speech_buffer.clear()

        # Discard utterances that are too short
        if len(utterance) < self._min_speech_samples:
            logger.debug(
                "Discarding short utterance: %.1fms",
                len(utterance) / VAD_SAMPLE_RATE * 1000,
            )
            return None

        return utterance

    def reset(self) -> None:
        """Reset VAD state. Call when starting a new session."""
        self._speech_buffer.clear()
        self._is_speaking = False
        self._silence_frame_count = 0
        self._total_speech_samples = 0
        self._residual = np.array([], dtype=np.float32)

        if self._model is not None:
            self._model.reset_states()

        logger.debug("VAD state reset")

    def flush_remaining(self) -> np.ndarray | None:
        """Flush any remaining speech in the buffer.

        Call this when stopping capture to avoid losing accumulated audio.

        Returns:
            Remaining audio or None if buffer is empty/too short.
        """
        if not self._speech_buffer:
            return None

        utterance = self._flush_buffer()
        if utterance is not None:
            logger.info(
                "Flushed remaining buffer: %.1fs of audio",
                len(utterance) / VAD_SAMPLE_RATE,
            )
        self._is_speaking = False
        self._silence_frame_count = 0
        self._total_speech_samples = 0
        return utterance

    @property
    def is_speaking(self) -> bool:
        """Whether VAD currently detects speech."""
        return self._is_speaking
