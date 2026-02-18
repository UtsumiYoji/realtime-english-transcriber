"""Audio capture module using WASAPI Loopback.

Captures system audio (or specific virtual device audio) and delivers
resampled 16kHz mono float32 chunks via a queue for downstream processing.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Any

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

# Target format for Whisper / VAD
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


@dataclass
class AudioDevice:
    """Represents an available audio device."""

    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_loopback: bool = False

    def __str__(self) -> str:
        lb = " [Loopback]" if self.is_loopback else ""
        return f"{self.name}{lb}"


class AudioCapture:
    """Captures audio from WASAPI loopback or input devices.

    Uses PyAudioWPatch as primary (reliable WASAPI loopback support),
    with sounddevice as fallback.

    Each instance carries a source_tag that is prepended to queue entries
    as (source_tag, chunk) tuples, allowing downstream consumers to
    distinguish between multiple audio sources.

    Usage:
        capture = AudioCapture(source_tag="output")
        devices = capture.get_devices()
        capture.start(device_index, audio_queue)
        ...  # audio_queue receives ("output", np.ndarray) tuples
        capture.stop()
    """

    def __init__(self, source_tag: str = "output") -> None:
        self._stream: Any = None
        self._running = False
        self._lock = threading.Lock()
        self._backend: str = ""
        self.source_tag = source_tag

    def get_devices(self) -> list[AudioDevice]:
        """Get list of available audio devices with loopback support.

        Tries PyAudioWPatch first (better WASAPI loopback enumeration),
        then falls back to sounddevice.

        Returns:
            List of AudioDevice objects.
        """
        devices = self._get_devices_pyaudiowpatch()
        if devices:
            return devices

        logger.info("PyAudioWPatch unavailable, falling back to sounddevice")
        return self._get_devices_sounddevice()

    def get_output_devices(self) -> list[AudioDevice]:
        """Get list of output (loopback) audio devices.

        Returns devices that can capture system audio output.
        With PyAudioWPatch, these are flagged as loopback devices.
        With sounddevice, output devices (max_output_channels > 0) are returned.
        """
        return [
            d for d in self.get_devices()
            if d.is_loopback or (d.max_output_channels > 0 and d.max_input_channels == 0)
        ]

    def get_input_devices(self) -> list[AudioDevice]:
        """Get list of input (microphone) audio devices.

        Returns devices that can capture microphone / line-in audio.
        """
        return [
            d for d in self.get_devices()
            if d.max_input_channels > 0 and not d.is_loopback
        ]

    def _get_devices_pyaudiowpatch(self) -> list[AudioDevice]:
        """Enumerate devices using PyAudioWPatch (WASAPI loopback support)."""
        try:
            import pyaudiowpatch as pyaudio

            p = pyaudio.PyAudio()
            devices = []

            try:
                # Find WASAPI host API
                wasapi_info = None
                for i in range(p.get_host_api_count()):
                    api_info = p.get_host_api_info_by_index(i)
                    if "wasapi" in api_info.get("name", "").lower():
                        wasapi_info = api_info
                        break

                if wasapi_info is None:
                    logger.warning("WASAPI host API not found in PyAudioWPatch")
                    return []

                # Enumerate all devices under WASAPI
                for i in range(p.get_device_count()):
                    try:
                        info = p.get_device_info_by_index(i)
                        host_api_idx = info.get("hostApi", -1)
                        host_api_info = p.get_host_api_info_by_index(host_api_idx)

                        if "wasapi" not in host_api_info.get("name", "").lower():
                            continue

                        is_loopback = info.get("isLoopbackDevice", False)
                        name = info.get("name", f"Device {i}")

                        device = AudioDevice(
                            index=i,
                            name=name,
                            host_api="WASAPI",
                            max_input_channels=info.get("maxInputChannels", 0),
                            max_output_channels=info.get("maxOutputChannels", 0),
                            default_sample_rate=info.get("defaultSampleRate", 44100),
                            is_loopback=is_loopback,
                        )

                        # Include loopback devices and input devices
                        if is_loopback or device.max_input_channels > 0:
                            devices.append(device)
                    except Exception:
                        continue

            finally:
                p.terminate()

            self._backend = "pyaudiowpatch"
            logger.info("Found %d WASAPI devices via PyAudioWPatch", len(devices))
            return devices

        except ImportError:
            logger.debug("PyAudioWPatch not installed")
            return []
        except Exception:
            logger.exception("Error enumerating PyAudioWPatch devices")
            return []

    def _get_devices_sounddevice(self) -> list[AudioDevice]:
        """Enumerate devices using sounddevice."""
        try:
            import sounddevice as sd

            devices = []
            raw_devices = sd.query_devices()
            host_apis = sd.query_hostapis()

            for i, info in enumerate(raw_devices):
                api_name = host_apis[info["hostapi"]]["name"]

                # Prefer WASAPI devices
                if "wasapi" not in api_name.lower():
                    continue

                device = AudioDevice(
                    index=i,
                    name=info["name"],
                    host_api=api_name,
                    max_input_channels=info["max_input_channels"],
                    max_output_channels=info["max_output_channels"],
                    default_sample_rate=info["default_samplerate"],
                    is_loopback=False,  # sounddevice doesn't flag loopback
                )

                if device.max_input_channels > 0 or device.max_output_channels > 0:
                    devices.append(device)

            self._backend = "sounddevice"
            logger.info("Found %d WASAPI devices via sounddevice", len(devices))
            return devices

        except ImportError:
            logger.error("Neither PyAudioWPatch nor sounddevice is installed")
            return []
        except Exception:
            logger.exception("Error enumerating sounddevice devices")
            return []

    def start(self, device: AudioDevice, audio_queue: Queue) -> None:
        """Start capturing audio from the specified device.

        Captured audio is resampled to 16kHz mono float32 and pushed
        to the provided queue as (source_tag, np.ndarray) tuples.

        Args:
            device: AudioDevice to capture from.
            audio_queue: Queue to receive (source_tag, chunk) tuples.
        """
        with self._lock:
            if self._running:
                logger.warning("Audio capture already running, stopping first")
                self._stop_internal()

            self._running = True

        if self._backend == "pyaudiowpatch" or device.is_loopback:
            self._start_pyaudiowpatch(device, audio_queue)
        else:
            self._start_sounddevice(device, audio_queue)

    def _start_pyaudiowpatch(self, device: AudioDevice, audio_queue: Queue) -> None:
        """Start capture using PyAudioWPatch."""
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()

        device_info = p.get_device_info_by_index(device.index)
        logger.info("Device info: %s", device_info)

        # For loopback devices, maxInputChannels may be 0;
        # use maxOutputChannels from the corresponding output device instead.
        input_ch = int(device_info.get("maxInputChannels", 0))
        output_ch = int(device_info.get("maxOutputChannels", 0))
        channels = max(input_ch, output_ch, 1)
        if device.is_loopback and input_ch == 0:
            channels = max(output_ch, 2)

        sample_rate = int(device_info.get("defaultSampleRate", 44100))

        logger.info(
            "Opening stream: device=%d, channels=%d, rate=%d, loopback=%s",
            device.index, channels, sample_rate, device.is_loopback,
        )

        # Calculate resampling ratio
        # We'll use a chunk size that gives ~100ms of audio
        frames_per_buffer = int(sample_rate * 0.1)

        def callback(in_data, frame_count, time_info, status):
            if not self._running:
                return (None, pyaudio.paComplete)

            try:
                # Convert bytes to numpy float32
                audio_data = np.frombuffer(in_data, dtype=np.float32)

                # Reshape to (samples, channels) and convert to mono
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)
                    audio_data = audio_data.mean(axis=1)

                # Resample to target rate
                if sample_rate != TARGET_SAMPLE_RATE:
                    # Use rational resampling
                    from math import gcd

                    g = gcd(TARGET_SAMPLE_RATE, sample_rate)
                    up = TARGET_SAMPLE_RATE // g
                    down = sample_rate // g
                    audio_data = resample_poly(audio_data, up, down).astype(np.float32)

                audio_queue.put((self.source_tag, audio_data))
            except Exception:
                logger.exception("Error in audio callback")

            return (None, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=frames_per_buffer,
            stream_callback=callback,
        )
        stream.start_stream()
        self._stream = (p, stream)
        logger.info(
            "Audio capture started: %s (%dHz, %dch)", device.name, sample_rate, channels
        )

    def _start_sounddevice(self, device: AudioDevice, audio_queue: Queue) -> None:
        """Start capture using sounddevice."""
        import sounddevice as sd

        device_info = sd.query_devices(device.index)
        channels = max(device_info["max_input_channels"], 1)
        sample_rate = int(device_info["default_samplerate"])

        def callback(indata, frames, time_info, status):
            if status:
                logger.debug("sounddevice status: %s", status)
            if not self._running:
                return

            try:
                audio_data = indata.copy()

                # Convert to mono
                if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                    audio_data = audio_data.mean(axis=1)
                elif audio_data.ndim > 1:
                    audio_data = audio_data.squeeze()

                # Resample to target rate
                if sample_rate != TARGET_SAMPLE_RATE:
                    from math import gcd

                    g = gcd(TARGET_SAMPLE_RATE, sample_rate)
                    up = TARGET_SAMPLE_RATE // g
                    down = sample_rate // g
                    audio_data = resample_poly(audio_data, up, down).astype(np.float32)

                audio_queue.put((self.source_tag, audio_data))
            except Exception:
                logger.exception("Error in audio callback")

        blocksize = int(sample_rate * 0.1)  # ~100ms chunks
        stream = sd.InputStream(
            device=device.index,
            channels=channels,
            samplerate=sample_rate,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        )
        stream.start()
        self._stream = stream
        logger.info(
            "Audio capture started (sounddevice): %s (%dHz, %dch)",
            device.name,
            sample_rate,
            channels,
        )

    def stop(self) -> None:
        """Stop audio capture."""
        with self._lock:
            self._stop_internal()

    def _stop_internal(self) -> None:
        """Internal stop method (must be called with lock held)."""
        self._running = False

        if self._stream is None:
            return

        try:
            if isinstance(self._stream, tuple):
                # PyAudioWPatch: (PyAudio, Stream)
                p, stream = self._stream
                stream.stop_stream()
                stream.close()
                p.terminate()
            else:
                # sounddevice stream
                self._stream.stop()
                self._stream.close()
        except Exception:
            logger.exception("Error stopping audio stream")
        finally:
            self._stream = None

        logger.info("Audio capture stopped")

    @property
    def is_running(self) -> bool:
        """Whether audio capture is currently active."""
        return self._running
