"""Tkinter-based GUI for Realtime English Transcriber.

Main application window with audio device selection, start/stop controls,
transcript display, and file save functionality.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from queue import Queue
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import numpy as np

from src.audio.capture import AudioCapture, AudioDevice
from src.stt.transcriber import TranscribeResult, Transcriber
from src.stt.vad import VoiceActivityDetector
from src.translation.translator import Translator
from src.utils.config import AppConfig
from src.utils.file_export import FileExporter, TranscriptEntry

logger = logging.getLogger(__name__)

# Poll interval for checking queues (ms)
POLL_INTERVAL_MS = 100

# Queue sizes
AUDIO_QUEUE_MAX = 100
TEXT_QUEUE_MAX = 50
DISPLAY_QUEUE_MAX = 100


class TranscriberApp:
    """Main application window and pipeline orchestrator.

    Manages the GUI and coordinates background threads for:
    - Audio capture
    - VAD + Speech-to-text
    - Translation
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Core components
        self._audio_capture = AudioCapture()
        self._vad = VoiceActivityDetector(
            threshold=config.vad_threshold,
            min_speech_ms=config.min_speech_ms,
            max_speech_s=config.max_speech_duration,
        )
        self._transcriber = Transcriber(
            model_size=config.whisper_model,
            compute_type=config.compute_type,
        )
        self._translator = Translator(api_key=config.deepl_api_key)
        self._file_exporter = FileExporter()

        # Queues for inter-thread communication
        self._audio_queue: Queue[np.ndarray] = Queue(maxsize=AUDIO_QUEUE_MAX)
        self._text_queue: Queue[TranscriptEntry] = Queue(maxsize=TEXT_QUEUE_MAX)
        self._display_queue: Queue[TranscriptEntry] = Queue(maxsize=DISPLAY_QUEUE_MAX)

        # Thread control
        self._stop_event = threading.Event()
        self._stt_thread: threading.Thread | None = None
        self._translation_thread: threading.Thread | None = None

        # State
        self._devices: list[AudioDevice] = []
        self._entries: list[TranscriptEntry] = []
        self._is_recording = False

        # Build GUI
        self._root = tk.Tk()
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the Tkinter UI."""
        root = self._root
        root.title("Realtime English Transcriber")
        root.geometry("800x600")
        root.minsize(600, 400)

        # --- Top control bar ---
        control_frame = ttk.Frame(root, padding=5)
        control_frame.pack(fill=tk.X)

        # Audio device selection
        ttk.Label(control_frame, text="Audio Device:").pack(side=tk.LEFT, padx=(0, 5))
        self._device_var = tk.StringVar()
        self._device_combo = ttk.Combobox(
            control_frame,
            textvariable=self._device_var,
            state="readonly",
            width=40,
        )
        self._device_combo.pack(side=tk.LEFT, padx=(0, 5))

        # Refresh devices button
        self._refresh_btn = ttk.Button(
            control_frame, text="🔄", width=3, command=self._refresh_devices
        )
        self._refresh_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Start / Stop buttons
        self._start_btn = ttk.Button(
            control_frame, text="▶ Start", command=self._start_capture
        )
        self._start_btn.pack(side=tk.LEFT, padx=2)

        self._stop_btn = ttk.Button(
            control_frame, text="■ Stop", command=self._stop_capture, state=tk.DISABLED
        )
        self._stop_btn.pack(side=tk.LEFT, padx=2)

        # --- Options bar ---
        options_frame = ttk.Frame(root, padding=5)
        options_frame.pack(fill=tk.X)

        # Translation toggle
        self._translate_var = tk.BooleanVar(value=False)
        self._translate_check = ttk.Checkbutton(
            options_frame,
            text="Translation (EN→JA)",
            variable=self._translate_var,
        )
        self._translate_check.pack(side=tk.LEFT, padx=(0, 15))

        # Auto-save toggle
        self._autosave_var = tk.BooleanVar(value=bool(self._config.auto_save_path))
        self._autosave_check = ttk.Checkbutton(
            options_frame,
            text="Auto-save",
            variable=self._autosave_var,
            command=self._toggle_autosave,
        )
        self._autosave_check.pack(side=tk.LEFT, padx=(0, 15))

        # Model info
        self._model_label = ttk.Label(
            options_frame,
            text=f"Model: {self._config.whisper_model}",
            foreground="gray",
        )
        self._model_label.pack(side=tk.RIGHT)

        # --- Main text display ---
        text_frame = ttk.Frame(root, padding=5)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self._text_display = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            selectbackground="#264f78",
            padx=10,
            pady=10,
        )
        self._text_display.pack(fill=tk.BOTH, expand=True)

        # Text tags for formatting
        self._text_display.tag_configure("timestamp", foreground="#6a9955")
        self._text_display.tag_configure("en_label", foreground="#569cd6")
        self._text_display.tag_configure("en_text", foreground="#d4d4d4")
        self._text_display.tag_configure("ja_label", foreground="#c586c0")
        self._text_display.tag_configure("ja_text", foreground="#ce9178")
        self._text_display.tag_configure("system", foreground="#808080", font=("Consolas", 10, "italic"))

        # --- Status bar ---
        status_frame = ttk.Frame(root, padding=5)
        status_frame.pack(fill=tk.X)

        self._status_indicator = ttk.Label(status_frame, text="⏹")
        self._status_indicator.pack(side=tk.LEFT)

        self._status_label = ttk.Label(status_frame, text=" Stopped")
        self._status_label.pack(side=tk.LEFT, padx=(2, 15))

        self._entry_count_label = ttk.Label(
            status_frame, text="Entries: 0", foreground="gray"
        )
        self._entry_count_label.pack(side=tk.LEFT, padx=(0, 15))

        # Save button
        self._save_btn = ttk.Button(
            status_frame, text="💾 Save Transcript", command=self._save_transcript
        )
        self._save_btn.pack(side=tk.RIGHT)

        # Clear button
        self._clear_btn = ttk.Button(
            status_frame, text="🗑 Clear", command=self._clear_transcript
        )
        self._clear_btn.pack(side=tk.RIGHT, padx=5)

        # --- Initial actions ---
        self._refresh_devices()

        # Handle window close
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _refresh_devices(self) -> None:
        """Refresh the audio device list."""
        self._append_system_message("Scanning audio devices...")
        self._devices = self._audio_capture.get_devices()

        device_names = [str(d) for d in self._devices]
        self._device_combo["values"] = device_names

        if device_names:
            # Try to select the default device from config
            # Default to last device (typically a Loopback device)
            default_idx = len(device_names) - 1
            if self._config.default_device:
                for i, d in enumerate(self._devices):
                    if self._config.default_device.lower() in d.name.lower():
                        default_idx = i
                        break

            self._device_combo.current(default_idx)
            self._append_system_message(
                f"Found {len(self._devices)} audio device(s). "
                f"Selected: {self._devices[default_idx].name}"
            )
        else:
            self._append_system_message(
                "No audio devices found. Please check your audio settings."
            )

    def _start_capture(self) -> None:
        """Start audio capture and processing pipeline."""
        if self._is_recording:
            return

        # Get selected device
        idx = self._device_combo.current()
        if idx < 0 or idx >= len(self._devices):
            messagebox.showerror("Error", "Please select an audio device.")
            return

        device = self._devices[idx]

        # Check translation availability
        if self._translate_var.get() and not self._translator.is_available:
            result = messagebox.askokcancel(
                "Translation Unavailable",
                "DeepL API key is not set. Translation will be disabled.\n\n"
                "To enable translation, set 'deepl_api_key' in config.yaml\n"
                "or set the DEEPL_API_KEY environment variable.\n\n"
                "Continue without translation?",
            )
            if not result:
                return
            self._translate_var.set(False)

        try:
            # Clear queues
            self._drain_queue(self._audio_queue)
            self._drain_queue(self._text_queue)
            self._drain_queue(self._display_queue)

            # Reset VAD state
            self._vad.reset()

            # Start stop event
            self._stop_event.clear()

            # Start audio capture
            self._audio_capture.start(device, self._audio_queue)

            # Start STT worker thread
            self._stt_thread = threading.Thread(
                target=self._stt_worker,
                daemon=True,
                name="stt-worker",
            )
            self._stt_thread.start()

            # Start translation worker thread
            self._translation_thread = threading.Thread(
                target=self._translation_worker,
                daemon=True,
                name="translation-worker",
            )
            self._translation_thread.start()

            # Update UI state
            self._is_recording = True
            self._start_btn.config(state=tk.DISABLED)
            self._stop_btn.config(state=tk.NORMAL)
            self._device_combo.config(state=tk.DISABLED)
            self._status_indicator.config(text="🔴")
            self._status_label.config(text=" Recording...")

            self._append_system_message(f"Recording started: {device.name}")

            # Start polling for results
            self._poll_results()

        except Exception as e:
            logger.exception("Failed to start capture")
            messagebox.showerror("Error", f"Failed to start capture:\n{e}")
            self._cleanup_capture()

    def _cleanup_capture(self) -> None:
        """Force-cleanup capture resources regardless of state."""
        self._stop_event.set()
        self._audio_capture.stop()

        if self._stt_thread and self._stt_thread.is_alive():
            self._stt_thread.join(timeout=3.0)
        if self._translation_thread and self._translation_thread.is_alive():
            self._translation_thread.join(timeout=3.0)

        self._is_recording = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._device_combo.config(state="readonly")
        self._status_indicator.config(text="⏹")
        self._status_label.config(text=" Stopped")

    def _stop_capture(self) -> None:
        """Stop audio capture and processing pipeline."""
        if not self._is_recording:
            return

        self._append_system_message("Stopping...")

        # Signal threads to stop
        self._stop_event.set()

        # Stop audio capture
        self._audio_capture.stop()

        # Wait for threads to finish (with timeout)
        if self._stt_thread and self._stt_thread.is_alive():
            self._stt_thread.join(timeout=5.0)
        if self._translation_thread and self._translation_thread.is_alive():
            self._translation_thread.join(timeout=5.0)

        # Update UI state
        self._is_recording = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._device_combo.config(state="readonly")
        self._status_indicator.config(text="⏹")
        self._status_label.config(text=" Stopped")

        self._append_system_message("Recording stopped.")

    def _stt_worker(self) -> None:
        """Background thread: VAD + Whisper transcription.

        Reads audio chunks from the audio queue, runs VAD, and
        transcribes completed utterances.
        """
        logger.info("STT worker started")

        while not self._stop_event.is_set():
            try:
                # Get audio chunk (blocking with timeout)
                try:
                    audio_chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Run VAD
                utterances = self._vad.process_chunk(audio_chunk)

                # Transcribe completed utterances
                self._transcribe_utterances(utterances)

            except Exception:
                logger.exception("Error in STT worker")

        # Flush remaining audio in VAD buffer before exiting
        logger.info("STT worker flushing remaining audio...")
        remaining = self._vad.flush_remaining()
        if remaining is not None:
            self._transcribe_utterances([remaining])

        logger.info("STT worker stopped")

    def _transcribe_utterances(self, utterances: list[np.ndarray]) -> None:
        """Transcribe a list of audio utterances and queue results.

        Non-English audio is detected and skipped with a log message.
        """
        for utterance in utterances:
            result: TranscribeResult = self._transcriber.transcribe(utterance)
            if not result.text:
                continue

            # Skip non-English audio (multilingual model only)
            if result.language and result.language != "en":
                logger.info(
                    "Skipped non-English audio (detected: %s, prob=%.2f): %s",
                    result.language,
                    result.language_probability,
                    result.text[:60],
                )
                continue

            entry = TranscriptEntry(
                timestamp=datetime.now(),
                english_text=result.text,
            )
            try:
                self._text_queue.put(entry, timeout=1.0)
            except queue.Full:
                logger.warning("Text queue full, dropping entry")

    def _translation_worker(self) -> None:
        """Background thread: translates English text to Japanese.

        Reads TranscriptEntry from the text queue, translates if enabled,
        and puts results in the display queue.
        """
        logger.info("Translation worker started")

        while not self._stop_event.is_set():
            try:
                # Get text entry (blocking with timeout)
                try:
                    entry = self._text_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Translate if enabled
                if self._translate_var.get() and self._translator.is_available:
                    japanese = self._translator.translate(entry.english_text)
                    entry.japanese_text = japanese

                # Put in display queue
                try:
                    self._display_queue.put(entry, timeout=1.0)
                except queue.Full:
                    logger.warning("Display queue full, dropping entry")

            except Exception:
                logger.exception("Error in translation worker")

        logger.info("Translation worker stopped")

    def _poll_results(self) -> None:
        """Poll the display queue and update the GUI.

        Called periodically via root.after().
        """
        if not self._is_recording:
            return

        # Process all available entries
        processed = 0
        while processed < 10:  # Limit per poll cycle
            try:
                entry = self._display_queue.get_nowait()
                self._display_entry(entry)
                self._entries.append(entry)

                # Auto-save
                if self._autosave_var.get() and self._file_exporter.is_auto_saving:
                    self._file_exporter.append_entry(
                        entry, include_japanese=self._translate_var.get()
                    )

                processed += 1
            except queue.Empty:
                break

        if processed > 0:
            self._entry_count_label.config(text=f"Entries: {len(self._entries)}")

        # Schedule next poll
        if self._is_recording:
            self._root.after(POLL_INTERVAL_MS, self._poll_results)

    def _display_entry(self, entry: TranscriptEntry) -> None:
        """Display a transcript entry in the text widget."""
        self._text_display.config(state=tk.NORMAL)

        ts = entry.timestamp.strftime("%H:%M:%S")

        # English line
        self._text_display.insert(tk.END, f"[{ts}] ", "timestamp")
        self._text_display.insert(tk.END, "EN: ", "en_label")
        self._text_display.insert(tk.END, f"{entry.english_text}\n", "en_text")

        # Japanese line (if available)
        if entry.japanese_text:
            self._text_display.insert(tk.END, f"[{ts}] ", "timestamp")
            self._text_display.insert(tk.END, "JA: ", "ja_label")
            self._text_display.insert(tk.END, f"{entry.japanese_text}\n", "ja_text")

        self._text_display.insert(tk.END, "\n")

        # Auto-scroll to bottom
        self._text_display.see(tk.END)
        self._text_display.config(state=tk.DISABLED)

    def _append_system_message(self, message: str) -> None:
        """Display a system message in the text widget."""
        self._text_display.config(state=tk.NORMAL)
        ts = datetime.now().strftime("%H:%M:%S")
        self._text_display.insert(tk.END, f"[{ts}] {message}\n", "system")
        self._text_display.see(tk.END)
        self._text_display.config(state=tk.DISABLED)

    def _save_transcript(self) -> None:
        """Save the current transcript to a file."""
        if not self._entries:
            messagebox.showinfo("Info", "No transcript entries to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Transcript",
            initialfile=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        if not filepath:
            return

        try:
            self._file_exporter.save_transcript(
                self._entries,
                filepath,
                include_japanese=self._translate_var.get(),
            )
            self._append_system_message(f"Transcript saved: {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save transcript:\n{e}")

    def _clear_transcript(self) -> None:
        """Clear the transcript display and entries."""
        if self._entries:
            result = messagebox.askyesno(
                "Confirm",
                f"Clear {len(self._entries)} entries?\n"
                "This cannot be undone unless auto-save is enabled.",
            )
            if not result:
                return

        self._entries.clear()
        self._text_display.config(state=tk.NORMAL)
        self._text_display.delete("1.0", tk.END)
        self._text_display.config(state=tk.DISABLED)
        self._entry_count_label.config(text="Entries: 0")
        self._append_system_message("Transcript cleared.")

    def _toggle_autosave(self) -> None:
        """Toggle auto-save mode."""
        if self._autosave_var.get():
            # Start auto-save
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Select Auto-Save File",
                initialfile=f"autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )
            if filepath:
                try:
                    self._file_exporter.start_auto_save(filepath)
                    self._append_system_message(f"Auto-save started: {filepath}")
                except Exception as e:
                    self._autosave_var.set(False)
                    messagebox.showerror("Error", f"Failed to start auto-save:\n{e}")
            else:
                self._autosave_var.set(False)
        else:
            # Stop auto-save
            self._file_exporter.stop_auto_save()
            self._append_system_message("Auto-save stopped.")

    def _on_close(self) -> None:
        """Handle window close event."""
        if self._is_recording:
            self._stop_capture()

        self._file_exporter.stop_auto_save()
        self._root.destroy()

    def run(self) -> None:
        """Start the application main loop."""
        # Show config warnings
        warnings = self._config.validate()
        for warning in warnings:
            self._append_system_message(f"⚠ {warning}")

        self._append_system_message("Ready. Select an audio device and click Start.")
        self._root.mainloop()

    @staticmethod
    def _drain_queue(q: Queue) -> None:
        """Empty a queue."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
