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

        # Core components — two capture instances for output and mic
        self._audio_capture_output = AudioCapture(source_tag="output")
        self._audio_capture_mic = AudioCapture(source_tag="mic")
        # Per-source VAD instances to handle simultaneous speech independently
        self._vad_output = VoiceActivityDetector(
            threshold=config.vad_threshold,
            min_speech_ms=config.min_speech_ms,
            max_speech_s=config.max_speech_duration,
        )
        self._vad_mic = VoiceActivityDetector(
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
        # audio_queue now carries (source_tag, chunk) tuples
        self._audio_queue: Queue[tuple[str, np.ndarray]] = Queue(maxsize=AUDIO_QUEUE_MAX)
        self._text_queue: Queue[TranscriptEntry] = Queue(maxsize=TEXT_QUEUE_MAX)
        self._display_queue: Queue[TranscriptEntry] = Queue(maxsize=DISPLAY_QUEUE_MAX)

        # Thread control
        self._stop_event = threading.Event()
        self._stt_thread: threading.Thread | None = None
        self._translation_thread: threading.Thread | None = None

        # State
        self._output_devices: list[AudioDevice] = []
        self._input_devices: list[AudioDevice] = []
        self._entries: list[TranscriptEntry] = []
        self._is_recording = False

        # Build GUI
        self._root = tk.Tk()
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the Tkinter UI."""
        root = self._root
        root.title("Realtime English Transcriber")
        root.geometry("900x650")
        root.minsize(700, 450)

        # Sentinel value for disabled device selection
        self._disabled_label = "(無効)"

        # --- Top control bar: device selection ---
        device_frame = ttk.Frame(root, padding=5)
        device_frame.pack(fill=tk.X)

        # Output device selection
        ttk.Label(device_frame, text="🔊 Output:").pack(side=tk.LEFT, padx=(0, 3))
        self._output_device_var = tk.StringVar()
        self._output_device_combo = ttk.Combobox(
            device_frame,
            textvariable=self._output_device_var,
            state="readonly",
            width=30,
        )
        self._output_device_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._output_device_combo.bind("<<ComboboxSelected>>", lambda _: self._update_start_btn_state())

        # Mic device selection
        ttk.Label(device_frame, text="🎤 Mic:").pack(side=tk.LEFT, padx=(0, 3))
        self._mic_device_var = tk.StringVar()
        self._mic_device_combo = ttk.Combobox(
            device_frame,
            textvariable=self._mic_device_var,
            state="readonly",
            width=30,
        )
        self._mic_device_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._mic_device_combo.bind("<<ComboboxSelected>>", lambda _: self._update_start_btn_state())

        # Refresh devices button
        self._refresh_btn = ttk.Button(
            device_frame, text="🔄", width=3, command=self._refresh_devices
        )
        self._refresh_btn.pack(side=tk.LEFT, padx=(0, 5))

        # --- Control bar: Start / Stop / Reload ---
        control_frame = ttk.Frame(root, padding=5)
        control_frame.pack(fill=tk.X)

        self._start_btn = ttk.Button(
            control_frame, text="▶ Start", command=self._start_capture
        )
        self._start_btn.pack(side=tk.LEFT, padx=2)

        self._stop_btn = ttk.Button(
            control_frame, text="■ Stop", command=self._stop_capture, state=tk.DISABLED
        )
        self._stop_btn.pack(side=tk.LEFT, padx=2)

        self._reload_btn = ttk.Button(
            control_frame, text="⟳ Reload Config", command=self._reload_config
        )
        self._reload_btn.pack(side=tk.LEFT, padx=(15, 2))

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

        # Japanese transcription toggle
        self._jp_transcription_var = tk.BooleanVar(
            value=self._config.japanese_transcription_enabled
        )
        self._jp_transcription_check = ttk.Checkbutton(
            options_frame,
            text="JP文字起こし",
            variable=self._jp_transcription_var,
        )
        self._jp_transcription_check.pack(side=tk.LEFT, padx=(0, 15))
        # Disable if using English-only model (no language detection)
        if self._config.whisper_model.endswith(".en"):
            self._jp_transcription_check.config(state=tk.DISABLED)
            self._jp_transcription_var.set(False)

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
        # Mic-specific tags (cyan tones for contrast)
        self._text_display.tag_configure("mic_label", foreground="#4ec9b0")
        self._text_display.tag_configure("mic_text", foreground="#9cdcfe")
        self._text_display.tag_configure("mic_ja_label", foreground="#c586c0")
        self._text_display.tag_configure("mic_ja_text", foreground="#ce9178")
        # JP language tags
        self._text_display.tag_configure("jp_label", foreground="#dcdcaa")
        self._text_display.tag_configure("jp_text", foreground="#d4d4d4")
        self._text_display.tag_configure("mic_jp_label", foreground="#dcdcaa")
        self._text_display.tag_configure("mic_jp_text", foreground="#9cdcfe")
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
        """Refresh both output and microphone device lists."""
        self._append_system_message("Scanning audio devices...")

        # Get all devices via the output capture instance (shares the same backend)
        all_devices = self._audio_capture_output.get_devices()

        # Split into output (loopback) and input (microphone) devices
        self._output_devices = [
            d for d in all_devices
            if d.is_loopback or (d.max_output_channels > 0 and d.max_input_channels == 0)
        ]
        self._input_devices = [
            d for d in all_devices
            if d.max_input_channels > 0 and not d.is_loopback
        ]

        # Populate output device combo
        output_names = [self._disabled_label] + [str(d) for d in self._output_devices]
        self._output_device_combo["values"] = output_names

        # Try to select default output device from config
        output_idx = 0  # default to disabled
        if self._config.default_device:
            for i, d in enumerate(self._output_devices):
                if self._config.default_device.lower() in d.name.lower():
                    output_idx = i + 1  # +1 because of disabled label at index 0
                    break
        elif self._output_devices:
            output_idx = len(output_names) - 1  # last device (typically loopback)
        self._output_device_combo.current(output_idx)

        # Populate mic device combo
        mic_names = [self._disabled_label] + [str(d) for d in self._input_devices]
        self._mic_device_combo["values"] = mic_names

        # Try to select default mic device from config
        mic_idx = 0  # default to disabled
        if self._config.mic_device:
            for i, d in enumerate(self._input_devices):
                if self._config.mic_device.lower() in d.name.lower():
                    mic_idx = i + 1
                    break
        self._mic_device_combo.current(mic_idx)

        self._update_start_btn_state()

        total = len(self._output_devices) + len(self._input_devices)
        self._append_system_message(
            f"Found {len(self._output_devices)} output device(s), "
            f"{len(self._input_devices)} input device(s)."
        )

    def _update_start_btn_state(self) -> None:
        """Enable Start button only if at least one device is selected."""
        if self._is_recording:
            return
        output_selected = self._output_device_var.get() != self._disabled_label
        mic_selected = self._mic_device_var.get() != self._disabled_label
        if output_selected or mic_selected:
            self._start_btn.config(state=tk.NORMAL)
        else:
            self._start_btn.config(state=tk.DISABLED)

    def _start_capture(self) -> None:
        """Start audio capture and processing pipeline."""
        if self._is_recording:
            return

        # Determine selected devices
        output_sel = self._output_device_var.get()
        mic_sel = self._mic_device_var.get()
        use_output = output_sel != self._disabled_label
        use_mic = mic_sel != self._disabled_label

        if not use_output and not use_mic:
            messagebox.showerror("Error", "少なくとも1つのデバイスを選択してください。")
            return

        output_device: AudioDevice | None = None
        mic_device: AudioDevice | None = None

        if use_output:
            out_idx = self._output_device_combo.current() - 1  # -1 for disabled label
            if 0 <= out_idx < len(self._output_devices):
                output_device = self._output_devices[out_idx]

        if use_mic:
            mic_idx = self._mic_device_combo.current() - 1  # -1 for disabled label
            if 0 <= mic_idx < len(self._input_devices):
                mic_device = self._input_devices[mic_idx]

        if not output_device and not mic_device:
            messagebox.showerror("Error", "Please select a valid audio device.")
            return

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
            self._vad_output.reset()
            self._vad_mic.reset()

            # Start stop event
            self._stop_event.clear()

            # Start audio capture(s)
            started_sources = []
            if output_device:
                self._audio_capture_output.start(output_device, self._audio_queue)
                started_sources.append(f"🔊 {output_device.name}")
            if mic_device:
                self._audio_capture_mic.start(mic_device, self._audio_queue)
                started_sources.append(f"🎤 {mic_device.name}")

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
            self._reload_btn.config(state=tk.DISABLED)
            self._output_device_combo.config(state=tk.DISABLED)
            self._mic_device_combo.config(state=tk.DISABLED)
            self._status_indicator.config(text="🔴")
            self._status_label.config(text=" Recording...")

            self._append_system_message(
                f"Recording started: {', '.join(started_sources)}"
            )

            # Start polling for results
            self._poll_results()

        except Exception as e:
            logger.exception("Failed to start capture")
            messagebox.showerror("Error", f"Failed to start capture:\n{e}")
            self._cleanup_capture()

    def _cleanup_capture(self) -> None:
        """Force-cleanup capture resources regardless of state."""
        self._stop_event.set()
        self._audio_capture_output.stop()
        self._audio_capture_mic.stop()

        if self._stt_thread and self._stt_thread.is_alive():
            self._stt_thread.join(timeout=3.0)
        if self._translation_thread and self._translation_thread.is_alive():
            self._translation_thread.join(timeout=3.0)

        self._is_recording = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._reload_btn.config(state=tk.NORMAL)
        self._output_device_combo.config(state="readonly")
        self._mic_device_combo.config(state="readonly")
        self._status_indicator.config(text="⏹")
        self._status_label.config(text=" Stopped")
        self._update_start_btn_state()

    def _stop_capture(self) -> None:
        """Stop audio capture and processing pipeline."""
        if not self._is_recording:
            return

        self._append_system_message("Stopping...")

        # Signal threads to stop
        self._stop_event.set()

        # Stop audio capture(s)
        self._audio_capture_output.stop()
        self._audio_capture_mic.stop()

        # Wait for threads to finish (with timeout)
        if self._stt_thread and self._stt_thread.is_alive():
            self._stt_thread.join(timeout=5.0)
        if self._translation_thread and self._translation_thread.is_alive():
            self._translation_thread.join(timeout=5.0)

        # Update UI state
        self._is_recording = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._reload_btn.config(state=tk.NORMAL)
        self._output_device_combo.config(state="readonly")
        self._mic_device_combo.config(state="readonly")
        self._status_indicator.config(text="⏹")
        self._status_label.config(text=" Stopped")
        self._update_start_btn_state()

        self._append_system_message("Recording stopped.")

    def _stt_worker(self) -> None:
        """Background thread: VAD + Whisper transcription.

        Reads (source_tag, audio_chunk) tuples from the audio queue,
        dispatches to the corresponding per-source VAD instance,
        and transcribes completed utterances.
        """
        logger.info("STT worker started")

        vad_map = {
            "output": self._vad_output,
            "mic": self._vad_mic,
        }

        while not self._stop_event.is_set():
            try:
                # Get audio chunk (blocking with timeout)
                try:
                    source_tag, audio_chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Select VAD instance for this source
                vad = vad_map.get(source_tag, self._vad_output)

                # Run VAD
                utterances = vad.process_chunk(audio_chunk)

                # Transcribe completed utterances
                self._transcribe_utterances(utterances, source_tag)

            except Exception:
                logger.exception("Error in STT worker")

        # Flush remaining audio in both VAD buffers before exiting
        logger.info("STT worker flushing remaining audio...")
        for tag, vad in vad_map.items():
            remaining = vad.flush_remaining()
            if remaining is not None:
                self._transcribe_utterances([remaining], tag)

        logger.info("STT worker stopped")

    def _transcribe_utterances(
        self, utterances: list[np.ndarray], source: str = "output"
    ) -> None:
        """Transcribe a list of audio utterances and queue results.

        Applies language filter based on the JP transcription toggle.
        """
        for utterance in utterances:
            result: TranscribeResult = self._transcriber.transcribe(utterance)
            if not result.text:
                continue

            detected_lang = result.language or "en"

            # Language filtering (multilingual model only)
            if detected_lang != "en" and not self._jp_transcription_var.get():
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
                source=source,
                language=detected_lang,
            )
            try:
                self._text_queue.put(entry, timeout=1.0)
            except queue.Full:
                logger.warning("Text queue full, dropping entry")

    def _translation_worker(self) -> None:
        """Background thread: translates English text to Japanese.

        Reads TranscriptEntry from the text queue, translates if enabled
        and the entry is in English, and puts results in the display queue.
        Non-English entries (e.g. Japanese) pass through without translation.
        """
        logger.info("Translation worker started")

        while not self._stop_event.is_set():
            try:
                # Get text entry (blocking with timeout)
                try:
                    entry = self._text_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Only translate English entries
                if (
                    entry.language == "en"
                    and self._translate_var.get()
                    and self._translator.is_available
                ):
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
        """Display a transcript entry in the text widget with source/language styling."""
        self._text_display.config(state=tk.NORMAL)

        ts = entry.timestamp.strftime("%H:%M:%S")
        is_mic = entry.source == "mic"
        source_icon = "🎤" if is_mic else "🔊"
        lang_label = entry.language.upper() if entry.language else "EN"

        # Choose tag sets based on source
        if is_mic:
            # Mic source uses cyan-toned tags
            if lang_label == "EN":
                main_label_tag = "mic_label"
                main_text_tag = "mic_text"
            else:
                main_label_tag = "mic_jp_label"
                main_text_tag = "mic_jp_text"
            ja_label_tag = "mic_ja_label"
            ja_text_tag = "mic_ja_text"
        else:
            # Output source uses standard tags
            if lang_label == "EN":
                main_label_tag = "en_label"
                main_text_tag = "en_text"
            else:
                main_label_tag = "jp_label"
                main_text_tag = "jp_text"
            ja_label_tag = "ja_label"
            ja_text_tag = "ja_text"

        # Main text line
        self._text_display.insert(tk.END, f"[{ts}] ", "timestamp")
        self._text_display.insert(tk.END, f"{source_icon} ", "timestamp")
        self._text_display.insert(tk.END, f"{lang_label}: ", main_label_tag)
        self._text_display.insert(tk.END, f"{entry.english_text}\n", main_text_tag)

        # Japanese translation line (if available, only for EN entries)
        if entry.japanese_text:
            self._text_display.insert(tk.END, f"[{ts}] ", "timestamp")
            self._text_display.insert(tk.END, f"{source_icon} ", "timestamp")
            self._text_display.insert(tk.END, "JA: ", ja_label_tag)
            self._text_display.insert(tk.END, f"{entry.japanese_text}\n", ja_text_tag)

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

    def _reload_config(self) -> None:
        """Reload config.yaml and apply changes to components.

        Only available when recording is stopped.
        """
        if self._is_recording:
            self._append_system_message("⚠ 録音中は設定リロードできません。先に停止してください。")
            return

        changes = self._config.reload()

        if not changes:
            self._append_system_message("設定を再読み込みしました（変更なし）。")
            return

        # Apply changes to components
        if "whisper_model" in changes or "compute_type" in changes:
            self._transcriber = Transcriber(
                model_size=self._config.whisper_model,
                compute_type=self._config.compute_type,
            )
            self._model_label.config(text=f"Model: {self._config.whisper_model}")

            # Update JP transcription checkbox state based on model type
            if self._config.whisper_model.endswith(".en"):
                self._jp_transcription_check.config(state=tk.DISABLED)
                self._jp_transcription_var.set(False)
            else:
                self._jp_transcription_check.config(state=tk.NORMAL)

        if "deepl_api_key" in changes:
            self._translator = Translator(api_key=self._config.deepl_api_key)

        if any(k in changes for k in ("vad_threshold", "max_speech_duration", "min_speech_ms")):
            self._vad_output = VoiceActivityDetector(
                threshold=self._config.vad_threshold,
                min_speech_ms=self._config.min_speech_ms,
                max_speech_s=self._config.max_speech_duration,
            )
            self._vad_mic = VoiceActivityDetector(
                threshold=self._config.vad_threshold,
                min_speech_ms=self._config.min_speech_ms,
                max_speech_s=self._config.max_speech_duration,
            )

        if "translation_enabled" in changes:
            self._translate_var.set(self._config.translation_enabled)

        if "japanese_transcription_enabled" in changes:
            if not self._config.whisper_model.endswith(".en"):
                self._jp_transcription_var.set(self._config.japanese_transcription_enabled)

        if "default_device" in changes or "mic_device" in changes:
            self._refresh_devices()

        # Display summary of changes
        change_lines = []
        for key, (old, new) in changes.items():
            if key == "deepl_api_key":
                change_lines.append(f"  {key}: (hidden) → (hidden)")
            else:
                change_lines.append(f"  {key}: {old} → {new}")
        self._append_system_message(
            f"設定を再読み込みしました（{len(changes)}件変更）:"
        )
        for line in change_lines:
            self._append_system_message(line)

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
