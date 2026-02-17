"""Configuration management for Realtime English Transcriber.

Loads settings from config.yaml and environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Default config file path (project root)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


@dataclass
class AppConfig:
    """Application configuration."""

    deepl_api_key: str = ""
    whisper_model: str = "base.en"
    compute_type: str = "int8"
    translation_enabled: bool = True
    auto_save_path: str = ""
    default_device: str = ""
    vad_threshold: float = 0.5
    max_speech_duration: float = 30.0
    min_speech_ms: int = 250

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        """Load configuration from YAML file and environment variables.

        Priority: environment variables > config file > defaults.
        If the config file doesn't exist, creates one with default values.
        """
        config = cls()
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

        # Load from YAML file if it exists
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    else:
                        logger.warning("Unknown config key: %s", key)
            except Exception:
                logger.exception("Failed to load config from %s", config_path)
        else:
            logger.info("Config file not found at %s, creating default", config_path)
            config.save(config_path)

        # Environment variable overrides
        env_api_key = os.environ.get("DEEPL_API_KEY")
        if env_api_key:
            config.deepl_api_key = env_api_key

        env_model = os.environ.get("WHISPER_MODEL")
        if env_model:
            config.whisper_model = env_model

        return config

    def save(self, path: str | Path | None = None) -> None:
        """Save current configuration to YAML file."""
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        data = {
            "deepl_api_key": self.deepl_api_key,
            "whisper_model": self.whisper_model,
            "compute_type": self.compute_type,
            "translation_enabled": self.translation_enabled,
            "auto_save_path": self.auto_save_path,
            "default_device": self.default_device,
            "vad_threshold": self.vad_threshold,
            "max_speech_duration": self.max_speech_duration,
            "min_speech_ms": self.min_speech_ms,
        }
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info("Config saved to %s", config_path)
        except Exception:
            logger.exception("Failed to save config to %s", config_path)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warning messages."""
        warnings = []

        valid_models = {"tiny.en", "base.en", "small.en", "tiny", "base", "small", "medium"}
        if self.whisper_model not in valid_models:
            warnings.append(
                f"Unknown Whisper model: {self.whisper_model}. "
                f"Valid options: {', '.join(sorted(valid_models))}"
            )

        valid_compute = {"int8", "float32", "float16"}
        if self.compute_type not in valid_compute:
            warnings.append(
                f"Unknown compute type: {self.compute_type}. "
                f"Valid options: {', '.join(sorted(valid_compute))}"
            )

        if not 0.0 <= self.vad_threshold <= 1.0:
            warnings.append(
                f"VAD threshold {self.vad_threshold} out of range [0.0, 1.0]"
            )

        if self.max_speech_duration <= 0:
            warnings.append("max_speech_duration must be positive")

        if self.min_speech_ms < 0:
            warnings.append("min_speech_ms must be non-negative")

        if not self.deepl_api_key and self.translation_enabled:
            warnings.append(
                "Translation is enabled but DeepL API key is not set. "
                "Set deepl_api_key in config.yaml or DEEPL_API_KEY environment variable."
            )

        return warnings
