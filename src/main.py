"""Realtime English Transcriber - Entry Point.

Loads configuration and launches the GUI application.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on the path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def setup_logging() -> None:
    """Configure logging for the application."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    # Catch uncaught exceptions in threads
    def handle_thread_exception(args):
        logging.getLogger(__name__).exception(
            "Uncaught exception in thread %s", args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    import threading
    threading.excepthook = handle_thread_exception


def main() -> None:
    """Application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Realtime English Transcriber")

    try:
        # Load configuration
        from src.utils.config import AppConfig

        config = AppConfig.load()

        # Show config warnings
        warnings = config.validate()
        for warning in warnings:
            logger.warning("Config: %s", warning)

        # Launch GUI
        from src.gui.app import TranscriberApp

        app = TranscriberApp(config)
        app.run()

    except Exception:
        logger.exception("Fatal error")
        raise
    finally:
        logger.info("Application exited")


if __name__ == "__main__":
    main()
