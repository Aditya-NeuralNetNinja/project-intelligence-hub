#!/usr/bin/env python3
"""Project Intelligence Hub - Application Entry Point."""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the GraphRAG application."""
    try:
        from src.config.settings import Settings
        from src.config.logging_config import configure_logging
        from src.ui.gradio_app import GradioApp

        configure_logging(use_colors=True)
        settings = Settings.from_env()
        app = GradioApp(settings=settings)
        app.launch()
        return 0

    except KeyboardInterrupt:
        return 0
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Application failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
