"""Small structured JSON logging setup for experiment infrastructure."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        return json.dumps(payload, default=str)


def configure_logging(log_file: str | Path, level: str = "INFO") -> logging.Logger:
    """Create an isolated experiment logger that writes JSON Lines to *log_file*."""
    target = Path(log_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"adaptive_plasticity.{target.resolve()}")
    logger.setLevel(level.upper())
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger

