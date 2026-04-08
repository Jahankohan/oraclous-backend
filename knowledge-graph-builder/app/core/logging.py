"""
Logging configuration for Oraclous Knowledge Graph Builder.

Supports two output formats controlled by LOG_FORMAT env var:
  - "text"  (default) — human-readable colored output for local dev
  - "json"  — structured JSON lines for production log aggregation

When OpenTelemetry is active, every JSON log record is enriched with
trace_id and span_id so logs can be correlated with distributed traces
in Jaeger / Grafana Tempo.
"""

import json
import logging
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record, enriched with OTel trace context."""

    # Standard LogRecord attributes we don't want to double-emit
    _SKIP = frozenset(
        {
            "args",
            "msg",
            "message",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "filename",
            "module",
            "pathname",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "levelno",
            "levelname",
            "name",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        from app.core.config import settings as _settings  # avoid circular import

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "service": _settings.SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject OTel trace context when available
        try:
            from app.core.telemetry import current_trace_context

            payload.update(current_trace_context())
        except Exception:
            pass

        # Merge extra= kwargs from the logger call site
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in self._SKIP:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class ColoredFormatter(logging.Formatter):
    """Human-readable colored formatter for local development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> None:
    """Configure root logger. Call once at application startup."""
    from app.core.config import settings  # avoid circular import at module load

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    use_json = getattr(settings, "LOG_FORMAT", "text").lower() == "json"

    if use_json:
        formatter: logging.Formatter = _JsonFormatter()
    else:
        text_format = "[%(asctime)s] %(levelname)s in %(name)s: %(message)s"
        formatter = (
            ColoredFormatter(text_format)
            if sys.stdout.isatty()
            else logging.Formatter(text_format)
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence chatty third-party libraries
    for noisy in ("neo4j", "httpx", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
