"""Structured JSON logging with a per-request correlation id.

Never pass clinical text, prompts or patient records to these loggers: log stage
names, ids, counts, durations and statuses only.
"""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED_ATTRS})
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


@contextmanager
def trace_stage(logger: logging.Logger, stage: str) -> Iterator[None]:
    """Log duration and outcome of one workflow stage.

    Single choke point so an OpenTelemetry span can be added here later without
    touching the agents.
    """
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        logger.info(
            "stage_finished",
            extra={
                "stage": stage,
                "status": status,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
        )
