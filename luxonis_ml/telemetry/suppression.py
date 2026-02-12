from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import contextmanager

_telemetry_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_telemetry_suppressed", default=False
)


def is_suppressed() -> bool:
    """Return True when telemetry is suppressed in this context."""
    return _telemetry_suppressed.get()


@contextmanager
def suppress_telemetry() -> Generator[None, None, None]:
    """Context manager to suppress telemetry for nested calls."""
    token = _telemetry_suppressed.set(True)
    try:
        yield
    finally:
        _telemetry_suppressed.reset(token)
