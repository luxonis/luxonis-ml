"""Telemetry backends and the `TELEMETRY_BACKENDS` registry.

The built-in backends are `PostHogBackend`, `StdoutBackend`, and
`NoopBackend`. Each one subclasses `TelemetryBackend`.
"""

from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.backends.noop import NoopBackend
from luxonis_ml.telemetry.backends.posthog import PostHogBackend
from luxonis_ml.telemetry.backends.stdout import StdoutBackend

__all__ = [
    "TELEMETRY_BACKENDS",
    "NoopBackend",
    "PostHogBackend",
    "StdoutBackend",
    "TelemetryBackend",
]
