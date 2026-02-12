from __future__ import annotations

from luxonis_ml.telemetry.cli import instrument_typer, skip_telemetry
from luxonis_ml.telemetry.client import Telemetry
from luxonis_ml.telemetry.config import TelemetryConfig
from luxonis_ml.telemetry.singleton import (
    get_telemetry,
    initialize_telemetry,
    shutdown_on_exit,
)
from luxonis_ml.telemetry.suppression import suppress_telemetry

__all__ = [
    "Telemetry",
    "TelemetryConfig",
    "get_telemetry",
    "initialize_telemetry",
    "instrument_typer",
    "shutdown_on_exit",
    "skip_telemetry",
    "suppress_telemetry",
]
