from __future__ import annotations

from luxonis_ml.telemetry.config import TelemetryConfig


def get_telemetry_config() -> TelemetryConfig:
    """Return the shared telemetry configuration for Luxonis ML."""
    return TelemetryConfig.from_environ()
