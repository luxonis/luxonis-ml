from __future__ import annotations

from luxonis_ml.telemetry.config import TelemetryConfig
from luxonis_ml.utils.environ import environ


def get_telemetry_config() -> TelemetryConfig | None:
    """Return the shared telemetry configuration for Luxonis ML."""
    api_key = (
        environ.LUXONIS_TELEMETRY_API_KEY.get_secret_value()
        if environ.LUXONIS_TELEMETRY_API_KEY is not None
        else None
    )
    return TelemetryConfig(
        enabled=bool(environ.LUXONIS_TELEMETRY_ENABLED),
        backend=environ.LUXONIS_TELEMETRY_BACKEND or "posthog",
        api_key=api_key,
        endpoint=environ.LUXONIS_TELEMETRY_ENDPOINT,
        debug=bool(environ.LUXONIS_TELEMETRY_DEBUG),
        install_id_path=environ.LUXONIS_TELEMETRY_INSTALL_ID_PATH,
        distinct_id=environ.LUXONIS_TELEMETRY_ID,
    )
