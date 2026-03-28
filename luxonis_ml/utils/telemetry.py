from __future__ import annotations

from luxonis_ml.telemetry.config import TelemetryConfig


def get_telemetry_config() -> TelemetryConfig:
    """Return the Luxonis ML telemetry config with project defaults."""
    config = TelemetryConfig.from_environ()  # TODO
    return TelemetryConfig(
        enabled=config.enabled,
        backend=config.backend,
        api_key=config.api_key,  # or "#TODO",
        endpoint=config.endpoint,  # or "#TODO",
        debug=config.debug,
        allowlist=config.allowlist,
        distinct_id=config.distinct_id,
        include_base_context=config.include_base_context,
        include_system_metadata=config.include_system_metadata,
    )
