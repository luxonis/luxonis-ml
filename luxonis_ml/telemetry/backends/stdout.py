import json

from loguru import logger

from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.events import TelemetryEvent


@TELEMETRY_BACKENDS.register(name="stdout")
class StdoutBackend(TelemetryBackend):
    """Backend that logs telemetry to stdout."""

    def capture(self, event: TelemetryEvent) -> None:
        """Log a telemetry event as JSON."""
        payload = event.to_payload()
        logger.info(f"telemetry: {json.dumps(payload, sort_keys=True)}")
