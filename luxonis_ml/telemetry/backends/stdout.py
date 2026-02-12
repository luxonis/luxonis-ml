from __future__ import annotations

import json
import logging

from luxonis_ml.telemetry.backends.base import TelemetryBackend
from luxonis_ml.telemetry.events import TelemetryEvent


class StdoutBackend(TelemetryBackend):
    """Backend that logs telemetry to stdout."""

    def __init__(self) -> None:
        """Initialize the stdout backend."""
        self._logger = logging.getLogger("luxonis_ml.telemetry")

    def capture(self, event: TelemetryEvent) -> None:
        """Log a telemetry event as JSON."""
        payload = event.to_payload()
        self._logger.info("telemetry %s", json.dumps(payload, sort_keys=True))

    def identify(self, user_id: str, traits: dict) -> None:
        """Log an identify call as JSON."""
        payload = {"user_id": user_id, "traits": traits}
        self._logger.info("telemetry_identify %s", json.dumps(payload))

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return
