from __future__ import annotations

import json

from loguru import logger

from luxonis_ml.telemetry.backends.base import TelemetryBackend
from luxonis_ml.telemetry.events import TelemetryEvent


class StdoutBackend(TelemetryBackend):
    """Backend that logs telemetry to stdout."""

    def __init__(self) -> None:
        """Initialize the stdout backend."""

    def capture(self, event: TelemetryEvent) -> None:
        """Log a telemetry event as JSON."""
        payload = event.to_payload()
        logger.info(f"telemetry: {json.dumps(payload, sort_keys=True)}")

    def identify(self, user_id: str, traits: dict) -> None:
        """Log an identify call as JSON."""
        payload = {"user_id": user_id, "traits": traits}
        logger.info(f"telemetry_identify: {json.dumps(payload)}")

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return
