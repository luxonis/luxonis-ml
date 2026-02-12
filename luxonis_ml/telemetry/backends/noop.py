from __future__ import annotations

from luxonis_ml.telemetry.backends.base import TelemetryBackend
from luxonis_ml.telemetry.events import TelemetryEvent


class NoopBackend(TelemetryBackend):
    """Backend that discards all telemetry events."""

    def capture(self, event: TelemetryEvent) -> None:
        """Discard the event."""
        return

    def identify(self, user_id: str, traits: dict) -> None:
        """Discard the identify call."""
        return

    def flush(self) -> None:
        """No-op flush."""
        return

    def shutdown(self) -> None:
        """No-op shutdown."""
        return
