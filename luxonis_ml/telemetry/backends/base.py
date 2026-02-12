from __future__ import annotations

from typing import Protocol

from luxonis_ml.telemetry.events import TelemetryEvent


class TelemetryBackend(Protocol):
    """Protocol for telemetry backends."""

    def capture(self, event: TelemetryEvent) -> None:
        """Capture a telemetry event."""
        ...

    def identify(self, user_id: str, traits: dict) -> None:
        """Associate a user id with optional traits."""
        ...

    def flush(self) -> None:
        """Flush any buffered events."""
        ...

    def shutdown(self) -> None:
        """Shutdown backend resources and flush pending events."""
        ...
