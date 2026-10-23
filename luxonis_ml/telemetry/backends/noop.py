from typing_extensions import override

from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.events import TelemetryEvent


@TELEMETRY_BACKENDS.register(name="noop")
class NoopBackend(TelemetryBackend):
    """Backend that discards all telemetry events."""

    @override
    def capture(self, event: TelemetryEvent) -> None:
        """Discard the event."""
        return

    @override
    def flush(self) -> None:
        """No-op flush."""
        return
