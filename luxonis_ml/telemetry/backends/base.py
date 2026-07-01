from abc import ABC, abstractmethod

from luxonis_ml.telemetry.config import TelemetryConfig
from luxonis_ml.telemetry.events import TelemetryEvent
from luxonis_ml.utils.registry import Registry

TELEMETRY_BACKENDS: Registry[type["TelemetryBackend"]] = Registry(
    name="telemetry_backends"
)


class TelemetryBackend(ABC):
    """Base class for telemetry backends."""

    def __init__(self, config: TelemetryConfig) -> None:
        self.config = config

    @abstractmethod
    def capture(self, event: TelemetryEvent) -> None:
        """Capture a telemetry event."""

    def flush(self) -> None:
        """Flush any buffered events."""
        return

    def shutdown(self) -> None:
        """Shutdown backend resources and flush pending events."""
        self.flush()
