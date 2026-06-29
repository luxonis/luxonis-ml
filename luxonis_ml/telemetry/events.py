from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TelemetryEvent:
    """Represents a single telemetry event."""

    name: str
    timestamp: datetime
    properties: dict[str, Any]
    context: dict[str, Any]
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        name: str,
        properties: dict[str, Any],
        context: dict[str, Any],
        schema_version: int = 1,
    ) -> "TelemetryEvent":
        """Create a telemetry event with a UTC timestamp.

        Args:
            name: Event name.
            properties: Event properties after sanitization.
            context: Merged context to attach to the event.
            schema_version: Event schema version.

        Returns:
            A telemetry event with a UTC timestamp.
        """
        return cls(
            name=name,
            timestamp=datetime.now(timezone.utc),
            properties=properties,
            context=context,
            schema_version=schema_version,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the event into a JSON-friendly dict."""
        return {
            "event": self.name,
            "timestamp": self.timestamp.isoformat(),
            "properties": self.properties,
            "context": self.context,
            "schema_version": self.schema_version,
        }
