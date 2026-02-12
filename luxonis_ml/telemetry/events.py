from __future__ import annotations

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
    library: str
    library_version: str | None
    distinct_id: str | None
    user_id: str | None
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        name: str,
        properties: dict[str, Any],
        context: dict[str, Any],
        library: str,
        library_version: str | None,
        distinct_id: str | None,
        user_id: str | None,
        schema_version: int = 1,
    ) -> TelemetryEvent:
        """Create a telemetry event with a UTC timestamp.

        @type name: str
        @param name: Event name.
        @type properties: dict
        @param properties: Event properties after sanitization.
        @type context: dict
        @param context: Merged context to attach to the event.
        @type library: str
        @param library: Library name emitting the event.
        @type library_version: Optional[str]
        @param library_version: Library version string.
        @type distinct_id: Optional[str]
        @param distinct_id: Anonymous distinct id.
        @type user_id: Optional[str]
        @param user_id: Authenticated user id, if available.
        @type schema_version: int
        @param schema_version: Event schema version.
        """
        return cls(
            name=name,
            timestamp=datetime.now(timezone.utc),
            properties=properties,
            context=context,
            library=library,
            library_version=library_version,
            distinct_id=distinct_id,
            user_id=user_id,
            schema_version=schema_version,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the event into a JSON-friendly dict."""
        return {
            "event": self.name,
            "timestamp": self.timestamp.isoformat(),
            "properties": self.properties,
            "context": self.context,
            "library": self.library,
            "library_version": self.library_version,
            "schema_version": self.schema_version,
            "distinct_id": self.distinct_id,
            "user_id": self.user_id,
        }
