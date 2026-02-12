from __future__ import annotations

from typing import Any

from luxonis_ml.telemetry.backends.base import TelemetryBackend
from luxonis_ml.telemetry.events import TelemetryEvent


class PostHogBackend(TelemetryBackend):
    """PostHog backend using the official python package."""

    def __init__(self, *, api_key: str, host: str | None = None) -> None:
        """Initialize the PostHog client.

        @type api_key: str
        @param api_key: PostHog project API key.
        @type host: Optional[str]
        @param host: Optional PostHog host URL.
        """
        if not api_key:
            raise ValueError("PostHog backend requires an API key.")
        try:
            from posthog import Posthog
        except ImportError as exc:
            raise ImportError(
                "PostHog backend requires the 'posthog' package."
            ) from exc
        kwargs = {"project_api_key": api_key, "disable_geoip": True}
        if host:
            kwargs["host"] = host
        self._client = Posthog(**kwargs)

    def capture(self, event: TelemetryEvent) -> None:
        """Capture an event using the PostHog client."""
        distinct_id = event.user_id or event.distinct_id or "anonymous"
        properties = _merge_properties(event)
        self._client.capture(
            distinct_id=distinct_id,
            event=event.name,
            properties=properties,
            timestamp=event.timestamp,
        )

    def identify(self, user_id: str, traits: dict) -> None:
        """Identify a user with optional traits."""
        self._client.identify(user_id, traits=traits)

    def flush(self) -> None:
        """Flush buffered PostHog events if supported."""
        if hasattr(self._client, "flush"):
            self._client.flush()

    def shutdown(self) -> None:
        self.flush()


def _merge_properties(event: TelemetryEvent) -> dict[str, Any]:
    """Merge event metadata into a single properties dict."""
    properties = {
        "library": event.library,
        "library_version": event.library_version,
        "schema_version": event.schema_version,
    }
    properties.update(event.context)
    properties.update(event.properties)
    return properties
