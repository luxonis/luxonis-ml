from typing import Any

from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.config import TelemetryConfig
from luxonis_ml.telemetry.events import TelemetryEvent


@TELEMETRY_BACKENDS.register(name="posthog")
class PostHogBackend(TelemetryBackend):
    """PostHog backend using the official python package."""

    def __init__(self, config: TelemetryConfig) -> None:
        """Initialize the PostHog client.

        @type config: L{TelemetryConfig}
        @param config: Telemetry backend configuration.
        """
        super().__init__(config)
        if not config.api_key:
            raise ValueError("PostHog backend requires an API key.")
        try:
            from posthog import Posthog
        except ImportError as exc:
            raise ImportError(
                "PostHog backend requires the 'posthog' package."
            ) from exc
        kwargs = {
            "project_api_key": config.api_key,
            "disable_geoip": False,
        }
        if config.endpoint:
            kwargs["host"] = config.endpoint
        self._client = Posthog(**kwargs)

    def capture(self, event: TelemetryEvent) -> None:
        """Capture an event using the PostHog client."""
        properties = _merge_properties(event)
        self._client.capture(
            distinct_id=event.context["$session_id"],
            event=event.name,
            properties=properties,
            timestamp=event.timestamp,
        )

    def flush(self) -> None:
        """Flush buffered PostHog events if supported."""
        if hasattr(self._client, "flush"):
            self._client.flush()


def _merge_properties(event: TelemetryEvent) -> dict[str, Any]:
    """Merge event metadata into a single properties dict."""
    properties = {"schema_version": event.schema_version}
    properties.update(event.context)
    properties.update(event.properties)
    return properties
