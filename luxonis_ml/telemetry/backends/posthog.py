from typing import Any

from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.config import TelemetryConfig
from luxonis_ml.telemetry.context import RESERVED_METADATA_KEYS
from luxonis_ml.telemetry.events import TelemetryEvent

PROTECTED_POSTHOG_PROPERTY_KEYS = frozenset(
    {*RESERVED_METADATA_KEYS, "schema_version"}
)


@TELEMETRY_BACKENDS.register(name="posthog")
class PostHogBackend(TelemetryBackend):
    """PostHog backend using the official python package."""

    def __init__(self, config: TelemetryConfig) -> None:
        """Initialize the PostHog client.

        Args:
            config: Telemetry backend configuration.

        Raises:
            ValueError: If the PostHog API key is missing.
            ImportError: If the ``posthog`` package is not installed.
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
            "disable_geoip": config.disable_geoip,
        }
        if config.endpoint:
            kwargs["host"] = config.endpoint
        self._client = Posthog(**kwargs)

    def capture(self, event: TelemetryEvent) -> None:
        """Capture an event using the PostHog client."""
        properties = _merge_properties(
            event,
            allow_reserved_overrides=self.config.allow_reserved_overrides,
        )
        self._client.capture(
            distinct_id=event.distinct_id or event.context["$session_id"],
            event=event.name,
            properties=properties,
            timestamp=event.timestamp,
        )

    def flush(self) -> None:
        """Flush buffered PostHog events if supported."""
        if hasattr(self._client, "flush"):
            self._client.flush()


def _merge_properties(
    event: TelemetryEvent,
    *,
    allow_reserved_overrides: bool = False,
) -> dict[str, Any]:
    """Merge event metadata into a single properties dict."""
    properties = dict(event.context)
    properties.update(event.properties)
    properties["schema_version"] = event.schema_version
    if allow_reserved_overrides:
        return properties
    for key in PROTECTED_POSTHOG_PROPERTY_KEYS:
        if key == "schema_version":
            continue
        if key in event.context:
            properties[key] = event.context[key]
    return properties
