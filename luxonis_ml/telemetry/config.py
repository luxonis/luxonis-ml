from dataclasses import dataclass

from luxonis_ml.utils.environ import environ


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for telemetry collection.

    Attributes:
        enabled: Whether telemetry should emit events.
        backend: Backend name to use, such as ``"posthog"`` or
            ``"stdout"``.
        api_key: API key for the backend, if applicable.
        endpoint: Custom endpoint or host for the backend, if
            applicable.
        debug: Whether to enable debug behavior such as the stdout
            backend default.
        allowlist: CLI parameter names that may be logged.
        include_base_context: Whether to include the shared default
            context on every event.
        include_system_metadata: Whether to include extended system
            metadata by default.
    """

    enabled: bool = True
    backend: str = "posthog"
    api_key: str | None = None
    endpoint: str | None = None
    debug: bool = False
    allowlist: set[str] | None = None
    include_base_context: bool = True
    include_system_metadata: bool = False

    @classmethod
    def from_environ(cls) -> "TelemetryConfig":
        """Build a config from environment variables.

        Reads the ``LUXONIS_TELEMETRY_*`` settings and returns a fully
        populated ``TelemetryConfig`` instance.

        Returns:
            A telemetry configuration built from environment variables.
        """
        debug = bool(environ.LUXONIS_TELEMETRY_DEBUG)
        enabled = bool(environ.LUXONIS_TELEMETRY_ENABLED)
        backend = environ.LUXONIS_TELEMETRY_BACKEND or (
            "stdout" if debug else "posthog"
        )
        api_key = (
            environ.LUXONIS_TELEMETRY_API_KEY.get_secret_value()
            if environ.LUXONIS_TELEMETRY_API_KEY is not None
            else None
        )
        endpoint = environ.LUXONIS_TELEMETRY_ENDPOINT
        return cls(
            enabled=enabled,
            backend=backend,
            api_key=api_key,
            endpoint=endpoint,
            debug=debug,
        )
