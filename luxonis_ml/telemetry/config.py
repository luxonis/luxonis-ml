from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from luxonis_ml.utils.environ import environ


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for telemetry collection.

    @type enabled: bool
    @param enabled: Whether telemetry should emit events.
    @type backend: str
    @param backend: Backend name to use (e.g., "posthog", "stdout").
    @type api_key: Optional[str]
    @param api_key: API key for the backend, if applicable.
    @type endpoint: Optional[str]
    @param endpoint: Custom endpoint/host for the backend, if
        applicable.
    @type debug: bool
    @param debug: If True, enables debug behavior (e.g., stdout
        backend).
    @type allowlist: Optional[Set[str]]
    @param allowlist: If set, only these CLI params are logged.
    @type distinct_id: Optional[str]
    @param distinct_id: Override for the anonymous distinct id.
    @type include_system_metadata: bool
    @param include_system_metadata: Include extended system metadata by
        default.
    @type install_id_path: Optional[Path]
    @param install_id_path: Path to persist the anonymous install id.
    """

    enabled: bool = False
    backend: str = "posthog"
    api_key: str | None = None
    endpoint: str | None = None
    debug: bool = False
    allowlist: set[str] | None = None
    install_id_path: Path | None = None
    distinct_id: str | None = None
    include_system_metadata: bool = False

    @classmethod
    def from_environ(cls) -> TelemetryConfig:
        """Build a config from environment variables.

        This reads the C{LUXONIS_TELEMETRY_*} settings and returns a
        fully populated L{TelemetryConfig} instance.
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
        install_id_path = environ.LUXONIS_TELEMETRY_INSTALL_ID_PATH
        distinct_id = environ.LUXONIS_TELEMETRY_ID
        return cls(
            enabled=enabled,
            backend=backend,
            api_key=api_key,
            endpoint=endpoint,
            debug=debug,
            install_id_path=install_id_path,
            distinct_id=distinct_id,
        )
