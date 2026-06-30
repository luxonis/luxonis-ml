from dataclasses import dataclass
from typing import TypeVar

from luxonis_ml.utils.environ import Environ


@dataclass(frozen=True)
class TelemetryDefaults:
    """Optional product-level defaults for telemetry configuration.

    Any field left as ``None`` falls back to the telemetry module's base
    default for that setting.
    """

    enabled: bool | None = None
    backend: str | None = None
    api_key: str | None = None
    endpoint: str | None = None
    debug: bool | None = None
    allowlist: set[str] | None = None
    include_base_context: bool | None = None
    include_system_metadata: bool | None = None


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
    def from_environ(
        cls,
        defaults: TelemetryDefaults | None = None,
    ) -> "TelemetryConfig":
        """Build a config from environment variables.

        Reads the ``LUXONIS_TELEMETRY_*`` settings and returns a fully
        populated ``TelemetryConfig`` instance. Environment variables
        take precedence over product-level defaults, which in turn take
        precedence over the telemetry module's base defaults.

        Args:
            defaults: Optional product-level defaults used when a given
                environment variable is not set.

        Returns:
            A telemetry configuration built from environment variables.
        """
        settings = Environ()
        defaults = defaults or TelemetryDefaults()
        base_defaults = cls()

        debug = _resolve_env_value(
            settings,
            "LUXONIS_TELEMETRY_DEBUG",
            value=settings.LUXONIS_TELEMETRY_DEBUG,
            default=_default_if_none(defaults.debug, base_defaults.debug),
        )
        enabled = _resolve_env_value(
            settings,
            "LUXONIS_TELEMETRY_ENABLED",
            value=settings.LUXONIS_TELEMETRY_ENABLED,
            default=_default_if_none(defaults.enabled, base_defaults.enabled),
        )
        backend = _resolve_env_value(
            settings,
            "LUXONIS_TELEMETRY_BACKEND",
            value=settings.LUXONIS_TELEMETRY_BACKEND,
            default=defaults.backend,
        ) or ("stdout" if debug else base_defaults.backend)
        api_key = _resolve_env_value(
            settings,
            "LUXONIS_TELEMETRY_API_KEY",
            value=(
                settings.LUXONIS_TELEMETRY_API_KEY.get_secret_value()
                if settings.LUXONIS_TELEMETRY_API_KEY is not None
                else None
            ),
            default=_default_if_none(defaults.api_key, base_defaults.api_key),
        )
        endpoint = _resolve_env_value(
            settings,
            "LUXONIS_TELEMETRY_ENDPOINT",
            value=settings.LUXONIS_TELEMETRY_ENDPOINT,
            default=_default_if_none(
                defaults.endpoint, base_defaults.endpoint
            ),
        )

        return cls(
            enabled=enabled,
            backend=backend,
            api_key=api_key,
            endpoint=endpoint,
            debug=debug,
            allowlist=_default_if_none(
                defaults.allowlist, base_defaults.allowlist
            ),
            include_base_context=_default_if_none(
                defaults.include_base_context,
                base_defaults.include_base_context,
            ),
            include_system_metadata=_default_if_none(
                defaults.include_system_metadata,
                base_defaults.include_system_metadata,
            ),
        )


T = TypeVar("T")


def _resolve_env_value(
    settings: Environ,
    field_name: str,
    *,
    value: T,
    default: T,
) -> T:
    return value if field_name in settings.model_fields_set else default


def _default_if_none(value: T | None, default: T) -> T:
    return default if value is None else value
