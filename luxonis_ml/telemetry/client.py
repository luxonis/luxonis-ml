from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

from loguru import logger

from luxonis_ml.telemetry.backends.base import TelemetryBackend
from luxonis_ml.telemetry.backends.noop import NoopBackend
from luxonis_ml.telemetry.backends.posthog import PostHogBackend
from luxonis_ml.telemetry.backends.stdout import StdoutBackend
from luxonis_ml.telemetry.config import TelemetryConfig
from luxonis_ml.telemetry.context import (
    base_context,
    default_install_id_path,
    load_install_id,
    system_context,
)
from luxonis_ml.telemetry.events import TelemetryEvent
from luxonis_ml.telemetry.redaction import sanitize_properties
from luxonis_ml.telemetry.suppression import is_suppressed

ContextProvider = Callable[["Telemetry"], dict[str, Any]]
SystemContextProvider = Callable[["Telemetry"], dict[str, Any]]
BackendFactory = Callable[[TelemetryConfig], TelemetryBackend]


class Telemetry:
    """Telemetry client for emitting events via pluggable backends."""

    _backend_factories: dict[str, BackendFactory] = {}
    _logged_enabled_notice: bool = False

    def __init__(
        self,
        library_name: str,
        *,
        library_version: str | None = None,
        config: TelemetryConfig | None = None,
        context_providers: list[ContextProvider] | None = None,
        system_context_providers: list[SystemContextProvider] | None = None,
    ) -> None:
        """Initialize a telemetry client.

        @type library_name: str
        @param library_name: Name of the library emitting telemetry
            (e.g. C{"luxonis_ml"}).
        @type library_version: Optional[str]
        @param library_version: Version string for the library. If
            C{None}, it is resolved from package metadata when possible.
        @type config: Optional[L{TelemetryConfig}]
        @param config: TelemetryConfig to use. If C{None}, values are
            read from environment variables.
        @type context_providers: Optional[list]
        @param context_providers: Callables that return extra context to
            attach to every event.
        @type system_context_providers: Optional[list]
        @param system_context_providers: Callables that return extra
            context to attach only when system metadata is requested.
        """
        self._config = config or TelemetryConfig.from_environ()
        if self._config.enabled and not Telemetry._logged_enabled_notice:
            logger.warning(
                "Using anonymized telemetry. Set "
                "LUXONIS_TELEMETRY_ENABLED=false to disable."
            )
            Telemetry._logged_enabled_notice = True
        self._library_name = library_name
        self._library_version = library_version or _safe_version(library_name)
        self._session_id = str(uuid4())
        if self._config.distinct_id:
            self._distinct_id = self._config.distinct_id
        elif self._config.enabled:
            install_path = (
                self._config.install_id_path or default_install_id_path()
            )
            self._distinct_id = load_install_id(install_path)
        else:
            self._distinct_id = None
        self._base_context = base_context(
            library_name=self._library_name,
            library_version=self._library_version,
            install_id=self._distinct_id,
            session_id=self._session_id,
        )
        self._context_providers: list[ContextProvider] = []
        self._system_context_providers: list[SystemContextProvider] = []
        self.extend_context_providers(context_providers)
        self.extend_system_context_providers(system_context_providers)
        self._backend = self._init_backend()

    @property
    def config(self) -> TelemetryConfig:
        return self._config

    @property
    def library_name(self) -> str:
        return self._library_name

    @property
    def library_version(self) -> str | None:
        return self._library_version

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    def add_context_provider(self, provider: ContextProvider) -> None:
        """Register a context provider for all events."""
        self._add_provider(self._context_providers, provider)

    def add_system_context_provider(
        self, provider: SystemContextProvider
    ) -> None:
        """Register a context provider used only with system
        metadata."""
        self._add_provider(self._system_context_providers, provider)

    def extend_context_providers(
        self, providers: list[ContextProvider] | None
    ) -> None:
        """Register multiple context providers."""
        for provider in providers or []:
            self.add_context_provider(provider)

    def extend_system_context_providers(
        self, providers: list[SystemContextProvider] | None
    ) -> None:
        """Register multiple system context providers."""
        for provider in providers or []:
            self.add_system_context_provider(provider)

    @classmethod
    def register_backend(cls, name: str, factory: BackendFactory) -> None:
        """Register a custom backend factory.

        @type name: str
        @param name: Backend name used in C{TelemetryConfig.backend}.
        @type factory: Callable
        @param factory: Callable that builds a backend from a
            L{TelemetryConfig}.
        """
        cls._backend_factories[name.lower()] = factory

    def capture(
        self,
        event: str,
        properties: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
        allowlist: set[str] | None = None,
        include_system_metadata: bool | None = None,
    ) -> None:
        """Capture a telemetry event.

        @type event: str
        @param event: Event name (e.g. C{"train.start"}).
        @type properties: Optional[dict]
        @param properties: Optional event properties. Values are
            sanitized and redacted before sending.
        @type user_id: Optional[str]
        @param user_id: Optional authenticated user id; overrides
            anonymous id.
        @type allowlist: Optional[set]
        @param allowlist: If set, only these property keys are included.
        @type include_system_metadata: Optional[bool]
        @param include_system_metadata: If True, adds extended system
            metadata for this event. If C{None}, uses the config
            default.
        """
        if not self.is_enabled:
            return
        if is_suppressed():
            return
        try:
            sanitized = sanitize_properties(
                properties,
                allowlist=allowlist or self._config.allowlist,
            )
            context = self._build_context(include_system_metadata)
            payload = TelemetryEvent.create(
                name=event,
                properties=sanitized,
                context=context,
                library=self._library_name,
                library_version=self._library_version,
                distinct_id=self._distinct_id,
                user_id=user_id,
            )
            self._backend.capture(payload)
        except Exception:
            logger.opt(exception=True).debug(
                "Telemetry capture failed for event '{}'; skipping.", event
            )
            return

    def identify(
        self, user_id: str, traits: dict[str, Any] | None = None
    ) -> None:
        """Identify a user with optional traits.

        @type user_id: str
        @param user_id: Stable user identifier.
        @type traits: Optional[dict]
        @param traits: Optional metadata for the user. Values are
            sanitized.
        """
        if not self.is_enabled:
            return
        if is_suppressed():
            return
        sanitized = sanitize_properties(traits)
        try:
            self._backend.identify(user_id, sanitized)
        except Exception:
            logger.opt(exception=True).debug(
                "Telemetry identify failed for user '{}'; skipping.",
                user_id,
            )
            return

    def flush(self) -> None:
        """Flush any buffered telemetry events."""
        if not self.is_enabled:
            return
        try:
            self._backend.flush()
        except Exception:
            logger.opt(exception=True).debug(
                "Telemetry flush failed; continuing."
            )
            return

    def shutdown(self) -> None:
        """Shutdown the backend and flush pending telemetry."""
        if not self.is_enabled:
            return
        try:
            self._backend.shutdown()
        except Exception:
            logger.opt(exception=True).debug(
                "Telemetry shutdown failed; continuing."
            )
            return

    def _init_backend(self) -> TelemetryBackend:
        """Initialize the configured backend or fall back to
        NoopBackend."""
        if not self.is_enabled:
            return NoopBackend()
        name = self._config.backend.lower()
        self._ensure_default_backends()
        factory = self._backend_factories.get(name)
        if factory is None:
            logger.debug(
                "Telemetry backend '{}' is not registered; using noop.",
                name,
            )
            return NoopBackend()
        try:
            return factory(self._config)
        except Exception:
            logger.opt(exception=True).debug(
                "Telemetry backend '{}' failed to initialize; using noop.",
                name,
            )
            return NoopBackend()

    def _build_context(
        self, include_system_metadata: bool | None
    ) -> dict[str, Any]:
        """Build the merged context for an event."""
        context = (
            dict(self._base_context)
            if self._config.include_base_context
            else {}
        )
        if include_system_metadata is None:
            include_system_metadata = self._config.include_system_metadata
        if include_system_metadata:
            context.update(system_context())
            self._merge_context_providers(
                context,
                self._system_context_providers,
                "Telemetry system context provider failed; skipping.",
            )
        self._merge_context_providers(
            context,
            self._context_providers,
            "Telemetry context provider failed; skipping.",
        )
        return context

    def _merge_context_providers(
        self,
        context: dict[str, Any],
        providers: list[ContextProvider | SystemContextProvider],
        error_message: str,
    ) -> None:
        """Merge context from providers into the event context."""
        for provider in providers:
            try:
                extra = provider(self)
            except Exception:
                logger.opt(exception=True).debug(error_message)
                continue
            if extra is None:
                continue
            if not isinstance(extra, Mapping):
                logger.debug(
                    "Telemetry context provider returned a non-mapping; "
                    "skipping."
                )
                continue
            if extra:
                context.update(extra)

    def _add_provider(
        self,
        providers: list[ContextProvider] | list[SystemContextProvider],
        provider: ContextProvider | SystemContextProvider,
    ) -> None:
        """Register a provider once while preserving call order."""
        if provider not in providers:
            providers.append(provider)

    @classmethod
    def _ensure_default_backends(cls) -> None:
        """Register built-in backend factories once."""
        if "noop" not in cls._backend_factories:
            cls.register_backend("noop", lambda _: NoopBackend())
        if "stdout" not in cls._backend_factories:
            cls.register_backend("stdout", lambda _: StdoutBackend())
        if "posthog" not in cls._backend_factories:
            cls.register_backend(
                "posthog",
                lambda cfg: PostHogBackend(
                    api_key=cfg.api_key or "",
                    host=cfg.endpoint,
                ),
            )


def _safe_version(dist_name: str) -> str | None:
    """Best-effort resolution of the package version."""
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None
