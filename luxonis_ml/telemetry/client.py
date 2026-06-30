from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

from loguru import logger

from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.backends.noop import NoopBackend
from luxonis_ml.telemetry.config import TelemetryConfig
from luxonis_ml.telemetry.context import base_context
from luxonis_ml.telemetry.events import TelemetryEvent
from luxonis_ml.telemetry.redaction import sanitize_properties
from luxonis_ml.telemetry.suppression import is_suppressed

ContextProvider = Callable[["Telemetry"], dict[str, Any]]
SystemContextProvider = Callable[["Telemetry"], dict[str, Any]]


class Telemetry:
    """Telemetry client for emitting events via pluggable backends."""

    _logged_enabled_notice: bool = False

    def __init__(
        self,
        library_name: str,
        *,
        source_component: str | None = None,
        library_version: str | None = None,
        config: TelemetryConfig | None = None,
        context_providers: list[ContextProvider] | None = None,
        system_context_providers: list[SystemContextProvider] | None = None,
    ) -> None:
        """Initialize a telemetry client.

        Args:
            library_name: Name of the library emitting telemetry, such
                as ``"luxonis_ml"``.
            source_component: Optional emitter/component name for the
                event context. If omitted, the library name is reused.
            library_version: Version string for the library. If omitted,
                it is resolved from package metadata when possible.
            config: Telemetry configuration to use. If omitted, values
                are read from environment variables.
            context_providers: Callables that return extra context to
                attach to every event.
            system_context_providers: Callables that return extra
                context to attach only when system metadata is
                requested.
        """
        self._config = config or TelemetryConfig.from_environ()
        if self._config.enabled and not Telemetry._logged_enabled_notice:
            logger.warning(
                "Using anonymized telemetry. Set "
                "LUXONIS_TELEMETRY_ENABLED=false to disable."
            )
            Telemetry._logged_enabled_notice = True
        self._library_name = library_name
        self._source_component = source_component or library_name
        self._library_version = library_version or _safe_version(library_name)
        self._session_id = str(uuid4())
        self._base_context = base_context(
            library_name=self._library_name,
            library_version=self._library_version,
            session_id=self._session_id,
            source_component=self._source_component,
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
    def source_component(self) -> str:
        return self._source_component

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
    def register_backend(
        cls,
        name: str,
        backend_cls: type[TelemetryBackend],
    ) -> None:
        """Register a custom backend class.

        Args:
            name: Backend name used in ``TelemetryConfig.backend``.
            backend_cls: Backend class instantiated with
                ``TelemetryConfig``.
        """
        TELEMETRY_BACKENDS.register(
            name=name.lower(),
            module=backend_cls,
            force=True,
        )

    def capture(
        self,
        event: str,
        properties: dict[str, Any] | None = None,
        *,
        allowlist: set[str] | None = None,
        include_system_metadata: bool | None = None,
        distinct_id: str | None = None,
    ) -> None:
        """Capture a telemetry event.

        Args:
            event: Event name, such as ``"train_started"``.
            properties: Optional event properties. Values are sanitized
                and redacted before sending.
            allowlist: If set, only these property keys are included.
            include_system_metadata: Whether to add extended system
                metadata for this event. If omitted, the config default
                is used.
            distinct_id: Optional backend identity override. When
                omitted, backends may fall back to `$session_id`.
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
                distinct_id=distinct_id,
            )
            self._backend.capture(payload)
        except Exception as exc:
            logger.debug(
                "Telemetry capture failed for event '{}': {}",
                event,
                type(exc).__name__,
            )
            return

    def flush(self) -> None:
        """Flush any buffered telemetry events."""
        if not self.is_enabled:
            return
        try:
            self._backend.flush()
        except Exception as exc:
            logger.debug(
                "Telemetry flush failed: {}",
                type(exc).__name__,
            )
            return

    def shutdown(self) -> None:
        """Shutdown the backend and flush pending telemetry."""
        if not self.is_enabled:
            return
        try:
            self._backend.shutdown()
        except Exception as exc:
            logger.debug(
                "Telemetry shutdown failed: {}",
                type(exc).__name__,
            )
            return

    def _init_backend(self) -> TelemetryBackend:
        """Initialize the configured backend or fall back to
        NoopBackend."""
        if not self.is_enabled:
            return NoopBackend(self._config)
        name = self._config.backend.lower()
        try:
            backend_cls = TELEMETRY_BACKENDS.get(name)
        except KeyError:
            logger.debug(
                "Telemetry backend '{}' is not registered; using noop.",
                name,
            )
            return NoopBackend(self._config)
        try:
            return backend_cls(self._config)
        except Exception:
            logger.debug(
                "Telemetry backend '{}' failed to initialize; using noop.",
                name,
            )
            return NoopBackend(self._config)

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
            except Exception as exc:
                logger.debug("{} ({})", error_message, type(exc).__name__)
                continue
            if extra is None:
                continue
            if not isinstance(extra, Mapping):
                logger.debug(
                    "Telemetry context provider returned a non-mapping; skipping."
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


def _safe_version(dist_name: str) -> str | None:
    """Best-effort resolution of the package version."""
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None
