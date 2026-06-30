import atexit
import threading
from contextlib import suppress
from dataclasses import dataclass

from loguru import logger

from luxonis_ml.telemetry.client import (
    ContextProvider,
    SystemContextProvider,
    Telemetry,
)
from luxonis_ml.telemetry.config import TelemetryConfig


@dataclass(frozen=True)
class TelemetryKey:
    library_name: str
    source_component: str


_telemetry_by_key: dict[TelemetryKey, Telemetry] = {}
_singleton_state = {"exit_handler_registered": False}
_singleton_lock = threading.RLock()


def get_or_init(
    library_name: str,
    *,
    source_component: str | None = None,
    library_version: str | None = None,
    config: TelemetryConfig | None = None,
    context_providers: list[ContextProvider] | None = None,
    system_context_providers: list[SystemContextProvider] | None = None,
    register_exit_handler: bool = True,
) -> Telemetry:
    """Return an existing telemetry instance or initialize one.

    Args:
        library_name: Name of the library emitting telemetry.
        source_component: Optional emitter/component name for the
            singleton instance. If omitted, the library name is reused.
        library_version: Version string for the library.
        config: Optional telemetry configuration.
        context_providers: Callables that return extra context to
            attach to every event.
        system_context_providers: Callables that return extra context
            to attach only when system metadata is requested.
        register_exit_handler: Whether to flush telemetry on process
            exit.

    Returns:
        The existing or newly initialized telemetry instance for the
        `(library_name, source_component)` key.
    """
    with _singleton_lock:
        existing = get_telemetry(
            library_name,
            source_component=source_component,
        )
        if existing is not None:
            _reconcile_existing_telemetry(
                existing=existing,
                library_name=library_name,
                source_component=source_component,
                library_version=library_version,
                config=config,
                context_providers=context_providers,
                system_context_providers=system_context_providers,
            )
            if register_exit_handler:
                shutdown_on_exit()
            return existing
        return initialize_telemetry(
            library_name=library_name,
            source_component=source_component,
            library_version=library_version,
            config=config,
            context_providers=context_providers,
            system_context_providers=system_context_providers,
            register_exit_handler=register_exit_handler,
        )


def get_telemetry(
    library_name: str | None = None,
    *,
    source_component: str | None = None,
) -> Telemetry | None:
    """Return a telemetry instance by singleton identity.

    Args:
        library_name: Optional library/product name to filter by.
        source_component: Optional emitter/component name to filter by.

    Returns:
        The matching telemetry instance when the lookup resolves to one
        instance. Ambiguous lookups return ``None``.
    """
    with _singleton_lock:
        if library_name is not None and source_component is not None:
            return _telemetry_by_key.get(
                _telemetry_key(
                    library_name=library_name,
                    source_component=source_component,
                )
            )

        matches = [
            telemetry
            for key, telemetry in _telemetry_by_key.items()
            if (library_name is None or key.library_name == library_name)
            and (
                source_component is None
                or key.source_component == source_component
            )
        ]
        if len(matches) == 1:
            return matches[0]
        return None


def initialize_telemetry(
    *,
    library_name: str,
    source_component: str | None = None,
    library_version: str | None = None,
    config: TelemetryConfig | None = None,
    context_providers: list[ContextProvider] | None = None,
    system_context_providers: list[SystemContextProvider] | None = None,
    register_exit_handler: bool = True,
) -> Telemetry:
    """Initialize and return a telemetry instance for one emitter.

    Args:
        library_name: Name of the library emitting telemetry.
        source_component: Optional emitter/component name. If omitted,
            the library name is reused.
        library_version: Optional version string for the library.
        config: Optional telemetry configuration.
        context_providers: Optional list of context provider callables.
        system_context_providers: Optional list of system context
            provider callables.
        register_exit_handler: Whether to flush telemetry on process
            exit.

    Returns:
        The initialized telemetry instance for the exact
        `(library_name, source_component)` key.
    """
    with _singleton_lock:
        key = _telemetry_key(
            library_name=library_name,
            source_component=source_component,
        )
        if key not in _telemetry_by_key:
            _telemetry_by_key[key] = Telemetry(
                library_name,
                source_component=key.source_component,
                library_version=library_version,
                config=config,
                context_providers=context_providers,
                system_context_providers=system_context_providers,
            )
            if register_exit_handler:
                _register_exit_handler()
        return _telemetry_by_key[key]


def shutdown_on_exit() -> None:
    """Register an exit handler to flush telemetry on shutdown."""
    with _singleton_lock:
        _register_exit_handler()


def _flush_on_exit() -> None:
    """Flush all registered telemetry instances."""
    with _singleton_lock:
        telemetry_instances = list(_telemetry_by_key.values())

    for telemetry in telemetry_instances:
        with suppress(Exception):
            telemetry.shutdown()


def _register_exit_handler() -> None:
    """Register the shared singleton flush handler once."""
    if not _singleton_state["exit_handler_registered"]:
        atexit.register(_flush_on_exit)
        _singleton_state["exit_handler_registered"] = True


def _reconcile_existing_telemetry(
    *,
    existing: Telemetry,
    library_name: str,
    source_component: str | None,
    library_version: str | None,
    config: TelemetryConfig | None,
    context_providers: list[ContextProvider] | None,
    system_context_providers: list[SystemContextProvider] | None,
) -> None:
    """Merge additive inputs and warn when immutable args are
    ignored."""
    resolved_source_component = _resolve_source_component(
        library_name=library_name,
        source_component=source_component,
    )
    if config is not None and config != existing.config:
        logger.warning(
            "Telemetry for '{}:{}' is already initialized; the new config "
            "was ignored.",
            library_name,
            resolved_source_component,
        )
    if (
        library_version is not None
        and library_version != existing.library_version
    ):
        logger.warning(
            "Telemetry for '{}:{}' is already initialized; the new library "
            "version was ignored.",
            library_name,
            resolved_source_component,
        )
    existing.extend_context_providers(context_providers)
    existing.extend_system_context_providers(system_context_providers)


def _telemetry_key(
    *, library_name: str, source_component: str | None
) -> TelemetryKey:
    return TelemetryKey(
        library_name=library_name,
        source_component=_resolve_source_component(
            library_name=library_name,
            source_component=source_component,
        ),
    )


def _resolve_source_component(
    *, library_name: str, source_component: str | None
) -> str:
    return source_component or library_name
