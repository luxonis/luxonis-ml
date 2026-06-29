import atexit
from contextlib import suppress

from loguru import logger

from luxonis_ml.telemetry.client import (
    ContextProvider,
    SystemContextProvider,
    Telemetry,
)
from luxonis_ml.telemetry.config import TelemetryConfig

_telemetry_by_name: dict[str, Telemetry] = {}
_singleton_state = {"exit_handler_registered": False}


def get_or_init(
    library_name: str,
    *,
    library_version: str | None = None,
    config: TelemetryConfig | None = None,
    context_providers: list[ContextProvider] | None = None,
    system_context_providers: list[SystemContextProvider] | None = None,
    register_exit_handler: bool = True,
) -> Telemetry:
    """Return an existing telemetry instance or initialize one.

    Args:
        library_name: Name of the library emitting telemetry.
        library_version: Version string for the library.
        config: Optional telemetry configuration.
        context_providers: Callables that return extra context to
            attach to every event.
        system_context_providers: Callables that return extra context
            to attach only when system metadata is requested.
        register_exit_handler: Whether to flush telemetry on process
            exit.

    Returns:
        The existing or newly initialized telemetry instance.
    """
    existing = get_telemetry(library_name)
    if existing is not None:
        _reconcile_existing_telemetry(
            existing=existing,
            library_name=library_name,
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
        library_version=library_version,
        config=config,
        context_providers=context_providers,
        system_context_providers=system_context_providers,
        register_exit_handler=register_exit_handler,
    )


def get_telemetry(library_name: str | None = None) -> Telemetry | None:
    """Return a telemetry instance by library name, if initialized.

    Args:
        library_name: Name used to initialize the telemetry instance. If
            omitted and only one instance exists, that instance is
            returned.

    Returns:
        The matching telemetry instance, if available.
    """
    if library_name is None:
        if len(_telemetry_by_name) == 1:
            return next(iter(_telemetry_by_name.values()))
        return None
    return _telemetry_by_name.get(library_name)


def initialize_telemetry(
    *,
    library_name: str,
    library_version: str | None = None,
    config: TelemetryConfig | None = None,
    context_providers: list[ContextProvider] | None = None,
    system_context_providers: list[SystemContextProvider] | None = None,
    register_exit_handler: bool = True,
) -> Telemetry:
    """Initialize and return a telemetry instance for a library.

    Args:
        library_name: Name of the library emitting telemetry.
        library_version: Optional version string for the library.
        config: Optional telemetry configuration.
        context_providers: Optional list of context provider callables.
        system_context_providers: Optional list of system context
            provider callables.
        register_exit_handler: Whether to flush telemetry on process
            exit.

    Returns:
        The initialized telemetry instance.
    """
    if library_name not in _telemetry_by_name:
        _telemetry_by_name[library_name] = Telemetry(
            library_name,
            library_version=library_version,
            config=config,
            context_providers=context_providers,
            system_context_providers=system_context_providers,
        )
        if (
            register_exit_handler
            and not _singleton_state["exit_handler_registered"]
        ):
            atexit.register(_flush_on_exit)
            _singleton_state["exit_handler_registered"] = True
    return _telemetry_by_name[library_name]


def shutdown_on_exit() -> None:
    """Register an exit handler to flush telemetry on shutdown."""
    if not _singleton_state["exit_handler_registered"]:
        atexit.register(_flush_on_exit)
        _singleton_state["exit_handler_registered"] = True


def _flush_on_exit() -> None:
    """Flush all registered telemetry instances."""
    for telemetry in list(_telemetry_by_name.values()):
        with suppress(Exception):
            telemetry.shutdown()


def _reconcile_existing_telemetry(
    *,
    existing: Telemetry,
    library_name: str,
    library_version: str | None,
    config: TelemetryConfig | None,
    context_providers: list[ContextProvider] | None,
    system_context_providers: list[SystemContextProvider] | None,
) -> None:
    """Merge additive inputs and warn when immutable args are
    ignored."""
    if config is not None and config != existing.config:
        logger.warning(
            "Telemetry for '{}' is already initialized; the new config was ignored.",
            library_name,
        )
    if (
        library_version is not None
        and library_version != existing.library_version
    ):
        logger.warning(
            "Telemetry for '{}' is already initialized; the new library "
            "version was ignored.",
            library_name,
        )
    existing.extend_context_providers(context_providers)
    existing.extend_system_context_providers(system_context_providers)
