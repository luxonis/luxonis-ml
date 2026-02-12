from __future__ import annotations

import atexit
from contextlib import suppress

from luxonis_ml.telemetry.client import ContextProvider, Telemetry
from luxonis_ml.telemetry.config import TelemetryConfig

_telemetry_by_name: dict[str, Telemetry] = {}
_singleton_state = {"exit_handler_registered": False}


def get_telemetry(library_name: str | None = None) -> Telemetry | None:
    """Return a telemetry instance by library name, if initialized.

    @type library_name: Optional[str]
    @param library_name: Name used to initialize the telemetry instance.
        If C{None} and only one instance exists, that instance is
        returned.
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
    register_exit_handler: bool = True,
) -> Telemetry:
    """Initialize and return a telemetry instance for a library.

    @type library_name: str
    @param library_name: Name of the library emitting telemetry.
    @type library_version: Optional[str]
    @param library_version: Optional version string for the library.
    @type config: Optional[L{TelemetryConfig}]
    @param config: Optional TelemetryConfig to use.
    @type context_providers: Optional[list]
    @param context_providers: Optional list of context provider
        callables.
    @type register_exit_handler: bool
    @param register_exit_handler: If True, flush telemetry on process
        exit.
    """
    if library_name not in _telemetry_by_name:
        _telemetry_by_name[library_name] = Telemetry(
            library_name,
            library_version=library_version,
            config=config,
            context_providers=context_providers,
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
