"""CLI instrumentation adapters for telemetry.

This package wraps the commands of a command-line application. An
instrumented command emits one ``cli_command`` event per run, if
telemetry stays enabled. Use `instrument_typer` for a Typer app and
`instrument_cyclopts` for a Cyclopts app. Both walk the app recursively,
so they cover the commands of every sub-app. The Cyclopts adapter skips
the generated help and version callbacks. ``exclude_commands`` and
`skip_telemetry` leave more commands uninstrumented.

By default, an instrumented command reports its name, a success flag,
and its duration in milliseconds. ``include_system_metadata`` adds the
system fields, but only when the client holds a system context provider.
Pass ``system_context_providers=[system_context_provider]`` to `Telemetry`
first. Argument values stay local unless the caller names them in an
``allowlist``.

Example:
    Instrument a Typer app and keep every argument value local.

    .. code-block:: python

        import typer
        from luxonis_ml.telemetry import Telemetry
        from luxonis_ml.telemetry.cli import instrument_typer

        app = typer.Typer()
        telemetry = Telemetry("luxonis_ml")

        instrument_typer(app, telemetry)

Note:
    This package imports each adapter module on first use, so a Typer
    application never loads the Cyclopts adapter. Cyclopts is a base
    dependency of LuxonisML. Typer is not, so install it yourself before
    you call `instrument_typer`.

See:
    `skip_telemetry` to exclude one command callback.

"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from luxonis_ml.telemetry.cli.cyclopts import instrument_cyclopts
    from luxonis_ml.telemetry.cli.shared import skip_telemetry
    from luxonis_ml.telemetry.cli.typer import instrument_typer

__all__ = [
    "instrument_cyclopts",
    "instrument_typer",
    "skip_telemetry",
]


def __getattr__(name: str) -> Any:
    """Lazily import optional CLI adapters so their deps stay
    optional.
    """
    if name == "instrument_typer":
        return import_module("luxonis_ml.telemetry.cli.typer").instrument_typer
    if name == "instrument_cyclopts":
        return import_module(
            "luxonis_ml.telemetry.cli.cyclopts"
        ).instrument_cyclopts
    if name == "skip_telemetry":
        return import_module("luxonis_ml.telemetry.cli.shared").skip_telemetry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
