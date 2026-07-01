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
    optional."""
    if name == "instrument_typer":
        return import_module("luxonis_ml.telemetry.cli.typer").instrument_typer
    if name == "instrument_cyclopts":
        return import_module(
            "luxonis_ml.telemetry.cli.cyclopts"
        ).instrument_cyclopts
    if name == "skip_telemetry":
        return import_module("luxonis_ml.telemetry.cli.shared").skip_telemetry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
