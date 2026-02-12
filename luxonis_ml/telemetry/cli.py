from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from functools import wraps
from typing import Any

import typer

from luxonis_ml.telemetry.client import Telemetry


def instrument_typer(
    app: typer.Typer,
    telemetry: Telemetry,
    *,
    allowlist: set[str] | None = None,
    include_system_metadata: bool | None = None,
) -> None:
    """Wrap Typer commands to emit telemetry events.

    @type app: L{typer.Typer}
    @param app: Typer application to instrument.
    @type telemetry: L{Telemetry}
    @param telemetry: Telemetry instance to emit events through.
    @type allowlist: Optional[set]
    @param allowlist: If set, only these params are logged.
    @type include_system_metadata: Optional[bool]
    @param include_system_metadata: If True, adds extended system
        metadata.
    """
    _wrap_typer(
        app,
        telemetry,
        allowlist=allowlist,
        include_system_metadata=include_system_metadata,
        prefix="",
    )


def _wrap_typer(
    app: typer.Typer,
    telemetry: Telemetry,
    *,
    allowlist: set[str] | None,
    include_system_metadata: bool | None,
    prefix: str,
) -> None:
    """Wrap commands/groups recursively in a Typer app."""
    for command in list(app.registered_commands or []):
        callback = command.callback
        if callback is None:
            continue
        name = command.name or callback.__name__
        full_name = _join_command(prefix, name)
        command.callback = _wrap_callback(
            callback,
            telemetry,
            full_name,
            allowlist=allowlist,
            include_system_metadata=include_system_metadata,
        )

    for group in list(app.registered_groups or []):
        group_prefix = _join_command(prefix, group.name or "")
        if group.typer_instance is None:
            continue
        _wrap_typer(
            group.typer_instance,
            telemetry,
            allowlist=allowlist,
            include_system_metadata=include_system_metadata,
            prefix=group_prefix,
        )


def _wrap_callback(
    func: Callable[..., Any],
    telemetry: Telemetry,
    command_name: str,
    *,
    allowlist: set[str] | None,
    include_system_metadata: bool | None,
) -> Callable[..., Any]:
    """Wrap a command callback to emit telemetry for execution."""
    if getattr(func, "_telemetry_wrapped", False):
        return func

    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        success = True
        try:
            return func(*args, **kwargs)
        except Exception:
            success = False
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            properties = {
                "command": command_name,
                "success": success,
                "duration_ms": duration_ms,
            }
            properties.update(_extract_params(signature, args, kwargs))
            with suppress(Exception):
                telemetry.capture(
                    "cli.command",
                    properties,
                    allowlist=allowlist,
                    include_system_metadata=include_system_metadata,
                )

    wrapper._telemetry_wrapped = True
    return wrapper


def _extract_params(
    signature: inspect.Signature,
    args: Iterable[Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Extract call arguments into a parameter map."""
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return {}

    output: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name in {"self", "ctx", "context"}:
            continue
        if _is_click_context(value):
            continue
        output[name] = value
    return output


def _is_click_context(value: Any) -> bool:
    """Return True if value looks like a Click/Typer context object."""
    return hasattr(value, "info_name") and hasattr(value, "command")


def _join_command(prefix: str, name: str) -> str:
    """Join nested command names into a single string."""
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix} {name}"
