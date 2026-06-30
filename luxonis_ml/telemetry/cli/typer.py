from typing import Any

from luxonis_ml.telemetry.cli.shared import join_command, wrap_command_callback
from luxonis_ml.telemetry.client import Telemetry


def instrument_typer(
    app: Any,
    telemetry: Telemetry,
    *,
    allowlist: set[str] | None = None,
    include_system_metadata: bool | None = None,
    exclude_commands: set[str] | None = None,
) -> None:
    """Wrap Typer commands to emit telemetry events."""
    _wrap_typer(
        app,
        telemetry,
        allowlist=allowlist,
        include_system_metadata=include_system_metadata,
        exclude_commands=exclude_commands,
        prefix="",
    )


def _wrap_typer(
    app: Any,
    telemetry: Telemetry,
    *,
    allowlist: set[str] | None,
    include_system_metadata: bool | None,
    exclude_commands: set[str] | None,
    prefix: str,
) -> None:
    """Wrap commands/groups recursively in a Typer app."""
    for command in list(app.registered_commands or []):
        callback = command.callback
        if callback is None:
            continue
        name = _resolve_command_name(command.name, callback)
        full_name = join_command(prefix, name)
        if exclude_commands and full_name in exclude_commands:
            continue
        command.callback = wrap_command_callback(
            callback,
            telemetry,
            full_name,
            allowlist=allowlist,
            include_system_metadata=include_system_metadata,
        )

    for group in list(app.registered_groups or []):
        group_prefix = join_command(prefix, group.name or "")
        if group.typer_instance is None:
            continue
        _wrap_typer(
            group.typer_instance,
            telemetry,
            allowlist=allowlist,
            include_system_metadata=include_system_metadata,
            exclude_commands=exclude_commands,
            prefix=group_prefix,
        )


def _resolve_command_name(command_name: str | None, callback: Any) -> str:
    """Resolve the CLI-visible Typer command name."""
    if command_name is not None:
        return command_name

    from typer.main import get_command_name

    return get_command_name(callback.__name__)
