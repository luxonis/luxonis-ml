import typer

from luxonis_ml.telemetry.cli.shared import join_command, wrap_command_callback
from luxonis_ml.telemetry.client import Telemetry


def instrument_typer(
    app: typer.Typer,
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
    app: typer.Typer,
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
        name = command.name or callback.__name__
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
