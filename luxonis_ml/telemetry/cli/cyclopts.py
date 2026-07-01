from collections.abc import Iterable

import cyclopts

from luxonis_ml.telemetry.cli.shared import join_command, wrap_command_callback
from luxonis_ml.telemetry.client import Telemetry


def instrument_cyclopts(
    app: cyclopts.App,
    telemetry: Telemetry,
    *,
    allowlist: set[str] | None = None,
    include_system_metadata: bool | None = None,
    exclude_commands: set[str] | None = None,
) -> None:
    """Wrap Cyclopts commands to emit telemetry events."""
    _wrap_cyclopts(
        app,
        telemetry,
        allowlist=allowlist,
        include_system_metadata=include_system_metadata,
        exclude_commands=exclude_commands,
        prefix="",
    )


def _wrap_cyclopts(
    app: cyclopts.App,
    telemetry: Telemetry,
    *,
    allowlist: set[str] | None,
    include_system_metadata: bool | None,
    exclude_commands: set[str] | None,
    prefix: str,
) -> None:
    """Wrap commands/subapps recursively in a Cyclopts app."""
    default_command = app.default_command
    if default_command is not None and not _is_builtin_cyclopts_command(
        app, default_command
    ):
        command_name = prefix or _primary_name(app.name)
        if not exclude_commands or command_name not in exclude_commands:
            app.default_command = wrap_command_callback(
                default_command,
                telemetry,
                command_name,
                allowlist=allowlist,
                include_system_metadata=include_system_metadata,
            )

    for subapp in _iter_unique_subapps(app._commands.values()):
        name = _primary_name(subapp.name)
        if not name or name.startswith("-"):
            continue
        subapp_prefix = join_command(prefix, name)
        _wrap_cyclopts(
            subapp,
            telemetry,
            allowlist=allowlist,
            include_system_metadata=include_system_metadata,
            exclude_commands=exclude_commands,
            prefix=subapp_prefix,
        )


def _iter_unique_subapps(subapps: Iterable[object]) -> list[cyclopts.App]:
    """Deduplicate Cyclopts subapps because aliases share the same app
    object."""
    unique: list[cyclopts.App] = []
    seen: set[int] = set()
    for subapp in subapps:
        if not isinstance(subapp, cyclopts.App):
            continue
        identity = id(subapp)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(subapp)
    return unique


def _primary_name(name: tuple[str, ...] | str | None) -> str:
    """Return the primary command name from a Cyclopts app name
    value."""
    if name is None:
        return ""
    if isinstance(name, tuple):
        return name[0] if name else ""
    return name


def _is_builtin_cyclopts_command(
    app: cyclopts.App,
    default_command: object,
) -> bool:
    """Return True for Cyclopts-generated help/version callbacks."""
    return default_command in {app.help_print, app.version_print}
