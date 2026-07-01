import inspect
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from functools import wraps
from typing import Any, Protocol, cast

from luxonis_ml.telemetry.client import Telemetry


class _TelemetryCallable(Protocol):
    _telemetry_skip: bool
    _telemetry_wrapped: bool

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def wrap_command_callback(
    func: Callable[..., Any],
    telemetry: Telemetry,
    command_name: str,
    *,
    allowlist: set[str] | None,
    include_system_metadata: bool | None,
) -> Callable[..., Any]:
    """Wrap a command callback to emit telemetry for execution."""
    telemetry_func = cast(_TelemetryCallable, func)

    if getattr(telemetry_func, "_telemetry_skip", False):
        return func
    if getattr(telemetry_func, "_telemetry_wrapped", False):
        return func

    signature = inspect.signature(func)

    def emit_command_event(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        success: bool,
        start: float,
    ) -> None:
        duration_ms = int((time.monotonic() - start) * 1000)
        properties = {
            "command": command_name,
            "success": success,
            "duration_ms": duration_ms,
        }
        properties.update(
            extract_params(
                signature,
                args,
                kwargs,
                allowlist=allowlist,
            )
        )
        with suppress(Exception):
            telemetry.capture(
                "cli_command",
                properties,
                include_system_metadata=include_system_metadata,
            )

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            success = True
            try:
                return await func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                emit_command_event(
                    args,
                    kwargs,
                    success=success,
                    start=start,
                )

        cast(_TelemetryCallable, async_wrapper)._telemetry_wrapped = True
        return async_wrapper

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
            emit_command_event(
                args,
                kwargs,
                success=success,
                start=start,
            )

    cast(_TelemetryCallable, wrapper)._telemetry_wrapped = True
    return wrapper


def skip_telemetry(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a command callback as excluded from telemetry."""
    cast(_TelemetryCallable, func)._telemetry_skip = True
    return func


def extract_params(
    signature: inspect.Signature,
    args: Iterable[Any],
    kwargs: dict[str, Any],
    *,
    allowlist: set[str] | None,
) -> dict[str, Any]:
    """Extract call arguments into a parameter map."""
    if allowlist is None:
        return {}

    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return {}

    output: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name not in allowlist:
            continue
        if name in {"self", "ctx", "context"}:
            continue
        if is_click_context(value):
            continue
        output[name] = value
    return output


def is_click_context(value: Any) -> bool:
    """Return True if value looks like a Click/Typer context object."""
    return hasattr(value, "info_name") and hasattr(value, "command")


def join_command(prefix: str, name: str) -> str:
    """Join nested command names into a single string."""
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix} {name}"
