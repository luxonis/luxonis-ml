import inspect
import sys
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from luxonis_ml.typing import PathType

from .environ import environ


def setup_logging(
    *,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    | None = None,
    file: PathType | None = None,
    use_rich: bool = True,
    **kwargs,
) -> None:  # pragma: no cover
    """Sets up global logging using loguru and rich.

    @type level: Optional[str]
    @param level: Logging level. If not set, reads from the environment
        variable C{LOG_LEVEL}. Defaults to "INFO".
    @type file: Optional[str]
    @param file: Path to the log file. If provided, logs will be saved
        to this file.
    @type use_rich: bool
    @param use_rich: If True, uses rich for logging. Defaults to True.
    @type kwargs: Any
    @param kwargs: Additional keyword arguments to pass to
        C{RichHandler}.
    """
    from loguru import logger

    level = level or environ.LOG_LEVEL
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            f"Invalid logging level: {level}. "
            "Use one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'."
        )

    logger.remove()
    if use_rich:
        theme = Theme(
            {
                "logging.level.debug": "magenta",
                "logging.level.info": "green",
                "logging.level.warning": "yellow",
                "logging.level.error": "bold red",
                "logging.level.critical": "bold white on red",
            },
            inherit=True,
        )
        console = Console(theme=theme)
        logger.add(
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                show_time=False,
                **kwargs,
            ),
            level=level,
            # NOTE: Needs to be a constant function to avoid
            # duplicate logging of exceptions, see
            # https://github.com/Delgan/loguru/issues/1172
            format=lambda _: "{message}",
            backtrace=False,
            filter=lambda record: "file_only" not in record["extra"],
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format="[{level}] {message} | {function}:{line}",
            filter=lambda record: "file_only" not in record["extra"],
        )

    if file is not None:
        logger.add(
            file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {message} | {function}:{line}",
            rotation=None,
        )

    def _custom_warning_handler(
        message: str,
        category: type[Warning],
        filename: str,
        lineno: int,
        _file: str | None = None,
        line: str | None = None,
    ) -> None:
        text = warnings.formatwarning(
            message, category, filename, lineno, line
        )
        logger.warning(text)

    warnings.showwarning = _custom_warning_handler


def deprecated(
    *args: str,
    suggest: dict[str, str] | None = None,
    additional_message: str | None = None,
    altogether: bool = False,
) -> Callable[[Callable], Callable]:
    """Decorator to mark a function or its parameters as deprecated.

    Example:

        >>> @deprecated("old_arg",
        ...             "another_old_arg",
        ...             suggest={"old_arg": "new_arg"},
        ...             additional_message="Usage of 'old_arg' is discouraged.")
        ...def my_func(old_arg, another_old_arg, new_arg=None):
        ...   pass
        >>> my_func("foo")
        >>> # DeprecationWarning: Argument 'old_arg'
        ... # in function `my_func` is deprecated and
        ... # will be removed in future versions.
        ... # Use 'new_arg' instead.
        ... # Usage of 'old_arg' is discouraged.

    @type args: str
    @param args: The names of the deprecated parameters.
    @type suggest: Dict[str, str]
    @param suggest: Suggested replacement parameters.
    @type additional_message: str
    @param additional_message: Additional message to display.
        If provided, it will be appended to the warning message.
    @type altogether: bool
    @param altogether: If True, the whole function is
        marked as deprecated. Defaults to False.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*f_args, **f_kwargs) -> Any:
            fname = func.__name__
            if altogether:
                msg = f"'{fname}' is deprecated and will be removed in future versions."
                if additional_message:
                    msg += f" {additional_message}"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)

            if args:
                pos_arg_names = list(sig.parameters.keys())[: len(f_args)]

                for arg_name in pos_arg_names:
                    if arg_name in args:
                        _warn_deprecated(
                            arg_name, fname, suggest, additional_message
                        )

                for arg_name in f_kwargs:
                    if arg_name in args:
                        _warn_deprecated(
                            arg_name, fname, suggest, additional_message
                        )

            return func(*f_args, **f_kwargs)

        return wrapper

    return decorator


def log_once(logger: Callable[[str], None], message: str) -> None:
    """Logs a message only once.

    @type logger: Logger
    @param logger: The logger to use.
    @type message: str
    @param message: The message to log.
    """
    _cache: set[str]

    if not hasattr(log_once, "_cache"):
        log_once._cache = set()

    if message not in log_once._cache:
        logger(message)
        log_once._cache.add(message)


def _warn_deprecated(
    arg_name: str,
    fname: str,
    suggest: dict[str, str] | None,
    additional_message: str | None,
) -> None:
    replacement = suggest.get(arg_name) if suggest else None
    msg = (
        f"Argument '{arg_name}' in function '{fname}' "
        "is deprecated and will be removed in future versions."
    )
    if replacement:
        msg += f" Use '{replacement}' instead."
    if additional_message:
        msg += f" {additional_message}"
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
