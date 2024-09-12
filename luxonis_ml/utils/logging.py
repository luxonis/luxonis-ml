import builtins
import logging
import warnings
from functools import wraps
from typing import Dict, Optional, Type

from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from .environ import environ


def setup_logging(
    *,
    file: Optional[str] = None,
    use_rich: bool = False,
    level: Optional[str] = None,
    configure_warnings: bool = True,
    rich_print: bool = False,
    **kwargs,
) -> None:
    """Globally configures logging.

    Configures a standar Luxonis logger. Optionally utilizes rich library, configures
    handling of warnings (from warnings module) and saves the logs to a file.

    @type file: str or None
    @param file: Path to a file where logs will be saved. If None, logs will not be
        saved. Defaults to None.
    @type use_rich: bool
    @param use_rich: If True, rich library will be used for logging. Defaults to False.
    @type level: str or None
    @param level: Logging level. One of "DEBUG", "INFO", "WARNING", "ERROR", and
        "CRITICAL". Defaults to "INFO". The log level can be changed using "LOG_LEVEL"
        environment variable.
    @type configure_warnings: bool
    @param configure_warnings: If True, warnings will be logged. Defaults to True.
    @type rich_print: bool
    @param rich_print: If True, builtins.print will be replaced with rich.print.
        Defaults to False.
    @param kwargs: Additional arguments passed to RichHandler.
    """

    if rich_print:
        builtins.print = rprint

    # NOTE: So we can simply run e.g. `LOG_LEVEL=DEBUG python ...`
    level = level or environ.LOG_LEVEL

    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(
            f"Invalid logging level: {level}. "
            "Use one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'."
        )

    handlers = []

    format = file_format = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    if use_rich:
        format = "%(message)s"
        # NOTE: The default rich logging colors are weird.
        theme = Theme(
            {
                "logging.level.debug": "magenta",
                "logging.level.info": "green",
                "logging.level.warning": "yellow",
            }
        )
        console = Console(theme=theme)
        handlers.append(
            RichHandler(
                console=console,
                **{
                    **{
                        "rich_tracebacks": True,
                        "tracebacks_show_locals": False,
                        "omit_repeated_times": False,
                        "show_time": False,
                    },
                    **kwargs,
                },
            )
        )
    else:
        handlers.append(logging.StreamHandler())

    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(logging.Formatter(file_format, datefmt=datefmt))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=format, datefmt=datefmt, handlers=handlers)

    def _custom_warning_handler(
        message: str,
        category: Type[Warning],
        filename: str,
        lineno: int,
        _=None,
        line: Optional[str] = None,
    ):
        logger = logging.getLogger(filename)
        message = warnings.formatwarning(
            message,
            category,
            filename,
            lineno,
            line,
        )
        logger.warning(message)

    if configure_warnings:
        warnings.showwarning = _custom_warning_handler


def reset_logging() -> None:
    """Resets the logging module back to its default state."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def deprecated(
    *args: str,
    suggested: Optional[Dict[str, str]] = None,
    additional_message: Optional[str] = None,
    altogether: bool = False,
):
    """Decorator to mark a function or its parameters as deprecated.

    Example:

        >>> @deprecated("old_arg",
        ...             "another_old_arg",
        ...             suggested={"old_arg": "new_arg"},
        ...             additional_message="Usage of 'old_arg' is discouraged.")
        )
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
    @type suggested: Dict[str, str]
    @param suggested: Suggested replacement parameters.
    @type additional_message: str
    @param additional_message: Additional message to display.
        If provided, it will be appended to the warning message.
    @type altogether: bool
    @param altogether: If True, the whole function is
        marked as deprecated. Defaults to False.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*f_args, **f_kwargs):
            fname = func.__name__
            if altogether:
                msg = f"'{fname}' is deprecated and will be removed in future versions."
                if additional_message:
                    msg += f" {additional_message}"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)

            if args:
                for arg in args:
                    if arg in f_kwargs:
                        replacement = suggested.get(arg) if suggested else None
                        msg = (
                            f"Argument '{arg}' in function '{fname}' "
                            "is deprecated and will be removed in "
                            "future versions."
                        )
                        if replacement:
                            msg += f" Use '{replacement}' instead."
                        if additional_message:
                            msg += f" {additional_message}"
                        warnings.warn(msg, DeprecationWarning, stacklevel=2)

            return func(*f_args, **f_kwargs)

        return wrapper

    return decorator
