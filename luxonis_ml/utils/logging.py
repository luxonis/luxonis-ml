import logging
import warnings
from typing import Optional

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
    """
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
        _theme = Theme(
            {
                "logging.level.debug": "magenta",
                "logging.level.info": "green",
                "logging.level.warning": "yellow",
            }
        )
        _console = Console(theme=_theme)
        handlers.append(
            RichHandler(
                rich_tracebacks=True, console=_console, omit_repeated_times=False
            )
        )
    else:
        handlers.append(logging.StreamHandler())

    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(logging.Formatter(file_format, datefmt=datefmt))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=format, datefmt=datefmt, handlers=handlers)

    if configure_warnings:
        logger = logging.getLogger("warnings.warn")

        def custom_warning_handler(message, *_):
            logger.warning(message)

        warnings.showwarning = custom_warning_handler


def reset_logging() -> None:
    """Resets the logging module back to its default state."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
