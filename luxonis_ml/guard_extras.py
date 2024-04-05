import logging
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def guard_missing_extra(name: str):
    try:
        yield
    except ImportError:
        logger.exception(
            f"Error importing `luxonis-ml.{name}`. This can mean that some of the dependencies of `luxonis-ml[{name}]` are not installed. "
            f"Ensure you installed the package with the `[{name}]` or `[all]` extra specified. "
            f"Use `pip install luxonis-ml[{name}]` to install dependencies for the `{name}` submodule."
        )
        sys.exit(1)


__all__ = ["guard_missing_extra"]
