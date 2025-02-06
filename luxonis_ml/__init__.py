__version__ = "0.6.0"

from contextlib import suppress

with suppress(ImportError):
    from .utils.environ import environ
    from .utils.logging import setup_logging

    if not environ.LUXONISML_DISABLE_SETUP_LOGGING:
        setup_logging()
