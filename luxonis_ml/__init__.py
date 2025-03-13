__version__ = "0.6.3"

from .utils.environ import environ
from .utils.logging import setup_logging

if not environ.LUXONISML_DISABLE_SETUP_LOGGING:
    setup_logging()
