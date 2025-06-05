__version__ = "0.7.0"

import os

from .utils.environ import environ
from .utils.logging import setup_logging

if not environ.LUXONISML_DISABLE_SETUP_LOGGING:
    setup_logging()

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
