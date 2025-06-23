__version__ = "0.7.1"

import os

from .utils.environ import environ
from .utils.logging import setup_logging

if not environ.LUXONISML_DISABLE_SETUP_LOGGING:
    setup_logging()

if "NO_ALBUMENTATIONS_UPDATE" not in os.environ:
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
