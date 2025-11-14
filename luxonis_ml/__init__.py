from typing import Final

from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.8.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)

import os

from .utils.environ import environ
from .utils.logging import setup_logging

if not environ.LUXONISML_DISABLE_SETUP_LOGGING:
    setup_logging()

if "NO_ALBUMENTATIONS_UPDATE" not in os.environ:
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
