"""MLOps utilities for training models for Luxonis devices.

LuxonisML provides helper functions and utilities used across the Luxonis
MLOps stack.

The package is organized around a few main areas:

- `luxonis_ml.data`: dataset creation, conversion, loading, augmentation, and
  export tools for computer vision workflows.
- `luxonis_ml.tracker`: experiment tracking utilities for PyTorch Lightning
  and LuxonisTrain workflows.
- `luxonis_ml.telemetry`: a lightweight telemetry layer with pluggable
  backends.
- `luxonis_ml.utils`: shared configuration, filesystem, logging, graph, path,
  and registry helpers.

The project is in beta, so APIs may change as the library evolves.
"""

from typing import Final

from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.8.6"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)

import os

from .utils.environ import environ
from .utils.logging import setup_logging

if not environ.LUXONISML_DISABLE_SETUP_LOGGING:
    setup_logging()

if "NO_ALBUMENTATIONS_UPDATE" not in os.environ:
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
