"""MLOps utilities for training models for Luxonis devices.

LuxonisML provides helper functions and utilities used across the Luxonis
MLOps stack.

The package is organized around a few main areas:

- `luxonis_ml.data`: dataset creation, conversion, loading, augmentation, and
  export tools for computer vision workflows.
- `luxonis_ml.ldf`: the annotation schemas of the Luxonis Data Format. The
  ``ldf`` extra is a subset of the ``data`` extra.
- `luxonis_ml.tracker`: experiment tracking utilities for PyTorch Lightning
  and LuxonisTrain workflows.
- `luxonis_ml.telemetry`: a lightweight telemetry layer with pluggable
  backends.
- `luxonis_ml.nn_archive`: NN Archive creation and inspection.
- `luxonis_ml.utils`: shared configuration, filesystem, logging, graph, path,
  and registry helpers.

Each area installs through its own extra, such as ``luxonis-ml[data]``. The
``all`` extra installs every module and every cloud integration. The ``data``,
``ldf``, ``tracker``, and ``utils`` modules raise ``ImportError`` when their
extra is absent, and the message names the extra to install.
`luxonis_ml.telemetry` is an exception: it imports without ``posthog`` and
falls back to a no-op backend.

The package also installs a ``luxonis_ml`` command. It provides the ``data``,
``archive``, and ``fs`` sub-applications, and a ``checkhealth`` command that
reports whether the ``ldf``, ``data``, ``utils``, and ``nn_archive`` modules
import correctly.

The project is in beta, so APIs may change as the library evolves.
"""

from typing import Final

from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.9.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)

import os

from .utils.environ import environ
from .utils.logging import setup_logging

if not environ.LUXONISML_DISABLE_SETUP_LOGGING:
    setup_logging()

if "NO_ALBUMENTATIONS_UPDATE" not in os.environ:
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
