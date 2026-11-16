"""The tracker backends.

Each backend connects `LuxonisTracker` to one tracking service, and
each has a keyword argument of `LuxonisTracker` that turns it on:

    - `TensorBoardBackend`, ``tensorboard``, writes TensorBoard event
      files;
    - `WandbBackend`, ``wandb``, logs to Weights & Biases, with the
      options of `WandbOptions`;
    - `MLflowBackend`, ``mlflow``, logs to an MLflow tracking server,
      with the options of `MLflowOptions`.

`TrackerBackend` is the base class of each backend, and
`TRACKER_BACKENDS` is the registry that the tracker looks the keyword
arguments up in. `RunContext` describes the run that a backend logs to.

See:
    `luxonis_ml.tracker` for what each backend does with each logging
    call, and for a complete example of a custom backend.

"""

from .base import (
    TRACKER_BACKENDS,
    RunContext,
    RunStatus,
    TrackerBackend,
    check_options,
)
from .mlflow import MLflowBackend, MLflowOptions
from .tensorboard import TensorBoardBackend
from .wandb import WandbBackend, WandbOptions

__all__ = [
    "TRACKER_BACKENDS",
    "MLflowBackend",
    "MLflowOptions",
    "RunContext",
    "RunStatus",
    "TensorBoardBackend",
    "TrackerBackend",
    "WandbBackend",
    "WandbOptions",
    "check_options",
]
