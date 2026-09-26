from .base import TRACKER_BACKENDS, RunContext, RunStatus, TrackerBackend
from .mlflow import MLflowBackend
from .tensorboard import TensorBoardBackend
from .wandb import WandbBackend

# The decorator form of `register` would widen each class to the type of
# the registry, and hide the options of its constructor from pyright.
TRACKER_BACKENDS.register(module=TensorBoardBackend, name="tensorboard")
TRACKER_BACKENDS.register(module=WandbBackend, name="wandb")
TRACKER_BACKENDS.register(module=MLflowBackend, name="mlflow")

__all__ = [
    "TRACKER_BACKENDS",
    "MLflowBackend",
    "RunContext",
    "RunStatus",
    "TensorBoardBackend",
    "TrackerBackend",
    "WandbBackend",
]
