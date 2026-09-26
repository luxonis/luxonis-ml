r"""Experiment tracking for Luxonis ML workflows.

`LuxonisTracker` logs a run to several tracking services at once.
Training and evaluation code log metrics, hyperparameters, images,
matrices, and artifacts through one API, and choose the services at
runtime. Each service is a `TrackerBackend` in `TRACKER_BACKENDS`:

.. list-table:: Built-in backends
   :header-rows: 1

   * - Name
     - Backend
     - Extra
     - Options
   * - ``"tensorboard"``
     - `TensorBoardBackend`
     - ``tensorboard``
     - None.
   * - ``"wandb"``
     - `WandbBackend`
     - ``wandb``
     - ``entity``
   * - ``"mlflow"``
     - `MLflowBackend`
     - ``mlflow``
     - ``tracking_uri``, ``parent_run_id``

Example:
    Log a run to TensorBoard and MLflow.

    .. code-block:: python

        from luxonis_ml.tracker import LuxonisTracker

        with LuxonisTracker(
            project_name="training",
            backends={
                "tensorboard": {},
                "mlflow": {"tracking_uri": "http://localhost:5000"},
            },
        ) as tracker:
            tracker.log_hyperparams({"lr": 1e-3, "batch_size": 32})
            tracker.log_metrics({"acc": 0.92, "loss": 0.18}, step=1)
            tracker.upload_artifact("model.onnx", typ="model")

Note:
    Install the extra of each backend that you enable:

    .. code-block:: bash

        pip install "luxonis-ml[tracker,tensorboard,mlflow]"

    The package itself needs only the ``tracker`` extra. A backend
    imports its SDK when it starts.

.. contents:: Table of Contents
   :depth: 2


Runs and Ranks
==============

A run without ``run_name`` gets ``<number>-<random name>``, and its local
files go to ``<save_directory>/<run_name>``.

Pass ``rank`` in distributed training. Only rank :math:`0` starts the
backends and logs. Rank :math:`0` exports its generated run name in
``LUXONIS_TRACKER_RUN_NAME``, so that a worker that it spawns later
joins the same run. Pass ``run_name`` when all ranks start at the same
time, as with ``torchrun``.


Unreachable Services
====================

A backend that sets `TrackerBackend.buffered`, such as MLflow, does not
stop the training while its service is down. `BufferedBackend` keeps the
calls and sends them later. The calls that never get through go to
``<run_directory>/unsent_logs/<backend name>/`` when the tracker closes.


Custom Backends
===============

Subclass `TrackerBackend` and register the class in `TRACKER_BACKENDS`,
or expose it in the ``tracker_plugins`` entry-point group. The name of
the entry point is the name of the backend:

.. code-block:: toml

    [project.entry-points.tracker_plugins]
    my_service = "my_package.tracking:MyServiceBackend"


Migration
=========

The ``is_tensorboard``, ``is_wandb``, ``is_mlflow``, ``wandb_entity``
and ``mlflow_tracking_uri`` arguments still work, with a
``DeprecationWarning``. Replace them with ``backends``:

.. code-block:: python

    # before
    LuxonisTracker(is_mlflow=True, mlflow_tracking_uri=uri, ...)
    # after
    LuxonisTracker(backends={"mlflow": {"tracking_uri": uri}}, ...)

Other changes that a caller can notice:

    - `LuxonisTracker.experiment` maps each backend name to its native
      handle: an ``MlflowClient`` for MLflow, not the ``mlflow`` module,
      and a WandB ``Run``, not the ``wandb`` module;
    - `LuxonisTracker.close` ends the run in every backend, and the
      tracker ignores the logging calls after it;
    - ``run_id`` and ``project_id`` keep the values that you pass. Read
      the MLflow identifiers from ``tracker.get_backend(MLflowBackend)``;
    - TensorBoard writes through ``tensorboardX``, from the
      ``tensorboard`` extra, and no longer through ``torch``;
    - the unsent MLflow calls go to ``unsent_logs/mlflow/calls.jsonl``,
      not to ``local_logs.json``;
    - ``LuxonisRequestHeaderProvider`` is removed. It was never
      registered with MLflow, and it sent the masked secret.

"""

from importlib.metadata import entry_points

from loguru import logger

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("tracker"):
    from .backends import (
        TRACKER_BACKENDS,
        MLflowBackend,
        RunContext,
        RunStatus,
        TensorBoardBackend,
        TrackerBackend,
        WandbBackend,
    )
    from .buffer import BufferedBackend
    from .tracker import LuxonisTracker


def _load_backend_plugins() -> None:
    """Register the backends of the ``tracker_plugins`` entry points.

    A plugin that fails to load is skipped with a warning, so that one
    broken package does not break this import.
    """
    for entry_point in entry_points(group="tracker_plugins"):
        try:
            backend = entry_point.load()
        except Exception as error:
            logger.warning(
                f"Skipping the tracker plugin '{entry_point.name}': {error}"
            )
            continue
        if not (
            isinstance(backend, type) and issubclass(backend, TrackerBackend)
        ):
            logger.warning(
                f"Skipping the tracker plugin '{entry_point.name}': "
                f"{backend!r} is not a `TrackerBackend` subclass."
            )
            continue
        if entry_point.name not in TRACKER_BACKENDS:
            TRACKER_BACKENDS.register(module=backend, name=entry_point.name)


_load_backend_plugins()

__all__ = [
    "TRACKER_BACKENDS",
    "BufferedBackend",
    "LuxonisTracker",
    "MLflowBackend",
    "RunContext",
    "RunStatus",
    "TensorBoardBackend",
    "TrackerBackend",
    "WandbBackend",
]
