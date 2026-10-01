r"""Experiment tracking for Luxonis ML workflows.

`LuxonisTracker` logs a run to several tracking services at once:
TensorBoard, Weights & Biases, MLflow, and any service that a plugin
adds. Training and evaluation code log hyperparameters, metrics, images,
matrices, and artifacts through one API, and choose the services at
runtime.

Example:
    Log a run to TensorBoard and MLflow.

    .. code-block:: python

        import numpy as np

        from luxonis_ml.tracker import LuxonisTracker

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        with LuxonisTracker(
            project_name="training",
            tensorboard=True,
            mlflow={"tracking_uri": "http://localhost:5000"},
        ) as tracker:
            tracker.log_hyperparams({"lr": 1e-3, "batch_size": 32})
            tracker.log_metrics({"acc": 0.92, "loss": 0.18}, step=1)
            tracker.log_image("val/prediction", image, step=1)
            tracker.upload_artifact("model.onnx", typ="model")

Note:
    Install the extra of each backend that you turn on:

    .. code-block:: bash

        pip install "luxonis-ml[tracker,tensorboard,mlflow]"

    ``luxonis_ml.tracker`` itself needs only the ``tracker`` extra. A
    backend imports its SDK when it starts.

.. contents:: Table of Contents
   :depth: 2


Enabling the Backends
=====================

Each backend has a keyword argument of `LuxonisTracker`. ``True`` turns
the backend on with its defaults, and a mapping turns it on with
options. ``None`` or ``False`` leaves it off. At least one backend must
be on.

.. list-table:: Built-in backends
   :header-rows: 1

   * - Keyword
     - Backend
     - Extra
     - Options
   * - ``tensorboard``
     - `TensorBoardBackend`
     - ``tensorboard``
     - None.
   * - ``wandb``
     - `WandbBackend`
     - ``wandb``
     - `WandbOptions`: ``entity``.
   * - ``mlflow``
     - `MLflowBackend`
     - ``mlflow``
     - `MLflowOptions`: ``tracking_uri``, which defaults to
       ``MLFLOW_TRACKING_URI`` from the environment, and
       ``parent_run_id``.

WandB and MLflow need ``project_name`` or ``project_id``. A backend
rejects an option that it does not know with ``TypeError``, and so does
the tracker for a keyword that names no backend.

The backends start on the first logging call. Call
`LuxonisTracker.start` to start them earlier, for example to create the
MLflow run of a sweep before its trials.

Each built-in backend is a property of the tracker:
`LuxonisTracker.tensorboard`, `LuxonisTracker.wandb` and
`LuxonisTracker.mlflow`. The property starts the backends first. On
rank :math:`0`, the SDK handle of the backend is then ready for a call
that the tracker does not make: `TensorBoardBackend.writer`,
`WandbBackend.wandb_run`, or `MLflowBackend.client`.

.. code-block:: python

    run_id = tracker.mlflow.run_id
    tracker.mlflow.client.set_tag(run_id, "stage", "export")
    tracker.mlflow.artifacts.put_file("model.onnx", "model.onnx")
    writer = tracker.tensorboard.writer

The property of a backend that is off raises ``AttributeError``.
`LuxonisTracker.backends` gives each enabled backend by its keyword, a
plugin backend included, and does not start them.


Logging API
===========

`LuxonisTracker` has the logging methods of `TrackerBackend`, and sends
each call to each backend that is on. A backend that fails a call gives
a warning, and the other backends still get the call.
Each backend stores the call in the form that its service knows:

.. list-table:: What each backend does with a call
   :header-rows: 1

   * - Call
     - TensorBoard
     - WandB
     - MLflow
   * - `LuxonisTracker.log_hyperparams`
     - One set in the HParams dashboard, written when the run closes.
       A value that is not a scalar becomes a string.
     - The configuration of the run.
     - The parameters of the run, as strings. MLflow rejects a new value
       for a parameter that the run has already.
   * - `LuxonisTracker.log_metric`, `LuxonisTracker.log_metrics`
     - A scalar at ``step``.
     - A value at the next WandB step.
     - A metric at ``step``.
   * - `LuxonisTracker.log_image`, `LuxonisTracker.log_images`
     - An image at ``step``.
     - A ``wandb.Image`` at the next WandB step.
     - The artifact ``<directory>/<step>/<caption>.png``. The directory
       is the part of the name before the last ``/``.
   * - `LuxonisTracker.log_matrix`
     - Text of the whole matrix at ``step``.
     - The table ``<name>_table``.
     - The artifact ``<name>.json``, with ``flat_array``, ``shape`` and
       the ``extra_data``. A later step replaces it.
   * - `LuxonisTracker.upload_artifact`
     - Nothing.
     - A WandB artifact of the type ``typ``.
     - A file at the root of the run artifacts.
   * - `LuxonisTracker.close`
     - Closes the event file.
     - Finishes the run with the exit code :math:`0` or :math:`1`.
     - Ends the run as ``FINISHED`` or ``FAILED``.

WandB receives no ``step``, because it drops a call whose step is lower
than the step of the call before. The images are ``numpy`` arrays of
shape :math:`\left(H, W, C\right)`.


Runs and Local Files
====================

A run without ``run_name`` gets the name ``<number>-<random name>``. The
number is one more than the highest number in ``save_directory``. The
tracker creates these local files:

.. code-block:: text

    <save_directory>/
        <run_name>/                     the run directory
            unsent_logs/<backend>/      the calls that never got through
        tensorboard_logs/<run_name>/    the TensorBoard events
            trial_<n>/                  one directory for each sweep trial
        wandb_logs/                     the local files of WandB

A sweep trial passes ``is_sweep=True``. Its TensorBoard events go to the
next ``trial_<n>`` directory, and its MLflow run nests under
``parent_run_id``, or else under the last open MLflow run of the process
that is not a sweep trial.


Distributed Training
====================

Pass ``rank``. Only rank :math:`0` starts the backends and logs. On the
other ranks `LuxonisTracker.start` and each logging call do nothing.

The ranks must agree on the run name. Rank :math:`0` exports a
generated name in ``LUXONIS_TRACKER_RUN_NAME``, and a worker that it
starts later, as Lightning does, joins that run. When all ranks start at
the same time, as with ``torchrun``, a worker waits for a new run
directory. After one second it joins the newest run, and after 30
seconds without any run it raises ``RuntimeError``. Pass ``run_name`` to
every rank to avoid the wait.


Closing the Run
===============

`LuxonisTracker.close` ends the run in each backend that started. The
status ``"success"`` or ``"finished"`` marks the run as successful, and
any other status as failed. A second call does nothing, and the tracker
ignores the logging calls after it, with a warning.

Use the tracker as a context manager to close the run with the right
status. A run that is still open when the interpreter exits closes then,
as failed after an uncaught error.

`LuxonisTracker.flush` writes the pending data and keeps the run open,
for example the TensorBoard events that are still in memory. Call it when
a run stays open for later calls, but its data must be on disk now.


Unreachable Services
====================

MLflow runs on a server that can be down for a while.
`LuxonisTracker` therefore wraps `MLflowBackend` in a `BufferedBackend`,
which keeps the training going:

    - A call that fails because the server is down waits in a buffer.
      The tracker tries the server again after 60 seconds, and sends the
      buffer first, in order.
    - A call that the server rejects, for example with a 4xx status, is
      dropped with a warning.
    - A first start that the server rejects, for example for an unknown
      ``project_id``, raises at once. A later start that it rejects keeps
      the calls for the local save.
    - The buffer holds at most 100 hyperparameter calls, 500 metric
      calls, 50 images, 500 matrices and 10 artifacts. A full buffer
      drops the oldest call of that kind.
    - A buffered artifact is kept as a hard link or a copy, so the
      caller can delete the file.

`LuxonisTracker.close` tries the buffer one last time. The calls that
still fail go to ``<run_directory>/unsent_logs/mlflow/``.
``calls.jsonl`` holds one JSON object for each call, in order, with the
name of the call and its arguments. The images are ``.npy`` files in
``images/``, and the artifacts stay in ``artifacts/``.

Unless the environment sets it already, the MLflow backend sets
``MLFLOW_HTTP_REQUEST_MAX_RETRIES`` to :math:`2`, so that a call during
an outage fails fast.


Custom Backends
===============

A backend is a subclass of `TrackerBackend`. The subclass registers
itself in `TRACKER_BACKENDS` under its ``register_name``, which is also
the keyword argument that turns it on. Its constructor takes the options
as keyword arguments. This backend writes the metrics of a run to a
JSON Lines file:

.. code-block:: python

    import json
    from collections.abc import Mapping
    from typing import IO, Any

    import numpy.typing as npt

    from luxonis_ml.tracker import (
        LuxonisTracker,
        RunContext,
        RunStatus,
        TrackerBackend,
    )
    from luxonis_ml.typing import ParamValue


    class JsonLinesBackend(TrackerBackend, register_name="jsonl"):
        _file: IO[str]

        def __init__(
            self, run: RunContext, *, filename: str = "log.jsonl"
        ) -> None:
            super().__init__(run)
            self.filename = filename

        def start(self) -> None:
            path = self.run.run_directory / self.filename
            self._file = path.open("a")

        def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
            self._write({"params": dict(params)})

        def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
            self._write({"step": step, "metrics": dict(metrics)})

        def log_image(
            self, name: str, image: npt.NDArray[Any], step: int
        ) -> None:
            pass  # the file holds no images

        def log_matrix(
            self,
            matrix: npt.NDArray[Any],
            name: str,
            step: int,
            extra_data: Mapping[str, ParamValue],
        ) -> None:
            self._write({"step": step, name: matrix.tolist()})

        def close(self, status: RunStatus) -> None:
            self._write({"status": status})
            self._file.close()

        def _write(self, record: dict[str, Any]) -> None:
            self._file.write(json.dumps(record) + "\n")


    with LuxonisTracker(jsonl={"filename": "train.jsonl"}) as tracker:
        tracker.log_metrics({"loss": 0.18}, step=1)

``tracker.backends["jsonl"]`` gives the backend itself.

The constructor runs on every rank, so it only checks and stores the
options. `TrackerBackend.start` runs once, on rank :math:`0`, before
the first logging call, and opens what the logging calls need.
`TrackerBackend.log_metric` and `TrackerBackend.log_images` call
`TrackerBackend.log_metrics` and `TrackerBackend.log_image`, unless the
backend overrides them. `TrackerBackend.upload_artifact` does nothing
unless the backend overrides it.

For a remote service, set `TrackerBackend.buffered` to ``True``, and
override `TrackerBackend.is_transient` to tell an outage from a call
that the service rejects. `LuxonisTracker` then wraps the backend in a
`BufferedBackend`.

A package makes its backend available through the ``tracker_plugins``
entry-point group. Importing ``luxonis_ml.tracker`` loads each entry
point of the group, and registers the class under the name of the entry
point. Name the entry point after the ``register_name`` of the class:

.. code-block:: toml

    [project.entry-points.tracker_plugins]
    jsonl = "my_package.tracking:JsonLinesBackend"

A plugin that fails to load is skipped with a warning. A plugin with the
name of a built-in backend replaces the built-in one, also when it is a
subclass of it.

See:
    `LuxonisTracker` for the arguments of the tracker,
    `luxonis_ml.tracker.backends` for the built-in backends, and
    `luxonis_ml.tracker.buffer` for the buffer of a remote backend.

"""

from importlib.metadata import entry_points

from loguru import logger

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("tracker"):
    from .backends import (
        TRACKER_BACKENDS,
        MLflowBackend,
        MLflowOptions,
        RunContext,
        RunStatus,
        TensorBoardBackend,
        TrackerBackend,
        WandbBackend,
        WandbOptions,
    )
    from .buffer import BufferedBackend
    from .tracker import LuxonisTracker


def _load_backend_plugins() -> None:
    """Load the backends of the ``tracker_plugins`` entry points.

    Each plugin is registered under the name of its entry point.
    `AutoRegisterMeta` keeps a registered class over a subclass of the
    same name, so without this a plugin that extends a built-in backend
    would not replace it. A plugin that fails to load, or that is not a
    `TrackerBackend` subclass, is skipped with a warning, so that one
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
        TRACKER_BACKENDS.register(
            module=backend, name=entry_point.name, force=True
        )


_load_backend_plugins()

__all__ = [
    "TRACKER_BACKENDS",
    "BufferedBackend",
    "LuxonisTracker",
    "MLflowBackend",
    "MLflowOptions",
    "RunContext",
    "RunStatus",
    "TensorBoardBackend",
    "TrackerBackend",
    "WandbBackend",
    "WandbOptions",
]
