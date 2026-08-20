r"""Experiment tracking facade for Luxonis ML workflows.

The `luxonis_ml.tracker` package exports `LuxonisTracker`, a unified logging
interface for TensorBoard, Weights & Biases, and MLflow. Training and
evaluation code can log metrics, hyperparameters, images, matrices, and
artifacts through one API while choosing the enabled backends at runtime.

Pass ``rank`` in distributed training. The rank-gated logging methods, such
as `LuxonisTracker.log_metric`, write only on rank 0. The helpers that save
or replay the local buffer do not check the rank.

Example:
    Start a TensorBoard-backed run and log a scalar metric.

    .. code-block:: python

        from luxonis_ml.tracker import LuxonisTracker

        with LuxonisTracker(
            project_name="training",
            run_name="baseline",
            is_tensorboard=True,
        ) as tracker:
            tracker.log_metric("loss", 0.42, step=1)

Note:
    The ``tracker`` extra does not install the backend SDKs. Install the SDK
    of each backend you enable:

        - TensorBoard needs ``torch``, because the writer comes from
          ``torch.utils.tensorboard``;
        - Weights & Biases needs ``wandb``;
        - MLflow needs ``mlflow``, which the ``mlflow`` extra installs.

    .. code-block:: bash

        pip install "luxonis-ml[tracker,mlflow]" torch wandb

    The tracker imports ``torch``, ``wandb``, and ``mlflow`` only when you
    enable those backends. An absent SDK thus matters only for the backends
    you turn on.

.. contents:: Table of Contents
   :depth: 2


Enabling the Backends
=====================

Each backend has its own flag. Turn on as many as you need in one run:

.. code-block:: python

    from luxonis_ml.tracker import LuxonisTracker

    tracker = LuxonisTracker(
        project_name="my-project",
        run_name="baseline",
        is_tensorboard=True,
        is_wandb=True,
        wandb_entity="my-entity",
    )

    tracker.log_hyperparams({"lr": 1e-3, "batch_size": 32})
    tracker.log_metrics({"acc": 0.92, "loss": 0.18}, step=1)
    tracker.upload_artifact("model.onnx", name="model", typ="model")
    tracker.close()

Two backends need one more argument, and the constructor raises
``ValueError`` without it:

.. list-table:: Backend arguments
   :header-rows: 1

   * - Flag
     - Also needs
   * - ``is_tensorboard``
     - Nothing. The writer uses
       ``<save_directory>/tensorboard_logs/<run_name>``. A sweep run adds a
       ``trial_<n>`` level.
   * - ``is_wandb``
     - ``wandb_entity``, and ``project_name`` or ``project_id``.
   * - ``is_mlflow``
     - ``mlflow_tracking_uri``, and ``project_name`` or ``project_id``.


Logging API
===========

`LuxonisTracker` sends each of these calls to every enabled backend that
supports it:

    - `LuxonisTracker.log_hyperparams`;
    - `LuxonisTracker.log_metric` and `LuxonisTracker.log_metrics`;
    - `LuxonisTracker.log_image` and `LuxonisTracker.log_images`;
    - `LuxonisTracker.log_matrix`;
    - `LuxonisTracker.upload_artifact`.

TensorBoard accepts no artifact, so `LuxonisTracker.upload_artifact` reaches
Weights & Biases and MLflow only.

`LuxonisTracker.close` finalizes each enabled backend. It flushes and closes
the TensorBoard writer, ends the MLflow run, and finishes the WandB run. A
second call does nothing. If one backend fails to shut down, the tracker
reports the failure and does not raise it. The other backends thus keep
their data.

The images are ``numpy`` arrays of shape :math:`\left(H, W, C\right)`.


MLflow Notes
============

MLflow starts on the first access to `LuxonisTracker.experiment`. A failed
MLflow call does not raise. The tracker buffers the call and replays the
buffer in order when the server answers again. It retries a failed
connection only once per minute, so an unreachable server does not stall the
training loop.

MLflow can reject one call and still accept the calls behind it. The tracker
then drops the rejected call, so one bad call cannot block every later log.

The tracker copies a buffered artifact into the run directory at once,
because callers routinely delete the file after they hand it over.

The buffer holds a maximum of 500 calls, of which 50 can be images. When the
buffer is full, the tracker drops the oldest call and warns once. The
hyperparameters and the artifacts go last, because no later call repeats
them. They still give way to the call that has just arrived, or a long
outage would let them block every later metric.

A call that arrives after `LuxonisTracker.close` is ignored. MLflow would
otherwise open a fresh run for it, and no later `close` would end that run.

`LuxonisTracker.close` waives the retry backoff to give the server one last
chance. It then writes a buffer that is still not empty under
``<save_directory>/<run_name>``:

    - ``local_logs.json`` holds the metrics, the parameters, the matrices,
      and the index of the saved images and artifacts. A later save appends
      to this file and keeps the earlier records;
    - ``images/`` holds the buffered images;
    - ``artifacts/<n>/`` holds a copy of each buffered artifact. Each
      artifact gets its own directory, which keeps artifacts with the same
      file name apart.

See:
    `luxonis_ml.tracker.tracker` for the logging implementation.

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("tracker"):
    from .tracker import LuxonisTracker

__all__ = ["LuxonisTracker"]
