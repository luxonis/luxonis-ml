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

        tracker = LuxonisTracker(
            project_name="training",
            run_name="baseline",
            is_tensorboard=True,
        )
        tracker.log_metric("loss", 0.42, step=1)
        tracker.close()

Note:
    The ``tracker`` extra installs no backend SDK. Install the SDK of each
    backend you enable:

        - TensorBoard needs ``torch``, because the writer comes from
          ``torch.utils.tensorboard``;
        - Weights & Biases needs ``wandb``;
        - MLflow needs ``mlflow``, which the ``luxonis-ml[mlflow]`` extra
          also provides.

    The tracker imports each SDK only when you enable that backend, so an
    absent SDK matters only for the backends you turn on.

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
     - Nothing. The writer uses ``<save_directory>/tensorboard_logs``.
   * - ``is_wandb``
     - ``wandb_entity``, and ``project_name`` or ``project_id``.
   * - ``is_mlflow``
     - ``mlflow_tracking_uri``, and ``project_name`` or ``project_id``.


Logging API
===========

`LuxonisTracker` sends each call to every enabled backend:

    - `LuxonisTracker.log_hyperparams`;
    - `LuxonisTracker.log_metric` and `LuxonisTracker.log_metrics`;
    - `LuxonisTracker.log_image` and `LuxonisTracker.log_images`;
    - `LuxonisTracker.log_matrix`;
    - `LuxonisTracker.upload_artifact`;
    - `LuxonisTracker.close`.

The images are ``numpy`` arrays of shape :math:`\left(H, W, C\right)`.


MLflow Notes
============

MLflow starts on the first access to `LuxonisTracker.experiment`. A failed
MLflow call does not raise. The tracker buffers the payload and retries it
after the next successful call.

`LuxonisTracker.close` writes a buffer that is still not empty under
``<save_directory>/<run_name>``:

    - ``local_logs.json`` holds the metrics, the parameters, the matrices,
      and the index of the saved images and artifacts;
    - ``images/`` holds the buffered images;
    - ``artifacts/`` holds a copy of each buffered artifact whose source
      file still exists.

Set ``MLFLOW_CLOUDFLARE_ID`` and ``MLFLOW_CLOUDFLARE_SECRET`` for an MLflow
server behind Cloudflare Access. Register
`LuxonisRequestHeaderProvider` in your application, because LuxonisML
declares no MLflow entry point for it.

See:
    `luxonis_ml.tracker.tracker` for the logging implementation and
    `luxonis_ml.tracker.mlflow_plugins` for MLflow request-header support.

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("tracker"):
    from .mlflow_plugins import LuxonisRequestHeaderProvider
    from .tracker import LuxonisTracker

__all__ = ["LuxonisRequestHeaderProvider", "LuxonisTracker"]
