"""Experiment tracking facade for Luxonis ML workflows.

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

See:
    `luxonis_ml.tracker.tracker` for the logging implementation and
    `luxonis_ml.tracker.mlflow_plugins` for MLflow request-header support.

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("tracker"):
    from .mlflow_plugins import LuxonisRequestHeaderProvider
    from .tracker import LuxonisTracker

__all__ = ["LuxonisRequestHeaderProvider", "LuxonisTracker"]
