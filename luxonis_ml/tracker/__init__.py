"""Experiment tracking facade for Luxonis ML workflows.

The `luxonis_ml.tracker` package exports `LuxonisTracker`, a unified logging
interface for TensorBoard, Weights & Biases, and MLflow. Training and
evaluation code can log metrics, hyperparameters, images, matrices, and
artifacts through one API while choosing the enabled backends at runtime.

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
    The tracker uses optional dependencies. Install ``luxonis-ml[tracker]``
    and the backend SDKs required by the integrations you enable.

See:
    `luxonis_ml.tracker.tracker` for the logging implementation and
    `luxonis_ml.tracker.mlflow_plugins` for MLflow request-header support.

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("tracker"):
    from .mlflow_plugins import LuxonisRequestHeaderProvider
    from .tracker import LuxonisTracker

__all__ = ["LuxonisRequestHeaderProvider", "LuxonisTracker"]
