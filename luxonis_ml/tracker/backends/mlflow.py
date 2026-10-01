# pyright: strict
"""The MLflow backend of the tracker.

`MLflowBackend` logs a run to an `MLflow`_ tracking server. It turns on
with ``LuxonisTracker(mlflow=True)``, or with a mapping of
`MLflowOptions`, and needs the ``mlflow`` extra. It is the only built-in
backend that `LuxonisTracker` wraps in a `BufferedBackend`, so an
unreachable server does not stop the training.

.. _MLflow:
    https://mlflow.org/

See:
    `luxonis_ml.tracker` for what each backend does with each logging
    call, and `luxonis_ml.tracker.buffer` for the calls that the server
    does not take.

"""

import os
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Any, TypedDict

import numpy.typing as npt
from loguru import logger
from typing_extensions import Unpack

from luxonis_ml.guard_extras import guard_missing_extra
from luxonis_ml.typing import ParamValue
from luxonis_ml.utils.environ import environ
from luxonis_ml.utils.filesystem import LuxonisFileSystem

from .base import RunContext, RunStatus, TrackerBackend, check_options

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.system_metrics.system_metrics_monitor import (
        SystemMetricsMonitor,
    )


class MLflowOptions(TypedDict, total=False):
    """The options of `MLflowBackend`.

    Attributes:
        tracking_uri: URI of the tracking server. ``None`` takes
            ``MLFLOW_TRACKING_URI`` from the environment.
        parent_run_id: The run to nest this run under. A sweep trial
            without it nests under the last run of this process that is
            still open and is not a sweep trial.

    """

    tracking_uri: str | None
    parent_run_id: str | None


_open_runs: list[str] = []
"""The open runs of this process that are not sweep trials, oldest
first. A sweep trial without ``parent_run_id`` nests under the last one.
"""


class MLflowBackend(TrackerBackend, register_name="mlflow"):
    """Log to an MLflow tracking server.

    The ``project_id`` of the run selects an existing experiment.
    Otherwise its ``project_name`` names the experiment, which is created
    if it does not exist. The ``run_id`` of the run continues an existing
    run. Without it, the backend continues the run in the
    ``MLFLOW_RUN_ID`` environment variable, as ``mlflow run`` sets it, and
    removes the variable, as ``mlflow.start_run`` does. Otherwise the
    backend creates a run with the run name.

    The backend talks to the server through its own ``MlflowClient``,
    so two trackers in one process do not share an active run. For the
    same reason, the run is not the active run of the ``mlflow`` module.
    `artifacts` gives a `LuxonisFileSystem` of the artifacts of the run.
    The backend logs system metrics as well when ``psutil`` is installed.

    `LuxonisTracker` wraps the backend in a `BufferedBackend`, so an
    unreachable server does not stop the training. `is_transient` tells
    an outage from a call that the server rejects.

    Unless the environment sets it already, the backend sets
    ``MLFLOW_HTTP_REQUEST_MAX_RETRIES`` to :math:`2`. The default of
    MLflow retries a failed request for minutes, and each logging call
    would wait that long during an outage.

    Attributes:
        buffered: ``True``, so that `LuxonisTracker` wraps the backend in
            a `BufferedBackend`.
        tracking_uri: URI of the tracking server.
        parent_run_id: The run that this run nests under, as the caller
            gave it.
        experiment_id: The MLflow experiment. Known once the backend is
            started.
        run_id: The MLflow run. Known once the backend is started.

    """

    buffered = True

    def __init__(
        self, run: RunContext, **options: Unpack[MLflowOptions]
    ) -> None:
        """Check the options.

        Args:
            run: The run to log to.
            **options: See `MLflowOptions`.

        Raises:
            TypeError: If an option is unknown.
            ValueError: If no tracking URI is known, or the run has no
                project.

        """
        super().__init__(run)
        check_options(options, MLflowOptions.__optional_keys__)
        tracking_uri = (
            options.get("tracking_uri") or environ.MLFLOW_TRACKING_URI
        )
        if not tracking_uri:
            raise ValueError(
                "MLflow needs `tracking_uri`, or `MLFLOW_TRACKING_URI` in "
                "the environment."
            )
        if run.project_name is None and run.project_id is None:
            raise ValueError("MLflow needs `project_name` or `project_id`.")
        self.tracking_uri = tracking_uri
        self.parent_run_id = options.get("parent_run_id")
        self.experiment_id = run.project_id
        self.run_id = run.run_id
        self._client: MlflowClient | None = None
        self._monitor: SystemMetricsMonitor | None = None

    @property
    def client(self) -> "MlflowClient":
        """The ``MlflowClient`` of the backend, for the calls that the
        tracker does not make, such as ``set_tag``.

        Raises:
            RuntimeError: If the backend is not started.

        """
        if self._client is None:
            raise RuntimeError("The MLflow backend is not started.")
        return self._client

    @property
    def artifacts(self) -> LuxonisFileSystem:
        """A file system for the artifacts of the run.

        Its root is the artifact root of the run, and it uses the
        tracking URI of the backend.

        Example:
            .. code-block:: python

                tracker.mlflow.artifacts.put_file("model.onnx", "model.onnx")

        Raises:
            RuntimeError: If the backend is not started.

        """
        return LuxonisFileSystem(
            f"mlflow://{self.experiment_id}/{self._run_id}",
            tracking_uri=self.tracking_uri,
        )

    def start(self) -> None:
        """Open the experiment and the run.

        The backend looks the experiment up by name, and creates it if it
        does not exist. It creates a run, or continues the run of
        ``run_id`` or of ``MLFLOW_RUN_ID`` and marks it as running again.
        A sweep trial nests under ``parent_run_id``, or else under the
        last open run of this process that is not a sweep trial.

        Raises:
            ImportError: If ``mlflow`` is not installed.
            mlflow.exceptions.MlflowException: If the server cannot be
                reached, or it rejects the experiment or the run.

        """
        with guard_missing_extra("mlflow"):
            from mlflow import MlflowClient

        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        # the client opens a local store at once, so it is not created
        # before the backend starts
        self._client = MlflowClient(self.tracking_uri)
        if self.experiment_id is None:
            self.experiment_id = self._get_or_create_experiment()

        if self.run_id is None:
            self.run_id = os.environ.pop("MLFLOW_RUN_ID", None)
        resumed = self.run_id is not None
        if self.run_id is None:
            self.run_id = self._create_run(self.experiment_id)
        else:
            self.client.update_run(self.run_id, status="RUNNING")

        self._start_system_metrics(self.run_id, resumed)
        if not self.run.is_sweep:
            _open_runs.append(self.run_id)

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Log the hyperparameters as the parameters of the run.

        MLflow stores a parameter as a string, so each value becomes its
        string. MLflow rejects a new value for a parameter that the run
        already has.

        Args:
            params: The hyperparameters, keyed by name.

        Raises:
            RuntimeError: If the backend has not started.
            mlflow.exceptions.MlflowException: If the server rejects the
                call or cannot be reached.

        """
        from mlflow.entities import Param

        self.client.log_batch(
            self._run_id,
            params=[Param(key, str(value)) for key, value in params.items()],
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log the metrics in one request.

        Args:
            metrics: The metric values, keyed by metric name.
            step: The training step of the values.

        Raises:
            RuntimeError: If the backend has not started.
            mlflow.exceptions.MlflowException: If the server rejects the
                call or cannot be reached.

        """
        from mlflow.entities import Metric

        timestamp = time_ns() // 1_000_000
        self.client.log_batch(
            self._run_id,
            metrics=[
                Metric(key, float(value), timestamp, step)
                for key, value in metrics.items()
            ],
        )

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        r"""Log the image as the artifact ``<directory>/<step>/<caption>.png``.

        Args:
            name: Name of the image. The part before the last ``/`` is
                the directory, and the rest is the caption. A name
                without a ``/`` puts the image under ``<step>/``.
            image: The image, of shape :math:`\left(H, W, C\right)`.
            step: The training step of the image.

        Raises:
            RuntimeError: If the backend has not started.
            mlflow.exceptions.MlflowException: If the server rejects the
                call or cannot be reached.

        """
        directory, _, caption = name.rpartition("/")
        path = f"{step}/{caption}.png"
        if directory:
            path = f"{directory}/{path}"
        # MLflow annotates `image` with a bare `numpy.ndarray`
        self.client.log_image(  # pyright: ignore[reportUnknownMemberType]
            self._run_id, image, artifact_file=path
        )

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        """Log the matrix as the artifact ``<name>.json``.

        The file holds the keys ``flat_array`` and ``shape``, and the keys
        of ``extra_data``. Each call replaces the file of the previous
        call with the same name, so the file holds the matrix of the last
        step.

        Args:
            matrix: The matrix.
            name: Name of the matrix, and of the file.
            step: Ignored, because the file holds one matrix.
            extra_data: More keys for the file, such as the class names.

        Raises:
            RuntimeError: If the backend has not started.
            mlflow.exceptions.MlflowException: If the server rejects the
                call or cannot be reached.

        """
        data: dict[str, ParamValue] = {
            "flat_array": matrix.flatten().tolist(),
            "shape": list(matrix.shape),
            **extra_data,
        }
        self.client.log_dict(self._run_id, data, f"{name}.json")

    def upload_artifact(self, path: Path, name: str | None, typ: str) -> None:
        """Upload the file to the root of the run artifacts.

        Args:
            path: Path to the file.
            name: Name to store the file under. Only the last component
                counts, so that a local directory does not leak into the
                artifact store. ``None`` keeps the name of the file.
            typ: Ignored, because MLflow has no artifact types.

        Raises:
            RuntimeError: If the backend has not started.
            Exception: The error of the artifact store, such as an
                ``OSError`` or an error of ``botocore``.

        """
        remote_name = Path(name).name if name else path.name
        LuxonisFileSystem.upload(
            path,
            f"mlflow://{self.experiment_id}/{self._run_id}/{remote_name}",
            tracking_uri=self.tracking_uri,
        )

    def close(self, status: RunStatus) -> None:
        """Stop the system metrics, and end the run.

        Args:
            status: ``"success"`` ends the run as ``FINISHED``,
                ``"failed"`` as ``FAILED``.

        Raises:
            RuntimeError: If the backend has not started.
            mlflow.exceptions.MlflowException: If the server rejects the
                call or cannot be reached.

        """
        if self._run_id in _open_runs:
            _open_runs.remove(self._run_id)
        if self._monitor is not None:
            self._monitor.finish()
        self.client.set_terminated(
            self._run_id, "FINISHED" if status == "success" else "FAILED"
        )

    def is_transient(self, error: Exception) -> bool:
        """Tell an outage from a rejected call.

        An error with an HTTP status is transient for a 5xx status and
        for 429. MLflow reports a connection failure as a status 500.
        A connection error of ``botocore``, from an S3 artifact store,
        is transient too. Any other error follows
        `TrackerBackend.is_transient`.

        Args:
            error: The error of a call to the server.

        Returns:
            ``True`` if a later attempt of the same call can succeed.

        """
        from mlflow.exceptions import MlflowException
        from requests import HTTPError

        if isinstance(error, MlflowException):
            status = error.get_http_status_code()
        elif isinstance(error, HTTPError) and error.response is not None:
            status = error.response.status_code
        else:
            return _is_s3_outage(error) or super().is_transient(error)
        return status >= 500 or status == 429

    @property
    def _run_id(self) -> str:
        """The run identifier, which only a started backend has."""
        if self._client is None or self.run_id is None:
            raise RuntimeError("The MLflow backend is not started.")
        return self.run_id

    def _get_or_create_experiment(self) -> str:
        """Return the experiment of ``project_name``, and create it if
        needed.
        """
        from mlflow.exceptions import MlflowException

        name = self.run.project_name
        # the constructor accepts no run without a project
        assert name is not None
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        try:
            return self.client.create_experiment(name)
        except MlflowException as error:
            # another process created it after the lookup
            if error.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            experiment = self.client.get_experiment_by_name(name)
            assert experiment is not None
            return experiment.experiment_id

    def _create_run(self, experiment_id: str) -> str:
        """Create the run, nested under its parent if it has one."""
        from mlflow.tracking.context.registry import (
            resolve_tags,  # pyright: ignore[reportUnknownVariableType]
        )
        from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

        parent = self.parent_run_id
        if parent is None and self.run.is_sweep and _open_runs:
            parent = _open_runs[-1]
        tags = {} if parent is None else {MLFLOW_PARENT_RUN_ID: parent}
        # `resolve_tags` adds the user, the source and the git commit, as
        # `mlflow.start_run` does. MLflow leaves its types unannotated.
        run = self.client.create_run(
            experiment_id,
            run_name=self.run.run_name,
            tags=resolve_tags(tags),  # pyright: ignore[reportUnknownArgumentType]
        )
        return run.info.run_id

    def _start_system_metrics(self, run_id: str, resumed: bool) -> None:
        """Start the system metrics monitor, which is optional."""
        if find_spec("psutil") is None:
            logger.warning("Install `psutil` to log system metrics to MLflow.")
            return
        from mlflow.system_metrics.system_metrics_monitor import (
            SystemMetricsMonitor,
        )

        try:
            self._monitor = SystemMetricsMonitor(
                run_id, resume_logging=resumed, tracking_uri=self.tracking_uri
            )
            self._monitor.start()
        except Exception as error:
            logger.warning(f"Could not log the system metrics: {error}")


def _is_s3_outage(error: Exception) -> bool:
    """Tell whether ``botocore`` could not reach the S3 store.

    Its connection errors are no ``OSError``.
    """
    try:
        # botocore ships no type stubs
        from botocore.exceptions import (  # pyright: ignore[reportMissingTypeStubs]
            ConnectionError,
            HTTPClientError,
        )
    except ImportError:
        return False
    return isinstance(error, ConnectionError | HTTPClientError)
