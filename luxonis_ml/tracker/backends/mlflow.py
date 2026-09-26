# pyright: strict
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

from .base import RunContext, RunStatus, TrackerBackend

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


class MLflowBackend(TrackerBackend):
    """Log to an `MLflow`_ tracking server.

    The backend talks to the server through its own ``MlflowClient``,
    so two trackers in one process do not share an active run. It logs
    system metrics as well when ``psutil`` is installed.

    `LuxonisTracker` wraps the backend in a `BufferedBackend`, so an
    unreachable server does not stop the training. A call that the
    server rejects with a 4xx status is dropped.

    Unless the environment sets it already, the backend sets
    ``MLFLOW_HTTP_REQUEST_MAX_RETRIES`` to :math:`2`. The default of
    MLflow retries a failed request for minutes, and each logging call
    would wait that long during an outage.

    Attributes:
        experiment_id: The MLflow experiment. Known once the backend is
            started.
        run_id: The MLflow run. Known once the backend is started.

    .. _MLflow:
        https://mlflow.org/

    """

    buffered = True

    def __init__(
        self, run: RunContext, **options: Unpack[MLflowOptions]
    ) -> None:
        """Check the options.

        The ``project_id`` of the run selects an existing experiment.
        Otherwise its ``project_name`` names the experiment, which is
        created if it does not exist. The ``run_id`` of the run continues
        an existing run.

        Args:
            run: The run to log to.
            **options: See `MLflowOptions`.

        Raises:
            ValueError: If no tracking URI is known, or the run has no
                project.

        """
        super().__init__(run)
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
    def experiment(self) -> "MlflowClient":
        """The ``MlflowClient`` of the backend."""
        if self._client is None:
            raise RuntimeError("The MLflow backend is not started.")
        return self._client

    def start(self) -> None:
        with guard_missing_extra("mlflow"):
            from mlflow import MlflowClient

        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        # the client opens a local store at once, so it is not created
        # before the backend starts
        self._client = MlflowClient(self.tracking_uri)
        if self.experiment_id is None:
            self.experiment_id = self._get_or_create_experiment()

        resumed = self.run_id is not None
        if self.run_id is None:
            self.run_id = self._create_run(self.experiment_id)
        else:
            self.experiment.update_run(self.run_id, status="RUNNING")

        self._start_system_metrics(self.run_id, resumed)
        if not self.run.is_sweep:
            _open_runs.append(self.run_id)

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        from mlflow.entities import Param

        self.experiment.log_batch(
            self._run_id,
            params=[Param(key, str(value)) for key, value in params.items()],
        )

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        from mlflow.entities import Metric

        timestamp = time_ns() // 1_000_000
        self.experiment.log_batch(
            self._run_id,
            metrics=[
                Metric(key, float(value), timestamp, step)
                for key, value in metrics.items()
            ],
        )

    def log_image(self, name: str, image: npt.NDArray[Any], step: int) -> None:
        """Log the image as ``<directory>/<step>/<caption>.png``.

        The directory is the part of ``name`` before the last ``/``. A
        name without a ``/`` puts the image under ``<step>/``.
        """
        directory, _, caption = name.rpartition("/")
        path = f"{step}/{caption}.png"
        if directory:
            path = f"{directory}/{path}"
        # MLflow annotates `image` with a bare `numpy.ndarray`
        self.experiment.log_image(  # pyright: ignore[reportUnknownMemberType]
            self._run_id, image, artifact_file=path
        )

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue],
    ) -> None:
        """Log the matrix as ``<name>.json``.

        The file holds ``flat_array``, ``shape`` and the ``extra_data``.
        Each call replaces the file of the previous one.
        """
        data: dict[str, ParamValue] = {
            "flat_array": matrix.flatten().tolist(),
            "shape": list(matrix.shape),
            **extra_data,
        }
        self.experiment.log_dict(self._run_id, data, f"{name}.json")

    def upload_artifact(self, path: Path, name: str | None, typ: str) -> None:
        """Upload the file to the root of the run artifacts.

        The file is stored under the last component of ``name``, or else
        under its own name, so that a local directory does not leak into
        the artifact store.
        """
        remote_name = Path(name).name if name else path.name
        LuxonisFileSystem.upload(
            path,
            f"mlflow://{self.experiment_id}/{self._run_id}/{remote_name}",
            tracking_uri=self.tracking_uri,
        )

    def close(self, status: RunStatus) -> None:
        if self._run_id in _open_runs:
            _open_runs.remove(self._run_id)
        if self._monitor is not None:
            self._monitor.finish()
        self.experiment.set_terminated(
            self._run_id, "FINISHED" if status == "success" else "FAILED"
        )

    def is_transient(self, error: Exception) -> bool:
        """Tell an outage from a rejected call.

        An error with an HTTP status is transient for a 5xx status and
        for 429. MLflow reports a connection failure as a status 500.
        Any other error follows `TrackerBackend.is_transient`.
        """
        from mlflow.exceptions import MlflowException
        from requests import HTTPError

        if isinstance(error, MlflowException):
            status = error.get_http_status_code()
        elif isinstance(error, HTTPError) and error.response is not None:
            status = error.response.status_code
        else:
            return super().is_transient(error)
        return status >= 500 or status == 429

    @property
    def _run_id(self) -> str:
        if self.run_id is None:
            raise RuntimeError("The MLflow backend is not started.")
        return self.run_id

    def _get_or_create_experiment(self) -> str:
        name = self.run.project_name
        # the constructor accepts no run without a project
        assert name is not None
        experiment = self.experiment.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        return self.experiment.create_experiment(name)

    def _create_run(self, experiment_id: str) -> str:
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
        run = self.experiment.create_run(
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
