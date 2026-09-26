# pyright: strict
import os
import re
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, TypeVar

import numpy.typing as npt
from loguru import logger
from typing_extensions import Self, deprecated

# the package has no type annotations
from unique_names_generator import (  # pyright: ignore[reportMissingTypeStubs]
    get_random_name,  # pyright: ignore[reportUnknownVariableType]
)

from luxonis_ml.typing import ParamValue, PathType

from .backends.base import (
    TRACKER_BACKENDS,
    RunContext,
    RunStatus,
    TrackerBackend,
)
from .backends.mlflow import MLflowOptions
from .backends.wandb import WandbOptions
from .buffer import BufferedBackend

B = TypeVar("B", bound=TrackerBackend)

RUN_NAME_ENV = "LUXONIS_TRACKER_RUN_NAME"
"""The environment variable that hands the run name to the other ranks.

Rank :math:`0` exports the run name that it generates. A worker process
that starts later inherits the variable and joins the same run.
"""

_JOIN_TIMEOUT = 30.0
_JOIN_GRACE_PERIOD = 1.0
_JOIN_POLL_INTERVAL = 0.5


class LuxonisTracker:
    """Log a run to several tracking services at once.

    The tracker sends each logging call to each enabled backend. The
    backends come from `TRACKER_BACKENDS`, so a plugin can add one.

    Only rank :math:`0` logs. On the other ranks every logging call does
    nothing, and no backend starts.

    The backends start on the first logging call, or on the first access
    to `experiment`. `close` ends the run in each backend. Use the
    tracker as a context manager to close it with the right status.

    Attributes:
        project_name: Project name.
        project_id: Project identifier.
        run_name: Name of the run.
        run_id: Identifier of an earlier run to continue.
        save_directory: Root directory of the local run outputs.
        run_directory: Local directory of the run,
            ``<save_directory>/<run_name>``.
        is_sweep: Whether the run is one trial of a sweep.
        rank: Rank of the process in distributed training.

    """

    def __init__(
        self,
        project_name: str | None = None,
        project_id: str | None = None,
        run_name: str | None = None,
        run_id: str | None = None,
        save_directory: PathType = "output",
        is_tensorboard: bool = False,
        is_wandb: bool = False,
        is_mlflow: bool = False,
        is_sweep: bool = False,
        wandb_entity: str | None = None,
        mlflow_tracking_uri: str | None = None,
        rank: int = 0,
        *,
        tensorboard: bool | None = None,
        wandb: WandbOptions | bool | None = None,
        mlflow: MLflowOptions | bool | None = None,
        **plugins: Mapping[str, object] | bool | None,
    ) -> None:
        """Create a tracker.

        Each backend has a keyword argument of its name. ``True`` turns
        the backend on with its defaults, a mapping passes its options,
        and ``None`` or ``False`` leaves it off.

        Args:
            project_name: Project name.
            project_id: Project identifier.
            run_name: Name of the run. If omitted, rank :math:`0`
                generates ``<number>-<random name>``, and the other ranks
                join that run.
            run_id: Identifier of an earlier run to continue.
            save_directory: Root directory of the local run outputs.
            is_tensorboard: Deprecated. Use ``tensorboard``.
            is_wandb: Deprecated. Use ``wandb``.
            is_mlflow: Deprecated. Use ``mlflow``.
            is_sweep: Whether the run is one trial of a sweep.
            wandb_entity: Deprecated. Use the ``entity`` option of
                ``wandb``.
            mlflow_tracking_uri: Deprecated. Use the ``tracking_uri``
                option of ``mlflow``.
            rank: Rank of the process in distributed training.
            tensorboard: Whether to log to `TensorBoardBackend`.
            wandb: `WandbBackend`, with the options of `WandbOptions`.
            mlflow: `MLflowBackend`, with the options of `MLflowOptions`.
            **plugins: The other backends in `TRACKER_BACKENDS`, keyed by
                their name.

        Raises:
            ValueError: If no backend is enabled, or a backend rejects
                its options.
            TypeError: If a keyword argument names no backend in
                `TRACKER_BACKENDS`.

        """
        configs = _legacy_backends(
            is_tensorboard=is_tensorboard,
            is_wandb=is_wandb,
            is_mlflow=is_mlflow,
            wandb_entity=wandb_entity,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        requested = {
            "tensorboard": tensorboard,
            "wandb": wandb,
            "mlflow": mlflow,
            **plugins,
        }
        for name, value in requested.items():
            if name not in TRACKER_BACKENDS:
                raise TypeError(
                    "LuxonisTracker got an unexpected keyword argument "
                    f"'{name}', and no tracker backend has that name."
                )
            # an explicit `False` also turns off a deprecated flag
            if value is False:
                configs.pop(name, None)
            elif value is True:
                configs[name] = {}
            elif value is not None:
                configs[name] = value
        if not configs:
            raise ValueError("Enable at least one backend.")

        self.project_name = project_name
        self.project_id = project_id
        self.run_id = run_id
        self.is_sweep = is_sweep
        self.rank = rank
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)

        if not run_name:
            if rank == 0:
                run_name = _new_run_name(self.save_directory)
                os.environ[RUN_NAME_ENV] = run_name
            else:
                run_name = _join_run(self.save_directory)
        self.run_name = run_name

        run = RunContext(
            run_name=run_name,
            save_directory=self.save_directory,
            project_name=project_name,
            project_id=project_id,
            run_id=run_id,
            is_sweep=is_sweep,
        )
        self._backends = {
            name: _create_backend(name, run, options)
            for name, options in configs.items()
        }
        self._started: dict[str, TrackerBackend] = {}
        self._closed = False

        self.run_directory = run.run_directory
        self.run_directory.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close("success" if exc_type is None else "failed")

    @property
    def name(self) -> str:
        """The run name."""
        return self.run_name

    @property
    def version(self) -> int:
        """The number of the run, or :math:`0` for a run name without
        one.
        """
        return _run_number(self.run_name) or 0

    @property
    def backends(self) -> Mapping[str, TrackerBackend]:
        """The enabled backends, keyed by name."""
        return MappingProxyType(self._backends)

    def get_backend(self, backend_type: type[B]) -> B:
        """Return the enabled backend of a type.

        Example:
            .. code-block:: python

                run_id = tracker.get_backend(MLflowBackend).run_id

        Raises:
            KeyError: If no backend of ``backend_type`` is enabled.

        """
        for backend in self._backends.values():
            if isinstance(backend, BufferedBackend):
                backend = backend.backend
            if isinstance(backend, backend_type):
                return backend
        raise KeyError(f"No {backend_type.__name__} is enabled.")

    @property
    def experiment(self) -> dict[str, Any]:
        """The native handles of the started backends, keyed by name.

        Reading it starts the backends. It is empty on a non-zero rank.
        Use `get_backend` for a typed handle.
        """
        if not self._closed:
            self._start_backends()
        return {
            name: backend.experiment for name, backend in self._started.items()
        }

    @property
    @deprecated("Use `'tensorboard' in tracker.backends` instead.")
    def is_tensorboard(self) -> bool:
        """Whether TensorBoard is enabled. Deprecated."""
        return "tensorboard" in self._backends

    @property
    @deprecated("Use `'wandb' in tracker.backends` instead.")
    def is_wandb(self) -> bool:
        """Whether WandB is enabled. Deprecated."""
        return "wandb" in self._backends

    @property
    @deprecated("Use `'mlflow' in tracker.backends` instead.")
    def is_mlflow(self) -> bool:
        """Whether MLflow is enabled. Deprecated."""
        return "mlflow" in self._backends

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Log the hyperparameters of the run."""
        for backend in self._live_backends():
            backend.log_hyperparams(params)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one scalar metric."""
        self.log_metrics({name: value}, step)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics at ``step``."""
        for backend in self._live_backends():
            backend.log_metrics(metrics, step)

    def log_image(self, name: str, img: npt.NDArray[Any], step: int) -> None:
        r"""Log an image of shape :math:`\left(H, W, C\right)`.

        Args:
            name: Caption of the image. MLflow uses the part before the
                last ``/`` as the directory.
            img: The image.
            step: Current step.

        """
        for backend in self._live_backends():
            backend.log_image(name, img, step)

    def log_images(
        self, imgs: Mapping[str, npt.NDArray[Any]], step: int
    ) -> None:
        """Log several images, keyed by caption."""
        for name, img in imgs.items():
            self.log_image(name, img, step)

    def log_matrix(
        self,
        matrix: npt.NDArray[Any],
        name: str,
        step: int,
        extra_data: Mapping[str, ParamValue] | None = None,
    ) -> None:
        """Log a matrix, such as a confusion matrix.

        Args:
            matrix: The matrix.
            name: Name of the matrix.
            step: Current step.
            extra_data: More data to store with the matrix. Only MLflow
                stores it.

        """
        for backend in self._live_backends():
            backend.log_matrix(matrix, name, step, extra_data or {})

    def upload_artifact(
        self, path: PathType, name: str | None = None, typ: str = "artifact"
    ) -> None:
        """Upload a file to the backends that store files.

        Args:
            path: Path to the file.
            name: Name to store the file under. ``None`` keeps the name
                of the file.
            typ: Kind of the artifact. Only WandB uses it.

        """
        for backend in self._live_backends():
            backend.upload_artifact(Path(path), name, typ)

    def close(self, status: str = "success") -> None:
        """End the run in each started backend.

        A backend that fails to close does not stop the others. It is
        reported instead. A second call does nothing, and the tracker
        ignores the logging calls that come after it.

        Args:
            status: ``"success"`` or ``"finished"`` for a run that
                succeeded. Any other value marks the run as failed.
                These are the values that a Lightning logger receives.

        """
        if self._closed:
            return
        self._closed = True
        final: RunStatus = (
            "success" if status in {"success", "finished"} else "failed"
        )
        for name, backend in self._started.items():
            try:
                backend.close(final)
            except Exception as error:
                logger.warning(f"Could not close the {name} run: {error}")

    def _live_backends(self) -> list[TrackerBackend]:
        if self._closed:
            if self.rank == 0:
                logger.warning(
                    "The tracker is closed. It ignores the logging call."
                )
            return []
        self._start_backends()
        return list(self._started.values())

    def _start_backends(self) -> None:
        if self.rank != 0:
            return
        for name, backend in self._backends.items():
            if name not in self._started:
                backend.start()
                self._started[name] = backend


def _legacy_backends(
    *,
    is_tensorboard: bool,
    is_wandb: bool,
    is_mlflow: bool,
    wandb_entity: str | None,
    mlflow_tracking_uri: str | None,
) -> dict[str, Mapping[str, object]]:
    """Turn the deprecated flags into backend options."""
    backends: dict[str, Mapping[str, object]] = {}
    if is_tensorboard:
        backends["tensorboard"] = {}
    if is_wandb:
        backends["wandb"] = {"entity": wandb_entity} if wandb_entity else {}
    if is_mlflow:
        backends["mlflow"] = (
            {"tracking_uri": mlflow_tracking_uri}
            if mlflow_tracking_uri
            else {}
        )
    if backends:
        replacement = ", ".join(
            f"{name}={dict(options) or True}"
            for name, options in backends.items()
        )
        warnings.warn(
            "The `is_tensorboard`, `is_wandb`, `is_mlflow`, `wandb_entity` "
            "and `mlflow_tracking_uri` arguments are deprecated. Use "
            f"`{replacement}` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return backends


def _create_backend(
    name: str, run: RunContext, options: Mapping[str, object]
) -> TrackerBackend:
    backend = TRACKER_BACKENDS.get(name)(run, **options)
    if backend.buffered:
        return BufferedBackend(backend, name)
    return backend


def _run_number(run_name: str) -> int | None:
    """Return the number of ``<number>-<name>``, or ``None``."""
    match = re.match(r"(\d+)(?:-|$)", run_name)
    return int(match[1]) if match else None


def _run_numbers(save_directory: Path) -> dict[str, int]:
    """Map each numbered run in ``save_directory`` to its number."""
    return {
        path.name: number
        for path in save_directory.iterdir()
        if (number := _run_number(path.name)) is not None and path.is_dir()
    }


def _new_run_name(save_directory: Path) -> str:
    number = max(_run_numbers(save_directory).values(), default=-1) + 1
    return f"{number}-{get_random_name(separator='-', style='lowercase')}"


def _join_run(save_directory: Path) -> str:
    """Return the run that rank :math:`0` created.

    A worker that inherits `RUN_NAME_ENV` from rank :math:`0` joins that
    run. The name is removed from the environment, because a later run
    of rank :math:`0` does not reach a worker that already runs.
    Otherwise, as with ``torchrun``, all ranks start together, and this
    waits for a new run directory to appear. After a short grace period
    it falls back to the newest existing run.

    Raises:
        RuntimeError: If no run directory exists after the timeout.

    """
    if name := os.environ.pop(RUN_NAME_ENV, None):
        return name
    known = set(_run_numbers(save_directory))
    start = time.monotonic()
    while True:
        runs = _run_numbers(save_directory)
        new_runs = {name: n for name, n in runs.items() if name not in known}
        if new_runs:
            return max(new_runs, key=new_runs.__getitem__)
        elapsed = time.monotonic() - start
        if runs and elapsed >= _JOIN_GRACE_PERIOD:
            logger.warning(
                f"No new run appeared in '{save_directory}'. Joining the "
                "newest run. Pass `run_name` to be sure that all ranks log "
                "to the same run."
            )
            return max(runs, key=runs.__getitem__)
        if elapsed >= _JOIN_TIMEOUT:
            raise RuntimeError(
                f"No run appeared in '{save_directory}' within "
                f"{_JOIN_TIMEOUT:.0f} seconds. Pass `run_name` to all ranks."
            )
        time.sleep(_JOIN_POLL_INTERVAL)
