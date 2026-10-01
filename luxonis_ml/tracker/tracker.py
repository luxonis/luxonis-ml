# pyright: strict
"""The tracker that logs a run to several tracking services at once.

`LuxonisTracker` sends each logging call to each backend that it
enables, names the run, and closes it. `RUN_NAME_ENV` hands a generated
run name to the other ranks of a distributed training.

See:
    `luxonis_ml.tracker` for a guide to the tracker, and
    `luxonis_ml.tracker.backends` for the backends.

"""

import atexit
import os
import re
import sys
import time
import warnings
from collections.abc import Iterator, Mapping
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
from .backends.mlflow import MLflowBackend, MLflowOptions
from .backends.wandb import WandbBackend, WandbOptions
from .buffer import BufferedBackend

_BackendT = TypeVar("_BackendT", bound=TrackerBackend)
"""The type of backend that `LuxonisTracker.get_backend` returns."""

RUN_NAME_ENV = "LUXONIS_TRACKER_RUN_NAME"
"""The environment variable that hands the run name to the other ranks.

Rank :math:`0` exports the run name that it generates. A worker process
that starts later inherits the variable and joins the same run.
"""

_JOIN_TIMEOUT = 30.0
"""Seconds that a worker waits for a run before it gives up."""

_JOIN_GRACE_PERIOD = 1.0
"""Seconds that a worker waits for a new run before it joins the newest
one.
"""

_JOIN_POLL_INTERVAL = 0.5
"""Seconds between two looks of a worker for a new run."""


class LuxonisTracker:
    """Log a run to several tracking services at once.

    The tracker sends each logging call to each enabled backend. Each
    backend in `TRACKER_BACKENDS`, a plugin backend included, has a
    keyword argument of its name:

    .. code-block:: python

        with LuxonisTracker(
            project_name="training",
            tensorboard=True,
            mlflow={"tracking_uri": "http://localhost:5000"},
        ) as tracker:
            tracker.log_metrics({"loss": 0.18}, step=1)

    Only rank :math:`0` logs. On the other ranks every logging call does
    nothing, and no backend starts.

    The backends start on the first logging call, or at `start`. `close`
    ends the run in each backend. Use the tracker as a context manager to
    close it with the right status. A run that is still open when the
    interpreter exits closes then, as failed after an uncaught error.

    Attributes:
        project_name: Project name, as the caller gave it.
        project_id: Project identifier, as the caller gave it. The
            identifiers that MLflow assigns are on
            ``tracker.get_backend(MLflowBackend)``.
        run_name: Name of the run.
        run_id: Identifier of an earlier run to continue, as the caller
            gave it.
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
            ValueError: If no backend is enabled, or a backend lacks an
                option that it needs, such as the MLflow tracking URI.
            TypeError: If a keyword argument names no backend in
                `TRACKER_BACKENDS`, or a backend gets an unknown option.
            RuntimeError: If ``run_name`` is omitted on a non-zero rank,
                and no run appears within 30 seconds.

        """
        configs, legacy_options = _legacy_backends(
            is_tensorboard=is_tensorboard,
            is_wandb=is_wandb,
            is_mlflow=is_mlflow,
            wandb_entity=wandb_entity,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        legacy_names = configs.keys() | legacy_options.keys()
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
                configs[name] = legacy_options.get(name, {})
            elif value is not None:
                configs[name] = value
        if not configs:
            raise ValueError("Enable at least one backend.")
        if legacy_names:
            _warn_deprecated(configs, legacy_names)

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
        self._status: RunStatus = "success"
        # an interactive session keeps the last error that it printed
        self._earlier_error = getattr(sys, "last_value", None)

        self.run_directory = run.run_directory
        self.run_directory.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Self:
        """Return the tracker, which `__exit__` closes.

        Returns:
            The tracker itself.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the run, as failed if the block raised.

        Args:
            exc_type: The type of the error that the block raised, or
                ``None``.
            exc_value: The error that the block raised, or ``None``.
            traceback: The traceback of the error, or ``None``.

        """
        self.close("success" if exc_type is None else "failed")

    @property
    def name(self) -> str:
        """The run name, the same as `run_name`.

        A Lightning logger reads it.
        """
        return self.run_name

    @property
    def version(self) -> int:
        """The number of the run, such as :math:`11` for ``11-foo``.

        It is :math:`0` for a run name without a number. A Lightning
        logger reads it.
        """
        return _run_number(self.run_name) or 0

    @property
    def backends(self) -> Mapping[str, TrackerBackend]:
        """The enabled backends, keyed by name.

        A backend that sets `TrackerBackend.buffered` is here in its
        `BufferedBackend`. The mapping is read-only.
        """
        return MappingProxyType(self._backends)

    def get_backend(self, backend_type: type[_BackendT]) -> _BackendT:
        """Return the enabled backend of a type.

        Unlike `backends`, it looks through a `BufferedBackend` to the
        backend that it wraps.

        Example:
            .. code-block:: python

                run_id = tracker.get_backend(MLflowBackend).run_id

        Args:
            backend_type: The class of the backend, such as
                `MLflowBackend`. A subclass of it matches too.

        Returns:
            The first enabled backend of ``backend_type``.

        Raises:
            KeyError: If no backend of ``backend_type`` is enabled.

        """
        for backend in self._backends.values():
            if isinstance(backend, BufferedBackend):
                backend = backend.backend
            if isinstance(backend, backend_type):
                return backend
        raise KeyError(f"No {backend_type.__name__} is enabled.")

    def start(self) -> None:
        """Start the backends now, not at the first logging call.

        For example, start the MLflow run of a sweep before its trials,
        so that they nest under it. A backend that started already does
        not start again. On a non-zero rank, and after `close`, it does
        nothing.

        Raises:
            Exception: The error of a backend that fails to start. A
                buffered backend raises only when the service rejects
                the start.

        """
        if not self._closed:
            self._start_backends()

    @property
    @deprecated("Use `'tensorboard' in tracker.backends` instead.")
    def is_tensorboard(self) -> bool:
        """Whether TensorBoard is enabled.

        Deprecated: use ``"tensorboard" in tracker.backends``.
        """
        return "tensorboard" in self._backends

    @property
    @deprecated("Use `'wandb' in tracker.backends` instead.")
    def is_wandb(self) -> bool:
        """Whether WandB is enabled.

        Deprecated: use ``"wandb" in tracker.backends``.
        """
        return "wandb" in self._backends

    @property
    @deprecated("Use `'mlflow' in tracker.backends` instead.")
    def is_mlflow(self) -> bool:
        """Whether MLflow is enabled.

        Deprecated: use ``"mlflow" in tracker.backends``.
        """
        return "mlflow" in self._backends

    @property
    @deprecated("Use `tracker.get_backend(WandbBackend).entity` instead.")
    def wandb_entity(self) -> str | None:
        """The WandB entity, or ``None`` without WandB.

        Deprecated: use ``tracker.get_backend(WandbBackend).entity``.
        """
        try:
            return self.get_backend(WandbBackend).entity
        except KeyError:
            return None

    @property
    @deprecated(
        "Use `tracker.get_backend(MLflowBackend).tracking_uri` instead."
    )
    def mlflow_tracking_uri(self) -> str | None:
        """The MLflow tracking URI, or ``None`` without MLflow.

        Deprecated: use ``tracker.get_backend(MLflowBackend).tracking_uri``.
        """
        try:
            return self.get_backend(MLflowBackend).tracking_uri
        except KeyError:
            return None

    def log_hyperparams(self, params: Mapping[str, ParamValue]) -> None:
        """Log the hyperparameters of the run.

        Each call adds to the hyperparameters of the earlier calls.

        Args:
            params: The hyperparameters, keyed by name. A value can be
                any value of a YAML configuration, such as a list.

        Raises:
            Exception: The error of a backend that fails to start or to
                log. A buffered backend raises only when the service
                rejects the start.

        """
        for backend in self._live_backends():
            backend.log_hyperparams(params)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one scalar metric.

        Args:
            name: Name of the metric.
            value: Value of the metric.
            step: The training step of the value.

        Raises:
            Exception: The error of a backend that fails to start or to
                log. A buffered backend raises only when the service
                rejects the start.

        """
        self.log_metrics({name: value}, step)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics.

        Args:
            metrics: The metric values, keyed by metric name.
            step: The training step of the values.

        Raises:
            Exception: The error of a backend that fails to start or to
                log. A buffered backend raises only when the service
                rejects the start.

        """
        for backend in self._live_backends():
            backend.log_metrics(metrics, step)

    def log_image(self, name: str, img: npt.NDArray[Any], step: int) -> None:
        r"""Log an image.

        Args:
            name: Name of the image. MLflow uses the part before the last
                ``/`` as the directory.
            img: The image, of shape :math:`\left(H, W, C\right)`.
            step: The training step of the image.

        Raises:
            Exception: The error of a backend that fails to start or to
                log. A buffered backend raises only when the service
                rejects the start.

        """
        for backend in self._live_backends():
            backend.log_image(name, img, step)

    def log_images(
        self, imgs: Mapping[str, npt.NDArray[Any]], step: int
    ) -> None:
        r"""Log several images.

        Args:
            imgs: The images, keyed by name. Each image has the shape
                :math:`\left(H, W, C\right)`.
            step: The training step of the images.

        Raises:
            Exception: The error of a backend that fails to start or to
                log. A buffered backend raises only when the service
                rejects the start.

        """
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
            step: The training step of the matrix.
            extra_data: More data to store with the matrix, such as the
                class names. Only MLflow stores it.

        Raises:
            Exception: The error of a backend that fails to start or to
                log. A buffered backend raises only when the service
                rejects the start.

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

        Raises:
            Exception: The error of a backend that fails to start or to
                log. A buffered backend raises only when the service
                rejects the start.

        """
        for backend in self._live_backends():
            backend.upload_artifact(Path(path), name, typ)

    def flush(self) -> None:
        """Write the pending data of each started backend, and keep the
        run open.

        For example, TensorBoard writes its events to disk. A backend
        that fails to flush does not stop the others. It is reported
        instead. After `close`, and on a non-zero rank, it does nothing.
        """
        if self._closed:
            return
        for name, backend in self._started.items():
            try:
                backend.flush()
            except Exception as error:
                logger.warning(f"Could not flush the {name} run: {error}")
        if self._started:
            # the TensorBoard flush opens a writer with an exit hook
            self._register_exit_hook()

    def close(self, status: str = "success") -> None:
        """End the run in each started backend.

        A backend that fails to close does not stop the others. It is
        reported instead. A second call does nothing, and the tracker
        ignores the logging calls that come after it.

        Args:
            status: ``"success"`` or ``"finished"`` for a run that
                succeeded. Any other value marks the run as failed.
                These are the values that a Lightning logger receives.

        Raises:
            BaseException: An interrupt, such as ``KeyboardInterrupt``,
                that stops the close of a backend. The tracker raises
                it after it closes the other backends.

        """
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._close_at_exit)
        self._status = (
            "success" if status in {"success", "finished"} else "failed"
        )
        interrupt: BaseException | None = None
        for name, backend in self._started.items():
            try:
                self._close_backend(name, backend)
            except BaseException as error:
                interrupt = interrupt or error
        if interrupt is not None:
            raise interrupt

    def _close_backend(self, name: str, backend: TrackerBackend) -> None:
        """End the run of one backend, and report a failure."""
        try:
            backend.close(self._status)
        except Exception as error:
            logger.warning(f"Could not close the {name} run: {error}")

    def _live_backends(self) -> Iterator[TrackerBackend]:
        """Yield the started backends, and start them if needed.

        A signal handler can close the run during a call. The backends
        after that point then do not get the call.
        """
        if self._closed:
            if self.rank == 0:
                logger.warning(
                    "The tracker is closed. It ignores the logging call."
                )
            return
        self._start_backends()
        for backend in list(self._started.values()):
            if self._closed:
                return
            yield backend

    def _start_backends(self) -> None:
        """Start each backend that did not start yet, on rank 0 only."""
        if self.rank != 0:
            return
        for name, backend in self._backends.items():
            if name in self._started:
                continue
            backend.start()
            if self._closed:
                # a signal handler closed the run during the start
                self._close_backend(name, backend)
                return
            self._started[name] = backend
            self._register_exit_hook()

    def _register_exit_hook(self) -> None:
        """Make `_close_at_exit` the last exit hook."""
        # the exit hooks run last first, so this one runs before the
        # hooks that a backend has just registered
        atexit.unregister(self._close_at_exit)
        atexit.register(self._close_at_exit)

    def _close_at_exit(self) -> None:
        """Close the run that is still open when the interpreter exits."""
        # the interpreter sets `last_value` when it prints an uncaught
        # error, and it runs the exit hooks after that
        failed = getattr(sys, "last_value", None) is not self._earlier_error
        self.close("failed" if failed else "success")


def _legacy_backends(
    *,
    is_tensorboard: bool,
    is_wandb: bool,
    is_mlflow: bool,
    wandb_entity: str | None,
    mlflow_tracking_uri: str | None,
) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]]]:
    """Turn the deprecated arguments into backend options.

    Returns:
        The backends that the flags turn on, and the options that
        ``wandb_entity`` and ``mlflow_tracking_uri`` give. The options
        also apply to a backend that its keyword turns on with ``True``.

    """
    options: dict[str, Mapping[str, object]] = {}
    if wandb_entity:
        options["wandb"] = {"entity": wandb_entity}
    if mlflow_tracking_uri:
        options["mlflow"] = {"tracking_uri": mlflow_tracking_uri}
    flags = {
        "tensorboard": is_tensorboard,
        "wandb": is_wandb,
        "mlflow": is_mlflow,
    }
    backends = {
        name: options.get(name, {}) for name, flag in flags.items() if flag
    }
    return backends, options


def _warn_deprecated(
    configs: Mapping[str, Mapping[str, object]], names: set[str]
) -> None:
    """Warn about the deprecated arguments, and name the backend
    keywords that replace them.

    The replacement holds each backend of ``names`` that stays on, with
    its final options. An option of a backend that stays off has no
    effect, so the replacement leaves it out.
    """
    replacement = ", ".join(
        f"{name}={dict(options) or True}"
        for name, options in configs.items()
        if name in names
    )
    advice = (
        f"Use `{replacement}` instead."
        if replacement
        else "Remove them, because they turn on no backend."
    )
    warnings.warn(
        "The `is_tensorboard`, `is_wandb`, `is_mlflow`, `wandb_entity` "
        f"and `mlflow_tracking_uri` arguments are deprecated. {advice}",
        DeprecationWarning,
        stacklevel=3,
    )


def _create_backend(
    name: str, run: RunContext, options: Mapping[str, object]
) -> TrackerBackend:
    """Create the backend of ``name``, in a buffer if it asks for one."""
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
    """Return ``<next number>-<random name>`` for a new run."""
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
