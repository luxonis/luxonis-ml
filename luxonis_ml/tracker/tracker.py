import json
import os
import re
import shutil
import time
from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
    Any,
    Concatenate,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeVar,
)

import cv2
import numpy as np
from loguru import logger
from typing_extensions import Self
from unique_names_generator import get_random_name

from luxonis_ml.typing import PathType

P = ParamSpec("P")
T = TypeVar("T")

Backend = Literal["tensorboard", "wandb", "wandb_run", "mlflow"]

MLflowCall = Literal[
    "log_metric",
    "log_metrics",
    "log_params",
    "log_image",
    "log_dict",
    "log_artifact",
]

MAX_BUFFERED_LOGS = 500
"""Maximum number of MLflow calls buffered during an outage."""

MAX_BUFFERED_IMAGES = 50
"""Maximum number of buffered images.

Images dominate the memory footprint of the buffer, so they are capped
more aggressively than the other calls.
"""

MLFLOW_RETRY_INTERVAL = 60.0
"""Seconds to wait before retrying a failed MLflow initialization."""

_LOG_GROUPS: dict[MLflowCall, tuple[str, tuple[str, ...]]] = {
    "log_metric": ("metric", ("name", "value", "step")),
    "log_metrics": ("metrics", ("metrics", "step")),
    "log_params": ("params", ("params",)),
    "log_image": ("images", ("image_data", "name")),
    "log_dict": ("matrices", ("matrix", "name")),
    "log_artifact": ("artifacts", ("path", "name")),
}
"""Maps an MLflow call to its group and field names in
``local_logs.json``."""

_IRREPLACEABLE_CALLS: frozenset[MLflowCall] = frozenset(
    {"log_params", "log_artifact"}
)
"""Calls whose data no later call repeats.

Metrics and images are logged again on the next step, hyperparameters and
artifacts are not, so they are the last to be dropped from a full buffer.
"""


class BufferedCall(NamedTuple):
    """An MLflow call that could not be sent and is kept for a later
    retry.
    """

    kind: MLflowCall
    args: tuple[Any, ...]


def _json_default(obj: Any) -> Any:
    """Serialize the values ``json`` does not know, such as NumPy
    scalars.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class LuxonisTracker:
    """Logger wrapper for `TensorBoard`_, `WandB`_, and `MLflow`_.

    `LuxonisTracker` stores run metadata, initializes the selected logging
    integrations lazily, and keeps a local fallback buffer for MLflow logs
    that fail transiently.

    Attributes:
        project_name: Project name used by WandB and MLflow.
        project_id: Project identifier used by WandB and MLflow. Set to
            the resolved experiment ID once MLflow is initialized.
        save_directory: Root directory where local run outputs are stored.
        is_tensorboard: Whether TensorBoard logging is enabled.
        is_wandb: Whether WandB logging is enabled.
        is_mlflow: Whether MLflow logging is enabled.
        is_sweep: Whether the current run belongs to a sweep.
        rank: Process rank. Only rank :math:`0` writes through
            rank-gated logging methods.
        local_logs: MLflow calls that failed and will be retried or
            written to disk on close.
        mlflow_initialized: Whether MLflow initialization has succeeded.
        run_id: MLflow run identifier, used to resume an existing run.
        wandb_entity: WandB entity used for logging.
        mlflow_tracking_uri: MLflow tracking URI used when MLflow logging
            is enabled.
        run_name: Name of the current run.
        run_directory: Directory for local run artifacts.

    .. _TensorBoard:
        https://www.tensorflow.org/tensorboard
    .. _WandB:
        https://wandb.ai/site
    .. _MLflow:
        https://mlflow.org/

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
    ):
        """Create a tracker for one or more logging integrations.

        Args:
            project_name: Project name used for WandB and MLflow.
            project_id: Project ID used for WandB and MLflow.
            run_name: Run name. If omitted, rank :math:`0` generates a
                new name and other ranks wait for it to appear.
            run_id: MLflow run ID used to continue a previous run.
            save_directory: Directory where local outputs are saved.
            is_tensorboard: Whether to use TensorBoard logging.
            is_wandb: Whether to use WandB logging.
            is_mlflow: Whether to use MLflow logging.
            is_sweep: Whether the current run is part of a sweep.
            wandb_entity: WandB entity to use.
            mlflow_tracking_uri: MLflow tracking URI to use.
            rank: Process rank used in distributed training.

        Raises:
            ValueError: If WandB or MLflow is enabled but neither
                `project_name` nor `project_id` is provided.
            ValueError: If WandB is enabled without `wandb_entity`.
            ValueError: If MLflow is enabled without
                `mlflow_tracking_uri`.
            ValueError: If no logging integration is enabled.

        """
        self.project_name = project_name
        self.project_id = project_id
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.is_tensorboard = is_tensorboard
        self.is_wandb = is_wandb
        self.is_mlflow = is_mlflow
        self.is_sweep = is_sweep
        self.rank = rank
        self.wandb_entity = wandb_entity
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_initialized = False
        self.local_logs: list[BufferedCall] = []

        # if using MLflow then it will continue previous run
        self.run_id = run_id

        if (
            (is_wandb or is_mlflow)
            and self.project_name is None
            and self.project_id is None
        ):
            raise ValueError(
                "Either project_name or project_id must be specified!"
            )

        if self.is_wandb and wandb_entity is None:
            raise ValueError("Must specify wandb_entity when using wandb!")

        if self.is_mlflow:
            if mlflow_tracking_uri is None:
                raise ValueError(
                    "Must specify mlflow_tracking_uri when using mlflow!"
                )
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

        if not (self.is_tensorboard or self.is_wandb or self.is_mlflow):
            raise ValueError("At least one integration must be used!")

        self._experiment: dict[Backend, Any] = {}
        self._mlflow_retry_at = 0.0
        self._closed = False
        self._buffered_artifacts = 0
        self._warned_buffer_limits: set[str] = set()

        if run_name:
            self.run_name = run_name
        # create new directory if rank==0 else wait for the newest run
        elif rank == 0:
            self.run_name = self._get_run_name()
        else:
            self.run_name = self._get_latest_run_name()

        self.run_directory = self.save_directory / self.run_name
        self.run_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def rank_zero_only(
        fn: Callable[Concatenate["LuxonisTracker", P], T],
    ) -> Callable[Concatenate["LuxonisTracker", P], T | None]:
        """Wrap a method so only processes with rank=0 execute it."""

        @wraps(fn)
        def wrapped_fn(
            self: "LuxonisTracker", *args: P.args, **kwargs: P.kwargs
        ) -> T | None:
            if self.rank == 0:
                return fn(self, *args, **kwargs)
            return None

        return wrapped_fn

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
        """Run name.

        Returns:
            Current run name.

        """
        return self.run_name

    @property
    def version(self) -> int:
        """Run number.

        Returns:
            The number prefix of the run name, or :math:`0` if the run
            name was provided by the user and carries no number.

        """
        number = self.run_name.split("-")[0]
        # `isdecimal`, unlike `isnumeric`, accepts exactly what `int` does
        return int(number) if number.isdecimal() else 0

    @property
    def experiment(self) -> dict[Backend, Any]:
        """Creates new experiments or returns active ones if already
        created.

        Returns:
            Mapping of the enabled backends to their handles. Empty for
            processes with a non-zero rank, which never log. After
            `close` the already-created handles are returned as they are,
            without starting anything new.

        """
        if self.rank != 0:
            return {}

        if self._closed:
            # re-initializing here would start a run that nothing ends,
            # as `close` refuses to run a second time
            return self._experiment

        if self.is_tensorboard and "tensorboard" not in self._experiment:
            self._init_tensorboard()

        if self.is_wandb and "wandb" not in self._experiment:
            self._init_wandb()

        if self.is_mlflow and not self.mlflow_initialized:
            self._init_mlflow()

        return self._experiment

    def _init_tensorboard(self) -> None:
        # torch is an optional backend dependency and not part of `[dev]`
        from torch.utils.tensorboard.writer import (  # pyright: ignore[reportMissingImports]
            SummaryWriter,
        )

        log_dir = self.save_directory / "tensorboard_logs" / self.run_name
        if self.is_sweep:
            trial_id = 0
            if log_dir.exists():
                trial_id = (
                    max(
                        (
                            int(path.name.split("_")[-1])
                            for path in log_dir.iterdir()
                            if path.name.startswith("trial_")
                        ),
                        default=0,
                    )
                    + 1
                )
            log_dir = log_dir / f"trial_{trial_id}"

        self._experiment["tensorboard"] = SummaryWriter(log_dir=log_dir)

    def _init_wandb(self) -> None:
        # wandb is an optional backend dependency and not part of `[dev]`
        import wandb  # pyright: ignore[reportMissingImports]

        log_dir = self.save_directory / "wandb_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        run = wandb.init(
            project=self.project_name
            if self.project_name is not None
            else self.project_id,
            entity=self.wandb_entity,
            dir=log_dir,
            name=self.run_name,
        )

        # both keys are stored only once `init` succeeded, otherwise a
        # failed initialization would look like a finished one and would
        # never be retried
        self._experiment["wandb"] = wandb
        self._experiment["wandb_run"] = run

    def _init_mlflow(self) -> None:
        """Initialize MLflow, backing off after a failed attempt.

        A failed attempt is not retried for `MLFLOW_RETRY_INTERVAL`
        seconds so that an unreachable server cannot stall the training
        loop with a full reconnect on every logged value.
        """
        if time.monotonic() < self._mlflow_retry_at:
            return

        try:
            import mlflow

            if find_spec("psutil") is not None:
                mlflow.enable_system_metrics_logging()
                if find_spec("pynvml") is None:
                    logger.warning(
                        "pynvml not found, GPU stats will not be monitored. "
                        "To enable GPU monitoring, install it using 'pip install pynvml'"
                    )
            else:
                logger.warning(
                    "`psutil` not found. To enable system metric logging, "
                    "install it using 'pip install psutil'"
                )

            # guaranteed by the constructor whenever MLflow is enabled
            assert self.mlflow_tracking_uri is not None
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            experiment = mlflow.set_experiment(
                experiment_name=self.project_name
                if self.project_id is None
                else None,
                experiment_id=self.project_id,
            )
            self.project_id = experiment.experiment_id

            # if self.run_id is None a new run is created,
            # otherwise the existing one is resumed
            run = mlflow.start_run(
                run_id=self.run_id,
                run_name=self.run_name,
                nested=self.is_sweep,
            )
            self.run_id = run.info.run_id
            # stored only once the run exists, so that a failed attempt
            # does not leave a handle behind
            self._experiment["mlflow"] = mlflow
            self.mlflow_initialized = True

        except Exception as e:
            self._mlflow_retry_at = time.monotonic() + MLFLOW_RETRY_INTERVAL
            logger.warning(
                f"Failed to initialize MLflow: {e}. Logs will be buffered "
                f"locally, the connection is retried in "
                f"{MLFLOW_RETRY_INTERVAL:.0f} seconds."
            )

    def _mlflow_call(self, kind: MLflowCall, *args: Any) -> None:
        """Send a call to MLflow, buffering it if MLflow is unavailable.

        The positional arguments are what gets buffered, so they have to
        be JSON-serializable.
        """
        # only MLflow is initialized here, the other backends have
        # nothing to do with this call
        if not self.mlflow_initialized and not self._closed:
            self._init_mlflow()

        call = BufferedCall(kind, args)

        if not self.mlflow_initialized:
            self._buffer(call)
            return

        if self.local_logs:
            # replay the buffered calls first so the order is preserved
            self._buffer(call)
            self.log_stored_logs_to_mlflow()
            return

        try:
            self._perform_mlflow_call(self._experiment["mlflow"], call)
        except Exception as e:
            logger.warning(f"Attempt to log to MLflow failed: {e}")
            self._buffer(call)

    @staticmethod
    def _perform_mlflow_call(mlflow: ModuleType, call: BufferedCall) -> None:
        if call.kind == "log_artifact":
            # the artifact name is only kept for the local fallback,
            # MLflow always stores artifacts under their file name
            mlflow.log_artifact(call.args[0])
        else:
            getattr(mlflow, call.kind)(*call.args)

    def _buffer(self, call: BufferedCall) -> None:
        """Buffer a call that could not be sent to MLflow."""
        if call.kind == "log_artifact":
            call = self._snapshot_artifact(call)
        self.local_logs.append(call)
        self._enforce_buffer_limit("log_image", MAX_BUFFERED_IMAGES)
        self._enforce_buffer_limit(None, MAX_BUFFERED_LOGS)

    def _snapshot_artifact(self, call: BufferedCall) -> BufferedCall:
        """Copy a buffered artifact into the run directory.

        Callers routinely delete an artifact right after handing it over,
        so it is copied before it can disappear. Every artifact gets its
        own sub-directory, which keeps its file name intact for MLflow and
        keeps artifacts that share a file name apart. The returned call
        points at the copy, or is the original one if the copy failed.
        """
        source = Path(call.args[0])
        target = (
            self.run_directory
            / "artifacts"
            / str(self._buffered_artifacts)
            / source.name
        )
        # claimed even if the copy fails, so that no two artifacts can
        # ever end up in the same directory
        self._buffered_artifacts += 1

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            copy = shutil.copy2(source, target)
        except OSError as e:
            logger.warning(
                f"Could not copy the artifact '{source}' to '{target}', it "
                f"will be lost if the file disappears: {e}"
            )
            return call

        return BufferedCall(call.kind, (str(copy), *call.args[1:]))

    def _enforce_buffer_limit(
        self, kind: MLflowCall | None, limit: int
    ) -> None:
        """Drop the oldest expendable buffered call once ``limit`` is
        exceeded.

        Hyperparameters and artifacts are only dropped when the buffer
        holds nothing else, because no later call repeats them.
        """
        indices = [
            i
            for i, call in enumerate(self.local_logs)
            if kind is None or call.kind == kind
        ]
        if len(indices) <= limit:
            return

        expendable = [
            i
            for i in indices
            if self.local_logs[i].kind not in _IRREPLACEABLE_CALLS
        ]
        dropped = self.local_logs.pop((expendable or indices)[0])

        # the buffer stays full for the rest of the outage, so warning
        # per dropped call would bury every other message
        warned = kind or "all"
        if warned in self._warned_buffer_limits:
            return
        self._warned_buffer_limits.add(warned)
        logger.warning(
            f"The MLflow log buffer is full ({limit} entries), dropping the "
            f"oldest '{dropped.kind}' call. Logs are buffered locally "
            "because MLflow is unreachable. Further drops are not reported."
        )

    def log_stored_logs_to_mlflow(self) -> None:
        """Replay the locally buffered calls to MLflow, oldest first.

        Replaying stops at the first failure, leaving the remaining calls
        buffered so that their original order is preserved. A call MLflow
        rejects while it still accepts the call behind it is dropped
        instead, because it would otherwise block every later log for the
        rest of the run.
        """
        if not self.mlflow_initialized or not self.local_logs:
            return

        mlflow = self._experiment["mlflow"]
        while self.local_logs:
            call = self.local_logs[0]
            try:
                self._perform_mlflow_call(mlflow, call)
            except Exception as e:
                if not self._replay_next_call(mlflow):
                    logger.warning(
                        f"Failed to re-log stored logs to MLflow: {e}"
                    )
                    return
                logger.warning(
                    f"MLflow rejected the buffered '{call.kind}' call while "
                    f"accepting later ones, dropping it: {e}"
                )
            # the probe already removed the call it sent, so the one that
            # was just handled is still the head
            self.local_logs.pop(0)

        logger.info("Successfully re-logged stored logs to MLflow.")

    def _replay_next_call(self, mlflow: ModuleType) -> bool:
        """Send the call queued behind a failing one.

        This tells a call MLflow rejects apart from an MLflow that is
        unreachable, without reordering the calls that survive. Returns
        whether the call went through, and removes it from the buffer
        when it did.
        """
        if len(self.local_logs) < 2:
            return False

        try:
            self._perform_mlflow_call(mlflow, self.local_logs[1])
        except Exception:
            return False

        self.local_logs.pop(1)
        return True

    def save_logs_locally(self) -> None:
        """Save buffered metrics, parameters, images, artifacts, and
        matrices to the run directory.

        Anything an earlier call already saved is kept, and the buffer is
        cleared afterwards so that repeated calls neither duplicate nor
        discard the saved data.
        """
        log_path = self.run_directory / "local_logs.json"
        grouped = self._load_saved_logs(log_path)
        # the earlier records are already on disk and keep their index
        first_image = len(grouped["images"])
        self._group_local_logs(grouped)

        image_dir = self.run_directory / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, image in enumerate(
            grouped["images"][first_image:], start=first_image
        ):
            # replace the image data with the path it was written to
            image["image_data"] = self._save_image_locally(
                image_dir, idx, image["name"], image["image_data"]
            )

        with open(log_path, "w") as f:
            json.dump(grouped, f, default=_json_default)

        self.local_logs.clear()

        logger.info(
            f"Logs saved locally at '{log_path}', images in {image_dir}, "
            f"artifacts in {self.run_directory / 'artifacts'}"
        )

    @staticmethod
    def _empty_log_groups() -> dict[str, Any]:
        """Return the empty group structure of ``local_logs.json``."""
        return {
            "metrics": [],
            "metric": [],
            "params": {},
            "images": [],
            "artifacts": [],
            "matrices": [],
        }

    def _load_saved_logs(self, log_path: Path) -> dict[str, Any]:
        """Load an earlier local save so a later one adds to it."""
        grouped = self._empty_log_groups()
        if not log_path.exists():
            return grouped

        try:
            saved = json.loads(log_path.read_text())
        except (OSError, ValueError) as e:
            logger.warning(
                f"Could not read the logs saved at '{log_path}', "
                f"they will be overwritten: {e}"
            )
            return grouped

        for group in grouped:
            if group in saved:
                grouped[group] = saved[group]
        return grouped

    def _group_local_logs(
        self, grouped: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Group the buffered calls by kind for local serialization."""
        grouped = grouped if grouped is not None else self._empty_log_groups()
        for call in self.local_logs:
            group, fields = _LOG_GROUPS[call.kind]
            record = dict(zip(fields, call.args, strict=True))
            if group == "params":
                grouped["params"].update(record["params"])
            else:
                grouped[group].append(record)
        return grouped

    @staticmethod
    def _save_image_locally(
        image_dir: Path, idx: int, name: str, image: np.ndarray
    ) -> str | None:
        """Write a buffered image to ``image_dir``.

        Returns the path it was written to, or ``None`` if the image
        could not be encoded.
        """
        stem = re.sub(r"\W+", "_", name.removesuffix(".png")).strip("_")
        path = image_dir / f"{idx}_{stem or 'image'}.png"

        try:
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(path), image)
        except Exception as e:
            logger.warning(f"Failed to save image '{name}' locally: {e}")
            return None
        return str(path)

    @rank_zero_only
    def log_hyperparams(
        self, params: dict[str, str | bool | int | float | None]
    ) -> None:
        """Log a hyperparameter dictionary.

        Note:
            TensorBoard only accepts scalar hyperparameters, so values of
            any other type are converted to strings.

        Args:
            params: Hyperparameter key-value pairs.

        """
        if self.is_tensorboard:
            self.experiment["tensorboard"].add_hparams(
                {
                    key: value
                    if isinstance(value, (bool, int, float, str))
                    else str(value)
                    for key, value in params.items()
                },
                # placeholder metric is needed due to this issue:
                # https://github.com/tensorflow/tensorboard/issues/5476
                {"placeholder_metric": 0},
            )
        if self.is_wandb:
            self.experiment["wandb_run"].config.update(params)
        if self.is_mlflow:
            self._mlflow_call("log_params", params)

    @rank_zero_only
    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one scalar metric value.

        Note:
            ``step`` is omitted when logging with WandB to avoid problems
            with inconsistent incrementation.

        Args:
            name: Metric name.
            value: Metric value.
            step: Current step.

        """
        if self.is_tensorboard:
            self.experiment["tensorboard"].add_scalar(name, value, step)

        if self.is_wandb:
            # let wandb increment step to avoid calls with inconsistent steps
            self.experiment["wandb_run"].log({name: value})

        if self.is_mlflow:
            self._mlflow_call("log_metric", name, value, step)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple scalar metrics.

        Args:
            metrics: Metric key-value pairs.
            step: Current step.

        """
        if self.is_tensorboard:
            for key, value in metrics.items():
                self.experiment["tensorboard"].add_scalar(key, value, step)
        if self.is_wandb:
            self.experiment["wandb_run"].log(metrics)
        if self.is_mlflow:
            self._mlflow_call("log_metrics", metrics, step)

    @rank_zero_only
    def log_image(self, name: str, img: np.ndarray, step: int) -> None:
        r"""Log one image.

        Note:
            ``step`` is omitted when logging with WandB to avoid problems
            with inconsistent incrementation.

        Args:
            name: Image caption. For MLflow, a slash-separated prefix of
                the name is used as the artifact directory.
            img: Image data of shape :math:`\left(H, W, C\right)`.
            step: Current step.

        """
        if self.is_tensorboard:
            self.experiment["tensorboard"].add_image(
                name, img, step, dataformats="HWC"
            )

        if self.is_wandb:
            wandb_image = self.experiment["wandb"].Image(img, caption=name)
            # if step is added here it doesn't work correctly with wandb
            self.experiment["wandb_run"].log({name: wandb_image})

        if self.is_mlflow:
            # split images into separate directories based on step
            base_path, _, caption = name.rpartition("/")
            img_path = f"{step}/{caption}.png"
            if base_path:
                img_path = f"{base_path}/{img_path}"
            self._mlflow_call("log_image", img, img_path)

    @rank_zero_only
    def log_images(self, imgs: dict[str, np.ndarray], step: int) -> None:
        r"""Log multiple images.

        Args:
            imgs: Mapping from image captions to image data of shape
                :math:`\left(H, W, C\right)`.
            step: Current step.

        """
        for caption, img in imgs.items():
            self.log_image(caption, img, step)

    @rank_zero_only
    def log_matrix(
        self,
        matrix: np.ndarray,
        name: str,
        step: int,
        extra_data: dict | None = None,
    ) -> None:
        r"""Log a matrix to the enabled logging services.

        Args:
            matrix: Matrix to log, usually of shape
                :math:`\left(M, N\right)`.
            name: Name used for the matrix artifact.
            step: Current step.
            extra_data: Optional dictionary of additional data to include
                in the logged matrix artifact.

        """
        if self.is_mlflow:
            matrix_data: dict = {
                "flat_array": matrix.flatten().tolist(),
                "shape": matrix.shape,
            }
            if extra_data is not None:
                matrix_data.update(extra_data)
            self._mlflow_call("log_dict", matrix_data, f"{name}.json")

        if self.is_tensorboard:
            matrix_str = np.array2string(
                matrix, separator=", ", threshold=matrix.size
            )
            self.experiment["tensorboard"].add_text(name, matrix_str, step)

        if self.is_wandb:
            table = self.experiment["wandb"].Table(
                columns=[
                    "Row Index",
                    *[f"Col {i}" for i in range(matrix.shape[1])],
                ]
            )
            for i, row in enumerate(matrix):
                table.add_data(i, *row)
            # let wandb increment step to avoid calls with inconsistent steps
            self.experiment["wandb_run"].log({f"{name}_table": table})

    @rank_zero_only
    def upload_artifact(
        self,
        path: PathType,
        name: str | None = None,
        typ: str = "artifact",
    ) -> None:
        """Upload an artifact to the logging service.

        Args:
            path: Path to the artifact.
            name: Artifact name. If ``None``, uses the file stem for
                WandB. MLflow always stores artifacts under their file
                name, so there the name is only used as a label if the
                artifact has to be saved locally instead.
            typ: The type of the artifact. Only used for WandB.

        """
        path = Path(path)
        if self.is_wandb:
            artifact = self.experiment["wandb"].Artifact(
                name=name or path.stem, type=typ
            )
            artifact.add_file(local_path=str(path))
            self.experiment["wandb_run"].log_artifact(artifact)

        if self.is_mlflow:
            self.upload_artifact_to_mlflow(path, name)

    @rank_zero_only
    def upload_artifact_to_mlflow(
        self,
        path: PathType,
        name: str | None = None,
    ) -> None:
        """Upload an artifact specifically to MLflow.

        Only rank :math:`0` uploads, like every other logging method.
        Non-zero ranks never finalize the run, so an artifact handed over
        by one of them could neither be sent nor saved.

        Args:
            path: Path to the artifact.
            name: Artifact name. Only used as a label if the artifact has
                to be saved locally instead.

        """
        self._mlflow_call("log_artifact", str(path), name)

    @rank_zero_only
    def close(self, status: str = "success") -> None:
        """Finalize the run and shut the enabled backends down.

        Buffered MLflow logs are flushed to the server if it became
        reachable again, and saved locally otherwise. Calling this more
        than once is a no-op.

        Args:
            status: Final run status. Anything other than ``"success"``
                or ``"finished"`` marks the run as failed.

        """
        if self._closed:
            return
        self._closed = True
        succeeded = status in {"success", "finished"}

        if self.is_mlflow:
            if self.local_logs and not self.mlflow_initialized:
                # give an MLflow server that came back up a last chance,
                # for which the backoff has to be waived
                self._mlflow_retry_at = 0.0
                self._init_mlflow()
            self.log_stored_logs_to_mlflow()
            if self.local_logs:
                self._finalize_backend(
                    "the local log fallback", self.save_logs_locally
                )

        if "tensorboard" in self._experiment:
            # `SummaryWriter.close` flushes the pending events first
            self._finalize_backend(
                "TensorBoard", self._experiment["tensorboard"].close
            )

        if self.mlflow_initialized:
            self._finalize_backend(
                "MLflow",
                self._experiment["mlflow"].end_run,
                "FINISHED" if succeeded else "FAILED",
            )

        if "wandb_run" in self._experiment:
            self._finalize_backend(
                "WandB",
                self._experiment["wandb_run"].finish,
                0 if succeeded else 1,
            )

    @staticmethod
    def _finalize_backend(
        name: str, finalize: Callable[..., Any], *args: Any
    ) -> None:
        """Shut a backend down without letting it break the teardown."""
        try:
            finalize(*args)
        except Exception as e:
            logger.warning(f"Failed to finalize {name}: {e}")

    def _get_run_name(self) -> str:
        """Generate a new run name."""
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def _get_next_run_number(self) -> int:
        """Return the number ID for the next run."""
        numbers = [int(name.split("-")[0]) for name in self._existing_runs()]
        if not numbers:
            return 0
        return max(numbers) + 1

    def _get_latest_run_name(
        self,
        timeout: float = 30.0,
        poll_interval: float = 0.5,
        grace_period: float = 1.0,
    ) -> str:
        """Return the name of the run the rank-zero process created.

        Waits for a run directory that was not there yet, because a run
        left over from an earlier training would otherwise be mistaken
        for the current one. Rank zero cannot be identified with any
        certainty from here, so once ``grace_period`` has passed without a
        new run appearing this falls back to the newest existing one. It
        raises ``RuntimeError`` if no run directory shows up within
        ``timeout``.
        """
        known = set(self._existing_runs())
        start = time.monotonic()
        while True:
            runs = self._existing_runs()
            new_runs = [name for name in runs if name not in known]
            if new_runs:
                return self._newest_run(new_runs)

            elapsed = time.monotonic() - start
            if runs and elapsed >= grace_period:
                logger.warning(
                    f"No new run directory appeared in "
                    f"'{self.save_directory}' within {grace_period} seconds, "
                    f"falling back to the newest existing run. Pass "
                    f"'run_name' explicitly to make sure all ranks use the "
                    f"same run."
                )
                return self._newest_run(runs)

            if elapsed >= timeout:
                raise RuntimeError(
                    f"No run directory found in '{self.save_directory}' "
                    f"after {timeout} seconds. Either the rank-zero process "
                    "did not start, or 'run_name' has to be passed explicitly."
                )
            time.sleep(poll_interval)

    @staticmethod
    def _newest_run(runs: list[str]) -> str:
        """Return the run with the highest number."""
        return max(runs, key=lambda name: int(name.split("-")[0]))

    def _existing_runs(self) -> list[str]:
        """Return the names of all numbered runs in the save
        directory.
        """
        return [
            path.name
            for path in self.save_directory.iterdir()
            if path.is_dir() and path.name.split("-")[0].isdecimal()
        ]
