import json
import os
import time
from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from loguru import logger
from unique_names_generator import get_random_name

from luxonis_ml.typing import PathType
from luxonis_ml.utils.filesystem import LuxonisFileSystem


class LuxonisTracker:
    """Logger wrapper for `TensorBoard`_, `WandB`_, and `MLflow`_.

    `LuxonisTracker` stores run metadata, initializes the selected logging
    integrations lazily, and keeps a local fallback cache for MLflow logs
    that fail transiently.

    Attributes:
        project_name: Project name used by WandB and MLflow.
        project_id: Project identifier used by WandB and MLflow.
        save_directory: Root directory where local run outputs are stored.
        is_tensorboard: Whether TensorBoard logging is enabled.
        is_wandb: Whether WandB logging is enabled.
        is_mlflow: Whether MLflow logging is enabled.
        is_sweep: Whether the current run belongs to a sweep.
        rank: Process rank. Only rank :math:`0` writes through
            rank-gated logging methods.
        local_logs: Locally cached MLflow payloads that will be retried
            or written to disk on close.
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
                new name and other ranks use the latest run name.
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
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "2"

        self.project_name = project_name
        self.project_id = project_id
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.is_tensorboard = is_tensorboard
        self.is_wandb = is_wandb
        self.is_mlflow = is_mlflow
        self.is_sweep = is_sweep
        self.rank = rank
        self.local_logs = {
            "metric": [],
            "params": {},
            "images": [],
            "artifacts": [],
            "matrices": [],
            "metrics": [],
        }
        self.mlflow_initialized = False

        self.run_id = (
            run_id  # if using MLFlow then it will continue previous run
        )

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
        self.wandb_entity = wandb_entity
        if self.is_mlflow:
            if mlflow_tracking_uri is None:
                raise ValueError(
                    "Must specify mlflow_tracking_uri when using mlflow!"
                )
            self.mlflow_tracking_uri = mlflow_tracking_uri

        if not (self.is_tensorboard or self.is_wandb or self.is_mlflow):
            raise ValueError("At least one integration must be used!")

        self._experiment = None

        if run_name:
            self.run_name = run_name
        # create new directory if rank==0 else return newest run
        elif rank == 0:
            self.run_name = self._get_run_name()
        else:
            time.sleep(1)  # DDP hotfix
            self.run_name = self._get_latest_run_name()

        self.run_directory = self.save_directory / self.run_name
        self.run_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def rank_zero_only(fn: Callable) -> Callable:
        """Wrap a function so only processes with rank=0 execute it."""

        @wraps(fn)
        def wrapped_fn(
            self: "LuxonisTracker", *args: Any, **kwargs: Any
        ) -> Any | None:
            if self.rank == 0:
                return fn(self, *args, **kwargs)
            return None

        return wrapped_fn

    def log_to_mlflow(self, log_fn: Callable, *args, **kwargs) -> None:
        """Log to MLflow with retries.

        Logs locally if failures persist.
        """
        try:
            log_fn(*args, **kwargs)
            self.log_stored_logs_to_mlflow()  # Attempt to log stored logs after successful log
        except Exception as e:
            logger.warning(f"Attempt to log to MLflow failed: {e}")
        else:
            return

        self.store_log_locally(log_fn, *args, **kwargs)

    def store_log_locally(self, log_fn: Callable, *args, **kwargs) -> None:
        """Store log data locally if logging to MLflow fails."""
        # Checking functions without triggering reconnections.
        if log_fn == self.experiment["mlflow"].log_metric:
            self.local_logs["metric"].append(
                {"name": args[0], "value": args[1], "step": args[2]}
            )
        elif log_fn == self.experiment["mlflow"].log_metrics:
            self.local_logs["metrics"].append(
                {"metrics": args[0], "step": args[1]}
            )
        elif log_fn == self.experiment["mlflow"].log_params:
            self.local_logs["params"].update(args[0])
        elif log_fn == self.experiment["mlflow"].log_image:
            self.local_logs["images"].append(
                {"image_data": args[0], "name": args[1]}
            )
        elif log_fn == self.upload_artifact_to_mlflow:
            self.local_logs["artifacts"].append(
                {"path": str(args[0]), "name": args[1]}
            )
        elif log_fn == self.experiment["mlflow"].log_dict:
            self.local_logs["matrices"].append(
                {"matrix": args[0], "name": args[1]}
            )

    def log_stored_logs_to_mlflow(self) -> None:
        """Log any data stored in local_logs to MLflow."""
        if not self.mlflow_initialized or not any(self.local_logs.values()):
            return

        try:
            if self.local_logs["params"]:
                self.experiment["mlflow"].log_params(self.local_logs["params"])
                self.local_logs["params"] = {}
            for metric in list(self.local_logs["metric"]):
                self.experiment["mlflow"].log_metric(
                    metric["name"], metric["value"], metric["step"]
                )
                self.local_logs["metric"].remove(metric)
            for metrics in list(self.local_logs["metrics"]):
                self.experiment["mlflow"].log_metrics(
                    metrics["metrics"], metrics["step"]
                )
                self.local_logs["metrics"].remove(metrics)
            for image in list(self.local_logs["images"]):
                self.experiment["mlflow"].log_image(
                    image["image_data"], image["name"]
                )
                self.local_logs["images"].remove(image)
            for matrix in list(self.local_logs["matrices"]):
                self.experiment["mlflow"].log_dict(
                    matrix["matrix"], matrix["name"]
                )
                self.local_logs["matrices"].remove(matrix)
            for artifact in list(self.local_logs["artifacts"]):
                self.upload_artifact_to_mlflow(
                    Path(artifact["path"]), artifact["name"]
                )
                self.local_logs["artifacts"].remove(artifact)

            logger.info("Successfully re-logged stored logs to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to re-log stored logs to MLflow: {e}")

    def save_logs_locally(self) -> None:
        """Save metrics, parameters, images, artifacts, and matrices locally."""
        image_dir = self.run_directory / "images"
        artifact_dir = self.run_directory / "artifacts"

        image_dir.mkdir(exist_ok=True)
        artifact_dir.mkdir(exist_ok=True)

        for idx, img in enumerate(self.local_logs["images"]):
            img_path = str(image_dir / f"{idx}.png")
            cv2.imwrite(
                img_path, cv2.cvtColor(img["image_data"], cv2.COLOR_RGB2BGR)
            )
            img["image_data"] = img_path  # Replace data with path

        for artifact in self.local_logs["artifacts"]:
            artifact_path = Path(artifact["path"])
            if artifact_path.exists():
                local_path = artifact_dir / artifact_path.name
                local_path.write_bytes(artifact_path.read_bytes())
                artifact["path"] = str(local_path)

        log_dir = self.run_directory / "local_logs.json"
        with open(log_dir, "w") as f:
            json.dump(
                {
                    k: self.local_logs[k]
                    for k in [
                        "metrics",
                        "metric",
                        "params",
                        "images",
                        "artifacts",
                        "matrices",
                    ]
                },
                f,
            )

        logger.info(
            f"Logs saved locally at '{log_dir}', "
            f"images in {image_dir}, artifacts in {artifact_dir}"
        )

    @property
    def name(self) -> str:
        """Run name.

        Returns:
            Current run name.

        """
        return self.run_name

    @property
    def version(self) -> int:
        """Tracker version.

        Returns:
            Version number :math:`1`.

        """
        return 1

    @property
    @rank_zero_only
    def experiment(
        self,
    ) -> dict[Literal["tensorboard", "wandb", "mlflow"], Any]:
        """Creates new experiments or returns active ones if already
        created.
        """
        if self._experiment is None:
            self._experiment = {}

        if self.is_tensorboard and "tensorboard" not in self._experiment:
            from torch.utils.tensorboard.writer import SummaryWriter

            log_dir = self.save_directory / "tensorboard_logs" / self.run_name
            if self.is_sweep:
                trial_id = 0
                if log_dir.exists():
                    trial_id = (
                        max(
                            (
                                int(f.split("_")[-1])
                                for f in os.listdir(log_dir)  # noqa: PTH208
                                if f.startswith("trial_")
                            ),
                            default=0,
                        )
                        + 1
                    )
                log_dir = log_dir / f"trial_{trial_id}"

            self._experiment["tensorboard"] = SummaryWriter(log_dir=log_dir)

        if self.is_wandb and "wandb" not in self._experiment:
            import wandb

            self._experiment["wandb"] = wandb

            log_dir = self.save_directory / "wandb_logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            self._experiment["wandb"].init(
                project=self.project_name
                if self.project_name is not None
                else self.project_id,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
            )

        if self.is_mlflow and self.mlflow_initialized is False:
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

                self._experiment["mlflow"] = mlflow

                self.artifacts_dir = self.run_directory / "artifacts"
                self.artifacts_dir.mkdir(parents=True, exist_ok=True)

                self._experiment["mlflow"].set_tracking_uri(
                    self.mlflow_tracking_uri
                )

                if self.project_id is not None:
                    self.project_name = None

                experiment = self._experiment["mlflow"].set_experiment(
                    experiment_name=self.project_name,
                    experiment_id=self.project_id,
                )
                self.project_id = experiment.experiment_id

                # If self.run_id is None, create a new run; otherwise, use the existing one
                run = self._experiment["mlflow"].start_run(
                    run_id=self.run_id,
                    run_name=self.run_name,
                    nested=self.is_sweep,
                )
                self.run_id = run.info.run_id
                self.mlflow_initialized = (
                    True  # Mark MLflow as initialized successfully
                )

            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
                self.mlflow_initialized = False  # Mark MLflow as unavailable

        return self._experiment

    @rank_zero_only
    def log_hyperparams(
        self, params: dict[str, str | bool | int | float | None]
    ) -> None:
        """Log a hyperparameter dictionary.

        Args:
            params: Hyperparameter key-value pairs.

        """
        if self.is_tensorboard:
            self.experiment["tensorboard"].add_hparams(
                params,
                {
                    "placeholder_metric": 0
                },  # placeholder metric is needed due to this issue: https://github.com/tensorflow/tensorboard/issues/5476
            )
        if self.is_wandb:
            self.experiment["wandb"].config.update(params)
        if self.is_mlflow:
            self.log_to_mlflow(self.experiment["mlflow"].log_params, params)

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
            self.experiment["wandb"].log({name: value})

        if self.is_mlflow:
            self.log_to_mlflow(
                self.experiment["mlflow"].log_metric, name, value, step
            )

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
            self.experiment["wandb"].log(metrics)
        if self.is_mlflow:
            self.log_to_mlflow(
                self.experiment["mlflow"].log_metrics, metrics, step
            )

    @rank_zero_only
    def log_image(self, name: str, img: np.ndarray, step: int) -> None:
        r"""Log one image.

        Note:
            ``step`` is omitted when logging with WandB to avoid problems
            with inconsistent incrementation.

        Args:
            name: Image caption. For MLflow, this should include a
                slash-separated base path and image caption.
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
            self.experiment["wandb"].log({name: wandb_image})

        if self.is_mlflow:
            # split images into separate directories based on step
            base_path, img_caption = name.rsplit("/", 1)
            img_path = f"{base_path}/{step}/{img_caption}.png"
            self.log_to_mlflow(
                self.experiment["mlflow"].log_image, img, img_path
            )

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
            name: Artifact name. If ``None``, uses the file stem for WandB
                and the file name for MLflow.
            typ: The type of the artifact. Only used for WandB.

        """
        path = Path(path)
        if self.is_wandb:
            import wandb

            artifact = wandb.Artifact(name=name or path.stem, type=typ)
            artifact.add_file(local_path=str(path))
            artifact.save()

        if self.is_mlflow:
            self.log_to_mlflow(self.upload_artifact_to_mlflow, path, name)

    def upload_artifact_to_mlflow(
        self,
        path: PathType,
        name: str | None = None,
    ) -> None:
        """Upload an artifact specifically to MLflow.

        Args:
            path: Path to the artifact.
            name: Artifact name. If ``None``, uses the file name.

        """
        fs = LuxonisFileSystem(
            "mlflow://",
            allow_active_mlflow_run=True,
            allow_local=False,
        )
        fs.put_file(
            local_path=path,
            remote_path=name or path.name,
            mlflow_instance=self.experiment.get("mlflow"),
        )

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
            self.log_to_mlflow(
                self.experiment["mlflow"].log_dict,
                matrix_data,
                f"{name}.json",
            )

        if self.is_tensorboard:
            matrix_str = np.array2string(matrix, separator=", ")
            self.experiment["tensorboard"].add_text(name, matrix_str, step)

        if self.is_wandb:
            import wandb

            table = wandb.Table(
                columns=["Row Index"]
                + [f"Col {i}" for i in range(matrix.shape[1])]
            )
            for i, row in enumerate(matrix):
                table.add_data(i, *row)
            self.experiment["wandb"].log({f"{name}_table": table}, step=step)

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

    def _get_next_run_number(self) -> int:
        """Return the number ID for the next run."""

        log_dirs = [
            path.name
            for path in self.save_directory.iterdir()
            if path.is_dir()
        ]

        nums = [path.split("-")[0] for path in log_dirs]
        nums = [int(num) for num in nums if num.isnumeric()]

        if len(nums) == 0:
            return 0
        return max(nums) + 1

    def close(self) -> None:
        """Finalize logging and save unsent logs locally."""
        if self.is_mlflow and any(self.local_logs.values()):
            self.save_logs_locally()

    def _get_run_name(self) -> str:
        """Generate a new run name."""
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def _get_latest_run_name(self) -> str:
        """Return the most recently created run name."""
        log_dirs = [
            path.relative_to(self.save_directory).name
            for path in self.save_directory.iterdir()
            if path.is_dir()
        ]
        runs = []
        for ld in log_dirs:
            if ld.split("-")[0].isnumeric():
                runs.append(ld)
        runs.sort(
            key=lambda x: (self.save_directory / x).stat().st_mtime,
            reverse=True,
        )
        return runs[0]
