import json
import os
import time
from functools import wraps
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

import cv2
import numpy as np
from loguru import logger
from unique_names_generator import get_random_name

from luxonis_ml.typing import PathType
from luxonis_ml.utils.filesystem import LuxonisFileSystem


class LuxonisTracker:
    def __init__(
        self,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        save_directory: PathType = "output",
        is_tensorboard: bool = False,
        is_wandb: bool = False,
        is_mlflow: bool = False,
        is_sweep: bool = False,
        wandb_entity: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        rank: int = 0,
    ):
        """Implementation of PytorchLightning Logger that wraps various
        logging software. Supported loggers: TensorBoard, WandB and
        MLFlow.

        @type project_name: Optional[str]
        @param project_name: Name of the project used for WandB and MLFlow.
            Defaults to None.

        @type project_id: Optional[str]
        @param project_id: Project id used for WandB and MLFlow.
            Defaults to None.

        @type run_name: Optional[str]
        @param run_name: Name of the run, if None then auto-generate random name.
            Defaults to None.

        @type run_id: Optional[str]
        @param run_id: Run id used for continuing MLFlow run.
            Defaults to None.

        @type save_directory: str
        @param save_directory: Path to save directory.
            Defaults to "output".

        @type is_tensorboard: bool
        @param is_tensorboard: Wheter use TensorBoard logging.
            Defaults to False.

        @type is_wandb: bool
        @param is_wandb: Wheter use WandB logging.
            Defaults to False.

        @type is_mlflow: bool
        @param is_mlflow: Wheter use MLFlow logging.
            Defaults to False.

        @type is_sweep: bool
        @param is_sweep: Wheter current run is part of a sweep.
            Defaults to False.

        @type wandb_entity: Optional[str]
        @param wandb_entity: WandB entity to use.
            Defaults to None.

        @type mlflow_tracking_uri: Optional[str]
        @param mlflow_tracking_uri: MLFlow tracking uri to use.
            Defaults to None.

        @type rank: int
        @param rank: Rank of the process, used when running on multiple threads.
            Defaults to 0.
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
        """Function wrapper that lets only processes with rank=0 execute
        it."""

        @wraps(fn)
        def wrapped_fn(
            self: "LuxonisTracker", *args: Any, **kwargs: Any
        ) -> Optional[Any]:
            if self.rank == 0:
                return fn(self, *args, **kwargs)
            return None

        return wrapped_fn

    def log_to_mlflow(self, log_fn: Callable, *args, **kwargs) -> None:
        """Attempts to log to MLflow, with retries.

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
        """Stores log data locally if logging to MLflow fails."""
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
        elif log_fn == self.upload_artifact:
            self.local_logs["artifacts"].append(
                {"path": str(args[0]), "name": args[1], "type": args[2]}
            )
        elif log_fn == self.experiment["mlflow"].log_dict:
            self.local_logs["matrices"].append(
                {"matrix": args[0], "name": args[1]}
            )

    def log_stored_logs_to_mlflow(self) -> None:
        """Attempts to log any data stored in local_logs to MLflow."""
        if not self.mlflow_initialized or not any(self.local_logs.values()):
            return

        try:
            for metric in self.local_logs["metric"]:
                self.experiment["mlflow"].log_metric(
                    metric["name"], metric["value"], metric["step"]
                )
            for metrics in self.local_logs["metrics"]:
                self.experiment["mlflow"].log_metrics(
                    metrics["metrics"], metrics["step"]
                )
            if self.local_logs["params"]:
                self.experiment["mlflow"].log_params(self.local_logs["params"])
            for image in self.local_logs["images"]:
                self.experiment["mlflow"].log_image(
                    image["image_data"], image["name"]
                )
            for artifact in self.local_logs["artifacts"]:
                self.upload_artifact(
                    Path(artifact["path"]), artifact["name"], artifact["type"]
                )
            for matrix in self.local_logs["matrices"]:
                self.experiment["mlflow"].log_dict(
                    matrix["matrix"], matrix["name"]
                )

            self.local_logs = {
                "metrics": [],
                "params": {},
                "images": [],
                "artifacts": [],
                "matrices": [],
                "metric": [],
            }
            logger.info("Successfully re-logged stored logs to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to re-log stored logs to MLflow: {e}")

    def save_logs_locally(self) -> None:
        """Saves metrics, parameters, images, artifacts, and matrices
        locally."""
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
        """Returns run name.

        @type: str
        """
        return self.run_name

    @property
    def version(self) -> int:
        """Returns tracker's version.

        @type: int
        """
        return 1

    @property
    @rank_zero_only
    def experiment(
        self,
    ) -> Dict[Literal["tensorboard", "wandb", "mlflow"], Any]:
        """Creates new experiments or returns active ones if already
        created."""
        if self._experiment is None:
            self._experiment = {}

        if self.is_tensorboard and "tensorboard" not in self._experiment:
            from torch.utils.tensorboard.writer import SummaryWriter

            log_dir = self.save_directory / "tensorboard_logs" / self.run_name
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
        self, params: Dict[str, Union[str, bool, int, float, None]]
    ) -> None:
        """Logs hyperparameter dictionary.

        @type params: Dict[str, Union[str, bool, int, float, None]]
        @param params: Dict of hyperparameters key-value pairs.
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
        """Logs metric value with name and step.

        @note: step is ommited when logging with wandb to avoid problems
            with inconsistent incrementation.
        @type name: str
        @param name: Metric name
        @type value: float
        @param value: Metric value
        @type step: int
        @param step: Current step
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
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Logs metric dictionary.

        @type metrics: Dict[str, float]
        @param metrics: Dict of metric key-value pairs
        @type step: int
        @param step: Current step
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
        """Logs image with name and step. Note: step is omitted when
        logging with wandb is used to avoid problems with inconsistent
        incrementation.

        @type name: str
        @param name: Caption of the image
        @type img: np.ndarray
        @param img: Image data
        @type step: int
        @param step: Current step
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
        self, path: PathType, name: Optional[str] = None, typ: str = "artifact"
    ) -> None:
        """Uploads artifact to the logging service.

        @type path: PathType
        @param path: Path to the artifact
        @type name: Optional[str]
        @param name: Name of the artifact, if None then use the name of
            the file
        @type typ: str
        @param typ: Type of the artifact, defaults to "artifact". Only
            used for WandB.
        """
        path = Path(path)
        if self.is_wandb:
            import wandb

            artifact = wandb.Artifact(name=name or path.stem, type=typ)
            artifact.add_file(local_path=str(path))
            artifact.save()

        if self.is_mlflow:
            try:
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
            except Exception as e:
                logger.warning(f"Failed to upload artifact to MLflow: {e}")
                self.store_log_locally(
                    self.upload_artifact, path, name, typ
                )  # Stores details for retrying later
                self.log_stored_logs_to_mlflow()

    @rank_zero_only
    def log_matrix(self, matrix: np.ndarray, name: str, step: int) -> None:
        """Logs matrix to the logging service.

        @type matrix: np.ndarray
        @param matrix: The matrix to log.
        @type name: str
        @param name: The name used to log the matrix.
        @type step: int
        @param step: The current step.
        """
        if self.is_mlflow:
            matrix_data = {
                "flat_array": matrix.flatten().tolist(),
                "shape": matrix.shape,
            }
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
    def log_images(self, imgs: Dict[str, np.ndarray], step: int) -> None:
        """Logs multiple images.

        @type imgs: Dict[str, np.ndarray]
        @param imgs: Dict of image key-value pairs where key is image
            caption and value is image data
        @type step: int
        @param step: Current step
        """
        for caption, img in imgs.items():
            self.log_image(caption, img, step)

    def _get_next_run_number(self) -> int:
        """Returns number id for next run."""

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
        """Finalizes logging and saves unsent logs locally."""
        if self.is_mlflow and any(self.local_logs.values()):
            self.save_logs_locally()

    def _get_run_name(self) -> str:
        """Generates new run name."""
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def _get_latest_run_name(self) -> str:
        """Returns most recently created run name."""
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
