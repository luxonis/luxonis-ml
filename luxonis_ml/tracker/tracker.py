import glob
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from unique_names_generator import get_random_name


class LuxonisTracker:
    def __init__(
        self,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        save_directory: str = "output",
        is_tensorboard: bool = False,
        is_wandb: bool = False,
        is_mlflow: bool = False,
        is_sweep: bool = False,
        wandb_entity: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        rank: int = 0,
    ):
        """Implementation of PytorchLightning Logger that wraps various logging
        software. Supported loggers: TensorBoard, WandB and MLFlow.

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
        self.project_name = project_name
        self.project_id = project_id
        self.save_directory = save_directory
        self.is_tensorboard = is_tensorboard
        self.is_wandb = is_wandb
        self.is_mlflow = is_mlflow
        self.is_sweep = is_sweep
        self.rank = rank

        self.run_id = run_id  # if using MLFlow then it will continue previous run

        if is_wandb or is_mlflow:
            if self.project_name is None and self.project_id is None:
                raise Exception("Either project_name or project_id must be specified!")

        if self.is_wandb and wandb_entity is None:
            raise Exception("Must specify wandb_entity when using wandb!")
        else:
            self.wandb_entity = wandb_entity
        if self.is_mlflow and mlflow_tracking_uri is None:
            raise Exception("Must specify mlflow_tracking_uri when using mlflow!")
        else:
            self.mlflow_tracking_uri = mlflow_tracking_uri

        if not (self.is_tensorboard or self.is_wandb or self.is_mlflow):
            raise Exception("At least one integration must be used!")

        self._experiment = None

        if run_name:
            self.run_name = run_name
        else:
            # create new directory if rank==0 else return newest run
            if rank == 0:
                self.run_name = self._get_run_name()
            else:
                self.run_name = self._get_latest_run_name()

        Path(f"{self.save_directory}/{self.run_name}").mkdir(
            parents=True, exist_ok=True
        )

    @staticmethod
    def rank_zero_only(fn: Callable) -> Callable:
        """Function wrapper that lets only processes with rank=0 execute it."""

        @wraps(fn)
        def wrapped_fn(self, *args: Any, **kwargs: Any) -> Optional[Any]:
            if self.rank == 0:
                return fn(self, *args, **kwargs)
            return None

        return wrapped_fn

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
    def experiment(self) -> Dict[str, Any]:
        """Creates new experiments or returns active ones if already created."""
        if self._experiment is not None:
            return self._experiment

        self._experiment = {}

        if self.is_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = f"{self.save_directory}/tensorboard_logs/{self.run_name}"
            self._experiment["tensorboard"] = SummaryWriter(log_dir=log_dir)

        if self.is_wandb:
            import wandb

            self._experiment["wandb"] = wandb

            log_dir = f"{self.save_directory}/wandb_logs"
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            self._experiment["wandb"].init(
                project=self.project_name
                if self.project_name is not None
                else self.project_id,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
            )

        if self.is_mlflow:
            import mlflow

            self._experiment["mlflow"] = mlflow

            self.artifacts_dir = f"{self.save_directory}/{self.run_name}/artifacts"
            Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

            self._experiment["mlflow"].set_tracking_uri(self.mlflow_tracking_uri)

            if self.project_id is not None:
                self.project_name = None
            experiment = self._experiment["mlflow"].set_experiment(
                experiment_name=self.project_name, experiment_id=self.project_id
            )
            self.project_id = experiment.experiment_id

            # if self.run_id == None then create new run, else use alredy created one
            run = self._experiment["mlflow"].start_run(
                run_id=self.run_id, run_name=self.run_name, nested=self.is_sweep
            )
            self.run_id = run.info.run_id

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
            self.experiment["mlflow"].log_params(params)

    @rank_zero_only
    def log_metric(self, name: str, value: float, step: int) -> None:
        """Logs metric value with name and step.

        @note: step is ommited when logging with wandb to avoid problems with
            inconsistent incrementation.
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
            self.experiment["mlflow"].log_metric(name, value, step)

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
            self.experiment["mlflow"].log_metrics(metrics, step)

    @rank_zero_only
    def log_image(self, name: str, img: np.ndarray, step: int) -> None:
        """Logs image with name and step. Note: step is omitted when logging with wandb
        is used to avoid problems with inconsistent incrementation.

        @type name: str
        @param name: Caption of the image
        @type img: np.ndarray
        @param img: Image data
        @type step: int
        @param step: Current step
        """
        if self.is_tensorboard:
            self.experiment["tensorboard"].add_image(name, img, step, dataformats="HWC")

        if self.is_wandb:
            wandb_image = self._experiment["wandb"].Image(img, caption=name)
            # if step is added here it doesn't work correctly with wandb
            self._experiment["wandb"].log({name: wandb_image})

        if self.is_mlflow:
            # split images into separate directories based on step
            base_path, img_caption = name.rsplit("/", 1)
            self._experiment["mlflow"].log_image(
                img, f"{base_path}/{step}/{img_caption}.png"
            )

    @rank_zero_only
    def log_images(self, imgs: Dict[str, np.ndarray], step: int) -> None:
        """Logs multiple images.

        @type imgs: Dict[str, np.ndarray]
        @param imgs: Dict of image key-value pairs where key is image caption and value
            is image data
        @type step: int
        @param step: Current step
        """
        for caption, img in imgs.items():
            self.log_image(caption, img, step)

    def _get_next_run_number(self) -> int:
        """Returns number id for next run."""

        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path.split("/")[-1] for path in log_dirs if os.path.isdir(path)]

        nums = [path.split("-")[0] for path in log_dirs]
        nums = [int(num) for num in nums if num.isnumeric()]

        if len(nums) == 0:
            return 0
        else:
            return max(nums) + 1

    def _get_run_name(self) -> str:
        """Generates new run name."""
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def _get_latest_run_name(self) -> str:
        """Returns most recently created run name."""
        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path for path in log_dirs if os.path.isdir(path)]
        runs = []
        for ld in log_dirs:
            ld = ld.replace(f"{self.save_directory}/", "")
            if ld.split("-")[0].isnumeric():
                runs.append(ld)
        runs.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.save_directory, x)),
            reverse=True,
        )
        return runs[0]
