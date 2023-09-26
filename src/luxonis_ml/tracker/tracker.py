#!/usr/bin/env python3

import os
from pathlib import Path
import glob
from unique_names_generator import get_random_name
from typing import Optional, Union

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import Logger as plLogger


class LuxonisTracker:
    def __init__(
        self,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        hyperparameter_config: Optional[dict] = None,
        save_directory: str = "runs",
        is_tensorboard: bool = False,
        is_wandb: bool = False,
        is_mlflow: bool = False,
        is_sweep: bool = False,
        wandb_entity: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """LuxonisTracker that wraps various logging software. Supported loggers: TensorBoard, WandB and MLFlow

        Args:
            project_name (Optional[str], optional): Name of the project used for WandB and MLFlow. Defaults to None.
            project_id (Optional[str], optional): Project id used for WandB and MLFlow. Defaults to None.
            run_name (Optional[str], optional): Name of the run, if None then auto-generate random name. Defaults to None.
            run_id (Optional[str], optional): Run id used for continuing MLFlow run. Defaults to None.
            hyperparameter_config (Optional[dict], optional): Dict of hyperparameters to log. Defaults to None.
            save_directory (str, optional): Path to save directory. Defaults to "runs".
            is_tensorboard (bool, optional): Weather use TensorBoard logging. Defaults to False.
            is_wandb (bool, optional): Weather use WandB logging. Defaults to False.
            is_mlflow (bool, optional): Weather use MLFlow logging. Defaults to False.
            is_sweep (bool, optional): Weather current run is part of a sweep. Defaults to False.
            wandb_entity (Optional[str], optional): WandB entity to use. Defaults to None.
            mlflow_tracking_uri (Optional[str], optional): MLFlow tracking uri to use. Defaults to None.

        """

        self.project_name = project_name
        self.project_id = project_id
        self.save_directory = save_directory
        self.is_tensorboard = is_tensorboard
        self.is_wandb = is_wandb
        self.is_mlflow = is_mlflow
        self.config = hyperparameter_config

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

        self.run_name = self._get_run_name() if run_name is None else run_name
        Path(f"{self.save_directory}/{self.run_name}").mkdir(
            parents=True, exist_ok=True
        )

        if self.is_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = f"{save_directory}/tensorboard_logs/{self.run_name}"
            self.tensorboard_logger = SummaryWriter(log_dir=log_dir)
            self.tensorboard_logger.add_hparams(
                self.config,
                {
                    "placeholder_metric": 0
                },  # placeholder metric is needed due to this issue: https://github.com/tensorflow/tensorboard/issues/5476
            )

        if self.is_wandb:
            import wandb

            self.wandb = wandb

            log_dir = f"{save_directory}/wandb_logs"
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            self.wandb.init(
                project=self.project_name
                if self.project_name != None
                else self.project_id,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
                config=self.config,
            )

        if self.is_mlflow:
            import mlflow

            self.mlflow = mlflow

            self.artifacts_dir = f"{self.save_directory}/{self.run_name}/artifacts"
            Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

            self.mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            if self.project_id is not None:
                self.project_name = None
            self.mlflow.set_experiment(
                experiment_name=self.project_name, experiment_id=self.project_id
            )

            # if self.run_id == None then create new run, else use alredy created one
            self.mlflow.start_run(
                run_id=self.run_id, run_name=self.run_name, nested=self.is_sweep
            )
            self.mlflow.log_params(self.config)

    def _get_next_run_number(self):
        """Returns number id for next run"""
        # find all directories that should be runs
        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path.split("/")[-1] for path in log_dirs if os.path.isdir(path)]
        # find the numbers based on the naming convention
        nums = [path.split("-")[0] for path in log_dirs]
        nums = [int(num) for num in nums if num.isnumeric()]

        if len(nums) == 0:
            return 0
        else:
            return max(nums) + 1

    def _get_run_name(self):
        """Generates new run name"""
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def log_metric(self, name, value, step):
        """Logs metric value with provided name and step"""
        if self.is_tensorboard:
            self.tensorboard_logger.add_scalar(name, value, step)

        if self.is_wandb:
            self.wandb.log({name: value, "epoch": step})

        if self.is_mlflow:
            self.mlflow.log_metric(name, value, step)

    def log_image(self, name, img, step):
        """Logs image with provided name and step"""
        if self.is_tensorboard:
            self.tensorboard_logger.add_image(name, img, dataformats="HWC")

        if self.is_wandb:
            wandb_image = self.wandb.Image(img, caption=name)
            self.wandb.log({name: wandb_image})

        if self.is_mlflow:
            self.mlflow.log_image(img, f"{name}_{step}.png")


class LuxonisTrackerPL(plLogger):
    def __init__(
        self,
        project_name: Optional[str] = None,
        project_id: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        save_directory: str = "runs",
        is_tensorboard: bool = False,
        is_wandb: bool = False,
        is_mlflow: bool = False,
        is_sweep: bool = False,
        wandb_entity: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        rank: int = 0,
    ):
        """Implementation of PytorchLightning Logger that wraps various logging software. Supported loggers: TensorBoard, WandB and MLFlow

        Args:
            project_name (Optional[str], optional): Name of the project used for WandB and MLFlow. Defaults to None.
            project_id (Optional[str], optional): Project id used for WandB and MLFlow. Defaults to None.
            run_name (Optional[str], optional): Name of the run, if None then auto-generate random name. Defaults to None.
            run_id (Optional[str], optional): Run id used for continuing MLFlow run. Defaults to None.
            save_directory (str, optional): Path to save directory. Defaults to "runs".
            is_tensorboard (bool, optional): Weather use TensorBoard logging. Defaults to False.
            is_wandb (bool, optional): Weather use WandB logging. Defaults to False.
            is_mlflow (bool, optional): Weather use MLFlow logging. Defaults to False.
            is_sweep (bool, optional): Weather current run is part of a sweep. Defaults to False.
            wandb_entity (Optional[str], optional): WandB entity to use. Defaults to None.
            mlflow_tracking_uri (Optional[str], optional): MLFlow tracking uri to use. Defaults to None.
            rank (int, optional): Rank of the instance, used when running on multiple threads. Defaults to 0.
        """
        plLogger.__init__(self)

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

    @property
    def name(self):
        """Returns run name"""
        return self.run_name

    @property
    def version(self):
        """Returns tracker's version"""
        return 1

    @property
    @rank_zero_only
    def experiment(self):
        """Creates new experiments or returns active ones if already created"""
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
                if self.project_name != None
                else self.project_id,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
                # config=self.config # TODO: this also?
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

    def _get_next_run_number(self):
        """Returns number id for next run"""
        # find all directories that should be runs
        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path.split("/")[-1] for path in log_dirs if os.path.isdir(path)]
        # find the numbers based on the naming convention
        nums = [path.split("-")[0] for path in log_dirs]
        nums = [int(num) for num in nums if num.isnumeric()]

        if len(nums) == 0:
            return 0
        else:
            return max(nums) + 1

    def _get_run_name(self):
        """Generates new run name"""
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def _get_latest_run_name(self):
        """Returns most recently created run name"""
        # find all directories that should be runs
        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path for path in log_dirs if os.path.isdir(path)]
        # find run names based on the naming convention and sort them by last modified time
        runs = []
        for l in log_dirs:
            l = l.replace(f"{self.save_directory}/", "")
            if l.split("-")[0].isnumeric():
                runs.append(l)
        runs.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.save_directory, x)),
            reverse=True,
        )
        return runs[0]

    @rank_zero_only
    def log_hyperparams(self, params: dict):
        """Log hypeparameter dictionary

        Args:
            params (dict): Dict of hyperparameters key and value pairs
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
    def log_metrics(self, metrics: dict, step: int):
        """Log metric dictionary

        Args:
            metrics (dict): Dict of metric key and value pairs
            step (int): Current step
        """
        if self.is_tensorboard:
            for key, value in metrics.items():
                self.experiment["tensorboard"].add_scalar(key, value, step)
        if self.is_wandb:
            # if step is added here it doesn't work correctly with wandb
            self.experiment["wandb"].log(metrics)
        if self.is_mlflow:
            self.experiment["mlflow"].log_metrics(metrics, step)

    @rank_zero_only
    def log_images(self, imgs: dict, step: int):
        """Log imgs dictionary

        Args:
            imgs (dict): Dict of image key and value pairs where key is image caption and value is img data
            step (int): Current step
        """
        for caption, img in imgs.items():
            self.log_image(caption, img, step)

    @rank_zero_only
    def log_image(
        self, caption: str, img: Union["torch.Tensor", "np.ndarray"], step: int
    ):
        """Log one image with a caption

        Args:
            caption (str): Image caption
            img (Union[torch.Tensor, numpy.ndarray]): Image data
            step (int): Current step
        """

        if self.is_tensorboard:
            self.experiment["tensorboard"].add_image(
                caption, img, step, dataformats="HWC"
            )

        if self.is_wandb:
            wandb_image = self._experiment["wandb"].Image(img, caption=caption)
            # if step is added here it doesn't work correctly with wandb
            self._experiment["wandb"].log({caption: wandb_image})

        if self.is_mlflow:
            # split images into separate directories based on step
            base_path, img_caption = caption.rsplit("/", 1)
            self._experiment["mlflow"].log_image(
                img, f"{base_path}/{step}/{img_caption}.png"
            )

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
