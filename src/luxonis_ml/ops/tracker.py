#!/usr/bin/env python3

import os
from pathlib import Path
import glob
from unique_names_generator import get_random_name
from PIL import Image
import cv2
import numpy as np

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import Logger as plLogger


class LuxonisTracker:
    """
    Wrapper for various logging software. Support for
        - TensorBoard
        - WandB
        - MLFlow
    """

    def __init__(self,
                 project_name,
                 run_name=None,
                 hyperparameter_config=None,
                 save_directory='runs',
                 is_tensorboard=False,
                 is_wandb=False,
                 is_mlflow=False,
                 is_sweep=False,
                 wandb_entity=None,
                 mlflow_tracking_uri=None):

        self.project_name = project_name
        self.save_directory = save_directory
        self.is_tensorboard = is_tensorboard
        self.is_wandb = is_wandb
        self.is_mlflow = is_mlflow
        self.config = hyperparameter_config

        if self.is_wandb and wandb_entity is None:
            raise Exception("Must specify wandb_entity when using wandb!")
        else:
            self.wandb_entity = wandb_entity
        if self.is_mlflow and mlflow_tracking_uri is None:
            raise Exception("Must specify mlflow_tracking_uri when using mlflow!")
        else:
            self.mlflow_tracking_uri = mlflow_tracking_uri

        if not (
            self.is_tensorboard or \
            self.is_wandb or \
            self.is_mlflow
        ):
            raise Exception("At least one integration must be used!")

        self.run_name = self._get_run_name() if run_name is None else run_name
        Path(f"{self.save_directory}/{self.run_name}").mkdir(parents=True, exist_ok=True)

        if self.is_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir=f"{save_directory}/tensorboard_logs/{self.run_name}"
            self.tensorboard_logger = SummaryWriter(
                log_dir=log_dir
            )
            self.tensorboard_logger.add_hparams(
                self.config,
                {'placeholder_metric': 0}, # placeholder metric is needed due to this issue: https://github.com/tensorflow/tensorboard/issues/5476
            )

        if self.is_wandb:
            import wandb
            self.wandb = wandb

            log_dir = f"{save_directory}/wandb_logs"
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            self.wandb.init(
                project=project_name,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
                config=self.config
            )

        if self.is_mlflow:
            import mlflow
            self.mlflow = mlflow

            self.artifacts_dir = f"{self.save_directory}/{self.run_name}/artifacts"
            Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

            self.mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.mlflow.set_experiment(self.project_name)
            self.mlflow.start_run(
                run_name=self.run_name, nested=is_sweep
            )
            self.mlflow.log_params(self.config)

    def _get_next_run_number(self):
        # find all directories that should be runs
        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path.split("/")[-1] for path in log_dirs if os.path.isdir(path)]
        # find the numbers based on the naming convention
        nums = [path.split('-')[0] for path in log_dirs]
        nums = [int(num) for num in nums if num.isnumeric()]

        if len(nums) == 0:
            return 0
        else:
            return max(nums)+1

    def _get_run_name(self):
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def log_metric(self, name, value, step):
        """
        name: name of the metric to log
        value: value of the metric
        step: epoch number
        """

        if self.is_tensorboard:
            self.tensorboard_logger.add_scalar(name, value, step)

        if self.is_wandb:
            self.wandb.log({name: value, "epoch": step})

        if self.is_mlflow:
            self.mlflow.log_metric(name, value, step)

    def log_image(self, name, img, step):
        """
        img: Image in RGB order, uint8, HWC
        """

        if self.is_tensorboard:
            self.tensorboard_logger.add_image(name, img, dataformats='HWC')

        if self.is_wandb:
            wandb_image = self.wandb.Image(img, caption=name)
            self.wandb.log({name: wandb_image})

        if self.is_mlflow:
            self.mlflow.log_image(
                img, f"{name}_{step}.png"
            )

class LuxonisTrackerPL(plLogger):
    def __init__(self,
                 project_name,
                 run_name=None,
                 save_directory='runs',
                 is_tensorboard=False,
                 is_wandb=False,
                 is_mlflow=False,
                 is_sweep=False,
                 wandb_entity=None,
                 mlflow_tracking_uri=None,
                 rank=0):             
        
        plLogger.__init__(self)

        self.project_name = project_name
        self.save_directory = save_directory
        self.is_tensorboard = is_tensorboard
        self.is_wandb = is_wandb
        self.is_mlflow = is_mlflow
        self.is_sweep = is_sweep
        self.rank = rank

        if self.is_wandb and wandb_entity is None:
            raise Exception("Must specify wandb_entity when using wandb!")
        else:
            self.wandb_entity = wandb_entity
        if self.is_mlflow and mlflow_tracking_uri is None:
            raise Exception("Must specify mlflow_tracking_uri when using mlflow!")
        else:
            self.mlflow_tracking_uri = mlflow_tracking_uri

        if not (
            self.is_tensorboard or \
            self.is_wandb or \
            self.is_mlflow
        ):
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

        Path(f"{self.save_directory}/{self.run_name}").mkdir(parents=True, exist_ok=True)


    @property
    def name(self):
        return self.run_name

    @property
    def version(self):
        return 1

    @property
    @rank_zero_only
    def experiment(self):
        if self._experiment is not None:
            return self._experiment
        
        self._experiment = {}

        if self.is_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir=f"{self.save_directory}/tensorboard_logs/{self.run_name}"
            self._experiment["tensorboard"] = SummaryWriter(
                log_dir=log_dir
            )

        if self.is_wandb:
            import wandb
            self._experiment["wandb"] = wandb

            log_dir = f"{self.save_directory}/wandb_logs"
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            self._experiment["wandb"].init(
                project=self.project_name,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
                #config=self.config # TODO: this also?
            )

        if self.is_mlflow:
            import mlflow
            self._experiment["mlflow"] = mlflow

            self.artifacts_dir = f"{self.save_directory}/{self.run_name}/artifacts"
            Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

            self._experiment["mlflow"].set_tracking_uri("https://ml-track.luxonis.com")
            self._experiment["mlflow"].set_experiment(self.project_name)
            self._experiment["mlflow"].start_run(
                run_name=self.run_name, nested=self.is_sweep
            )
        
        return self._experiment
    
    def _get_next_run_number(self):
        # find all directories that should be runs
        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path.split("/")[-1] for path in log_dirs if os.path.isdir(path)]
        # find the numbers based on the naming convention
        nums = [path.split('-')[0] for path in log_dirs]
        nums = [int(num) for num in nums if num.isnumeric()]

        if len(nums) == 0:
            return 0
        else:
            return max(nums)+1

    def _get_run_name(self):
        name_without_number = get_random_name(separator="-", style="lowercase")
        number = self._get_next_run_number()
        return f"{number}-{name_without_number}"

    def _get_latest_run_name(self):
        # find all directories that should be runs
        log_dirs = glob.glob(f"{self.save_directory}/*")
        log_dirs = [path for path in log_dirs if os.path.isdir(path)]
        # find run names based on the naming convention and sort them by last modified time
        runs = [l.replace(f"{self.save_directory}/","") for l in log_dirs if l.split("-")[0].isnumeric()]
        runs.sort(key = lambda x: os.path.getmtime(os.path.join(self.save_directory, x)), reverse=True)
        return runs[0]

    @rank_zero_only
    def log_hyperparams(self, params):
        if self.is_tensorboard:
            self.experiment["tensorboard"].add_hparams(
                params,
                {'placeholder_metric': 0}, # placeholder metric is needed due to this issue: https://github.com/tensorflow/tensorboard/issues/5476
            )
        if self.is_wandb:
            self.experiment["wandb"].config.update(params)
        if self.is_mlflow:
            self.experiment["mlflow"].log_params(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if self.is_tensorboard:
            for key, value in metrics.items():
                self.experiment["tensorboard"].add_scalar(key, value, step)
        if self.is_wandb:
            self.experiment["wandb"].log(metrics, step)
        if self.is_mlflow:
            self.experiment["mlflow"].log_metrics(metrics, step)

    @rank_zero_only
    def log_images(self, head_name, img_batch, step):
        """
        head_name: name of the model head
        img_batch: np.array of images in RGB order, uint8, NHWC
        step: epoch in which we logged the image
        """
        if self.is_tensorboard:
            self.experiment["tensorboard"].add_images(head_name, img_batch, step, dataformats='NHWC')
        
        if self.is_wandb:
            ## TODO
            pass

        if self.is_mlflow:
            ## TODO
            pass


    @rank_zero_only
    def log_image(self, head_name, img, step):
        """
        head_name: name of the model head
        img: Image in RGB order, uint8, HWC
        step: epoch in which we logged the image
        """

        if self.is_tensorboard:
            self.experiment["tensorboard"].add_image(head_name, img, step, dataformats='HWC')

        if self.is_wandb:
            name = "name_of_choice"
            wandb_image = self._experiment["wandb"].Image(img, caption=name)
            self._experiment["wandb"].log({name: wandb_image})

        if self.is_mlflow:
            ## TODO: has yet to be tested!
            name = "name_of_choice"
            self.mlflow.log_image(
                img, f"{name}_{step}.png"
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
