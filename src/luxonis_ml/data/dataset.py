import luxonis_ml.data.utils.data_utils as data_utils

# from luxonis_ml.data.utils.exceptions import *
from luxonis_ml.data.utils.parquet import ParquetFileManager
import os, shutil
from pathlib import Path
import cv2
import json
import boto3
from google.cloud import storage
import logging, time
from tqdm import tqdm
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Generator

# TODO: from luxonis_ml.filesystem import LuxonisFilesystem
# TODO: from luxonis_ml.logger import LuxonisLogger
from .utils.s3_utils import *
from .utils.gcs_utils import *
from .utils.constants import LDF_VERSION
from .utils.enums import *


class LuxonisComponent:
    """Abstract class for a piece of media within a source.
    Most commonly, this abstracts an image sensor."""

    def __init__(
        self,
        name: str,
        media_type: MediaType = MediaType.IMAGE,
        image_type: Optional[ImageType] = ImageType.COLOR,
    ) -> None:
        if media_type not in MediaType:
            raise Exception(f"{media_type.value} is not a valid MediaType")
        if image_type not in ImageType:
            raise Exception(f"{image_type.value} is not a valid ImageType")

        self.name = name
        self.media_type = media_type
        self.image_type = image_type

        if media_type == MediaType.IMAGE:
            self.image_type = image_type
        else:
            self.image_type = None


class LuxonisSource:
    """Abstracts the structure of a dataset and which components/media are included"""

    def __init__(
        self,
        name: str,
        components: Optional[List[LuxonisComponent]] = None,
        main_component: Optional[str] = None,
    ) -> None:
        self.name = name
        if components is None:
            components = [
                LuxonisComponent(name)
            ]  # basic source includes a single color image

        self.components = {component.name: component for component in components}

        if main_component is not None:
            self.main_component = main_component
        else:
            self.main_component = list(self.components.keys())[
                0
            ]  # make first component main component by default


class LuxonisDataset:
    """Luxonis Dataset Format (LDF). Used to define datasets in the Luxonis MLOps ecosystem"""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        team_id: Optional[str] = None,
        team_name: Optional[str] = None,
        bucket_type: BucketType = BucketType.INTERNAL,
        bucket_storage: BucketStorage = BucketStorage.LOCAL
        # TODO: Some param for offline mode? Or should this be bucket_storage?
    ) -> None:
        """
        Initializes LDF

        dataset_name:
            Name of the dataset
        team_id:
            Optional unique team identifier for the cloud
        team_name:
            Optional team name for the cloud
        dataset_id:
            Optional dataset ID unique identifier
        bucket_type:
            Whether to use external cloud buckets
        bucket_storage:
            Underlying bucket storage from local (no cloud), S3, or GCS
        """

        if dataset_name is None and dataset_id is None:
            raise Exception(
                "Must provide either dataset_name or dataset_id when initializing LuxonisDataset"
            )

        if team_id is None:
            team_id = os.getenv("LUXONISML_TEAM_ID", "offline")
        if team_name is None:
            team_name = os.getenv("LUXONISML_TEAM_ID", "offline")

        self.base_path = os.getenv(
            "LUXONISML_BASE_PATH", str(Path.home() / "luxonis_ml")
        )

        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.team_id = team_id
        self.team_name = team_name
        self.bucket_type = bucket_type
        self.bucket_storage = bucket_storage
        if not isinstance(self.bucket_type, BucketType):
            raise Exception(f"Must use a valid BucketType!")
        if not isinstance(self.bucket_storage, BucketStorage):
            raise Exception(f"Must use a valid BucketStorage!")

        credentials_cache_file = str(Path(self.base_path) / "credentials.json")
        if os.path.exists(credentials_cache_file):
            with open(credentials_cache_file) as file:
                self.creds = json.load(file)
        else:
            self.creds = {}

        self.datasets_cache_file = str(Path(self.base_path) / "datasets.json")
        if os.path.exists(self.datasets_cache_file):
            with open(self.datasets_cache_file) as file:
                self.datasets = json.load(file)
        else:
            self.datasets = {}

        # TODO: make a logging class in a different file
        # log_format = "%(asctime)s [%(levelname)s] %(message)s"
        # self.log_level = os.environ.get("LUXONISML_LEVEL", "INFO").upper()
        # logging.basicConfig(level=self.log_level, format=log_format)
        # self.logger = logging.getLogger(__name__)
        # file_handler = logging.FileHandler("ldf_debug.log")
        # file_handler.setLevel(logging.DEBUG)
        # formatter = logging.Formatter(log_format)
        # file_handler.setFormatter(formatter)
        # self.logger.addHandler(file_handler)
        # self.last_time = None

        # offline support only
        if self.dataset_name is None:
            raise NotImplementedError(
                "Cloud datasets initialized with dataset ID are not implemented"
            )
        if self.dataset_name in self.datasets:
            self.dataset_doc = self.datasets[self.dataset_name]

        else:
            self.source = LuxonisSource("default")
            self.datasets[self.dataset_name] = {
                "source": self._source_to_document(self.source),
                "ldf_version": LDF_VERSION,
            }
            self._write_datasets()

        self._init_path()

    def _write_datasets(self) -> None:
        with open(self.datasets_cache_file, "w") as file:
            json.dump(self.datasets, file, indent=4)

    def _component_to_document(self, component: LuxonisComponent) -> Dict:
        return {
            "name": component.name,
            "media_type": component.media_type.value,
            "image_type": component.image_type.value,
        }

    def _source_to_document(self, source: LuxonisSource) -> Dict:
        return {
            "name": source.name,
            "main_component": source.main_component,
            "components": [
                self._component_to_document(component)
                for component in source.components.values()
            ],
        }

    def _component_from_document(self, document: Dict) -> LuxonisComponent:
        if document["image_type"] is not None:
            return LuxonisComponent(
                name=document["name"],
                media_type=MediaType(document["media_type"]),
                image_type=ImageType(document["image_type"]),
            )
        else:
            return LuxonisComponent(
                name=document["name"], media_type=MediaType(document["media_type"])
            )

    def _source_from_document(self, document: Dict) -> LuxonisSource:
        return LuxonisSource(
            name=document["name"],
            main_component=document["main_component"],
            components=[
                self._component_from_document(component_doc)
                for component_doc in document["components"]
            ],
        )

    def _get_credentials(self, key: str) -> str:
        """Gets secret credentials from credentials file or ENV variables"""

        if key in self.creds.keys():
            return self.creds[key]
        else:
            return os.getenv(key, None)

    def _init_boto3_client(self) -> None:
        """Initializes connection to S3 bucket"""

        self.client = boto3.client(
            "s3",
            endpoint_url=self._get_credentials("AWS_S3_ENDPOINT_URL"),
            aws_access_key_id=self._get_credentials("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=self._get_credentials("AWS_SECRET_ACCESS_KEY"),
        )
        self.bucket = self.path.split("//")[1].split("/")[0]
        self.bucket_path = self.path.split(self.bucket + "/")[-1]

    def _init_gcs_client(self) -> None:
        """Initializes connection to GCS bucket"""

        self.bucket = self.path.split("//")[1].split("/")[0]
        self.bucket_path = self.path.split(self.bucket + "/")[-1]
        self.client = storage.Client()

    def _init_path(self) -> None:
        """Configures local path or bucket directory"""

        if self.bucket_storage.value == "local":
            self.path = str(
                Path(self.base_path)
                / "data"
                / self.team_id
                / "datasets"
                / self.dataset_name
            )
            os.makedirs(self.path, exist_ok=True)
        elif self.bucket_storage.value == "s3":
            self.path = f"s3://{self._get_credentials('AWS_BUCKET')}/{self.team_id}/datasets/{self.dataset_name}"
            self._init_boto3_client()
        elif self.bucket_storage.value == "gcs":
            self.path = f"gs://{self._get_credentials('GCS_BUCKET')}/{self.team_id}/datasets/{self.dataset_name}"
            self._init_gcs_client()

    # TODO: move to another file
    # def _log_time(self, note: Optional[str] = None, final: bool = False) -> None:
    #     """Helper to log the time taken within a function to INFO"""

    #     if self.logger.isEnabledFor(logging.DEBUG):
    #         if self.last_time is None:
    #             self.last_time = time.time()
    #         else:
    #             new_time = time.time()
    #             self.logger.debug(f"{note} took {(new_time-self.last_time)*1000} ms")
    #             self.last_time = new_time
    #             if final:
    #                 self.last_time = None

    def update_source(self, source: LuxonisSource) -> None:
        """
        Updates underlying source of the dataset with a new LuxonisSource
        """

        self.datasets["dataset_name"]["source"] = self._source_to_document(source)
        self._write_datasets()
        self.source = source

    # def set_classes(self, classes: List[str], task: Optional[str] = None) -> None:
    #     """
    #     Sets the names of classes for the dataset. This can be across all CV tasks or certain tasks

    #     classes:
    #         List of class names to set
    #     task:
    #         Optionally specify the task where these classes apply. This can be from the options in self.tasks
    #     """

    #     if task is not None:
    #         if task not in self.tasks:
    #             raise Exception(f"Task {task} is not a supported task")
    #         self.fo_dataset.classes[task] = classes
    #     else:
    #         for task in self.tasks:
    #             self.fo_dataset.classes[task] = classes

    def sync_from_cloud(self) -> None:
        """Downloads media from cloud bucket"""

        if self.bucket_storage.value == "local":
            self.logger.warning("This is a local dataset! Cannot sync")
        elif self.bucket_storage.value == "s3":
            sync_from_s3(
                non_streaming_dir=str(Path(self.base_path) / "data"),
                bucket=self.bucket,
                bucket_dir=str(Path(self.team_id) / "datasets" / self.dataset_id),
                endpoint_url=self._get_credentials("AWS_S3_ENDPOINT_URL"),
            )
        elif self.bucket_storage.value == "gcs":
            sync_from_gcs(
                non_streaming_dir=str(Path(self.base_path) / "data"),
                bucket=self.bucket,
                bucket_dir=str(Path(self.team_id) / "datasets" / self.dataset_id),
            )
        else:
            raise NotImplementedError

    # def get_classes(self) -> Tuple[List[str], Dict]:
    #     """Gets overall classes in the dataset and classes according to CV task"""

    #     classes = set()
    #     classes_by_task = {}
    #     for task in self.fo_dataset.classes:
    #         if len(self.fo_dataset.classes[task]):
    #             classes_by_task[task] = self.fo_dataset.classes[task]
    #             for cls in self.fo_dataset.classes[task]:
    #                 classes.add(cls)
    #     mask_classes = set()
    #     if "segmentation" in self.fo_dataset.mask_targets.keys():
    #         for key in self.fo_dataset.mask_targets["segmentation"]:
    #             mask_classes.add(self.fo_dataset.mask_targets["segmentation"][key])
    #     classes_by_task["segmentation"] = list(mask_classes)
    #     classes = list(classes.union(mask_classes))
    #     classes.sort()

    #     return classes, classes_by_task

    # def get_classes_count(self) -> Dict:
    #     """Returns dictionary with number of occurances for each class. If no class label is present returns empty dict."""

    #     try:
    #         count_dict = self.fo_dataset.count_values("class.classifications.label")
    #     except:
    #         self.logger.warning(
    #             "No 'class' label present in the dataset. Returning empty dictionary."
    #         )
    #         count_dict = {}
    #     return count_dict

    def delete_dataset(self) -> None:
        """Deletes the entire dataset (locally)"""

        del self.datasets[self.dataset_name]
        self._write_datasets()
        shutil.rmtree(self.path)

    def add(self, generator: Generator) -> None:
        """Write annotations to parquet files"""

        self.pfm = ParquetFileManager(self.path)
        for data in generator():
            data_utils.check_annotation(data)
            # TODO: ability to get instance_id also from bucket
            data["instance_id"] = data_utils.generate_hashname(data["file"])
            data["file"] = os.path.basename(data["file"])
            self.pfm.write(data)

        self.pfm.close()

    # TODO: method to generate splits.json for offline mode
