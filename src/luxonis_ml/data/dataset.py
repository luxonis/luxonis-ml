import luxonis_ml.data.utils.data_utils as data_utils

# from luxonis_ml.data.utils.exceptions import *
from luxonis_ml.data.utils.parquet import ParquetFileManager
from luxonis_ml.enums import LabelType
import os, shutil, subprocess
from pathlib import Path
import cv2
import json
import boto3
from google.cloud import storage
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Generator

# TODO: from luxonis_ml.filesystem import LuxonisFilesystem
# TODO: from luxonis_ml.logger import LuxonisLogger
from .utils.s3_utils import *
from .utils.gcs_utils import *
from .utils.constants import LDF_VERSION, LABEL_TYPES
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
                "classes": {},
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
            self.media_path = os.path.join(self.path, "media")
            self.annotations_path = os.path.join(self.path, "annotations")
            self.metadata_path = os.path.join(self.path, "metadata")
            os.makedirs(self.media_path, exist_ok=True)
            os.makedirs(self.annotations_path, exist_ok=True)
            os.makedirs(self.metadata_path, exist_ok=True)
        elif self.bucket_storage.value == "s3":
            self.path = f"s3://{self._get_credentials('AWS_BUCKET')}/{self.team_id}/datasets/{self.dataset_name}"
            self._init_boto3_client()
        elif self.bucket_storage.value == "gcs":
            self.path = f"gs://{self._get_credentials('GCS_BUCKET')}/{self.team_id}/datasets/{self.dataset_name}"
            self._init_gcs_client()

    def _load_df_offline(self) -> Optional[pd.DataFrame]:
        dfs = []
        for file in os.listdir(self.annotations_path):
            if os.path.splitext(file)[1] == ".parquet":
                dfs.append(pd.read_parquet(os.path.join(self.annotations_path, file)))
        if len(dfs):
            return pd.concat(dfs)
        else:
            return None

    def _try_instance_id(
        self, file: str, index: Optional[pd.DataFrame]
    ) -> Optional[str]:
        if index is None:
            return None

        if file in index["file"]:
            test = index[index["file"] == file]
            print(test)
            return test
        else:
            return None

    def _get_file_index(self) -> Optional[pd.DataFrame]:
        index = None
        file_index_path = os.path.join(self.metadata_path, "file_index.parquet")
        if os.path.exists(file_index_path):
            index = pd.read_parquet(file_index_path)
        return index

    def _write_index(self, index: Optional[pd.DataFrame], new_index: Dict) -> None:
        file_index_path = os.path.join(self.metadata_path, "file_index.parquet")
        df = pd.DataFrame(new_index)
        if index is not None:
            df = pd.concat(index, df)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_index_path)

    def update_source(self, source: LuxonisSource) -> None:
        """
        Updates underlying source of the dataset with a new LuxonisSource
        """

        self.datasets[self.dataset_name]["source"] = self._source_to_document(source)
        self._write_datasets()
        self.source = source

    def set_classes(self, classes: List[str], task: Optional[str] = None) -> None:
        """
        Sets the names of classes for the dataset. This can be across all CV tasks or certain tasks

        classes:
            List of class names to set
        task:
            Optionally specify the LabelType where these classes apply
        """

        if task is not None:
            if task not in LABEL_TYPES:
                raise Exception(f"Task {task} is not a supported task")
            self.datasets[self.dataset_name]["classes"][task] = classes
        else:
            for task in LABEL_TYPES:
                self.datasets[self.dataset_name]["classes"][task] = classes
        self._write_datasets()

    # TODO: method to auto-set classes per-task using pandas

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

    def get_classes(self) -> Tuple[List[str], Dict]:
        """Gets overall classes in the dataset and classes according to CV task"""

        classes = set()
        classes_by_task = {}
        for task in self.datasets[self.dataset_name]["classes"]:
            task_classes = self.datasets[self.dataset_name]["classes"][task]
            if len(task_classes):
                classes_by_task[task] = task_classes
                for cls in task_classes:
                    classes.add(cls)
        classes = list(classes)
        classes.sort()

        return classes, classes_by_task

    def delete_dataset(self) -> None:
        """Deletes all local files belonging to the dataset"""

        del self.datasets[self.dataset_name]
        self._write_datasets()
        shutil.rmtree(self.path)

    def add(self, generator: Generator) -> None:
        """
        Write annotations to parquet files.

        generator: A python generator that yields dictionaries of data
            with the key described by the ANNOTATIONS_SCHEMA but also listed below
            - file [str] : path to file on local disk or object storage
            - class [str]: string specifying the class name or label name
            - type [str] : the type of label or annotation
            - value [Union[str, list, int, float, bool]]: the actual annotation value
                For here are the expected structures for `value`.
                The function will check to ensure `value` matches this for each annotation type

                value (classification) [bool] : Marks whether the class is present or not
                    (e.g. True/False)
                value (box) [List[float]] : the normalized (0-1) x, y, w, and h of a bounding box
                    (e.g. [0.5, 0.4, 0.1, 0.2])
                value (polyline) [List[List[float]]] : an ordered list of [x, y] polyline points
                    (e.g. [[0.2, 0.3], [0.4, 0.5], ...])
                value (keypoints) [List[List[float]]] : an ordered list of keypoints for a keypoint skeleton instance
                    (e.g. [[0.2, 0.3], [0.4, 0.5], ...])
        """

        self.pfm = ParquetFileManager(self.annotations_path)

        index = self._get_file_index()
        new_index = {"instance_id": [], "file": []}

        for data in generator():
            # Add media
            filepath = data["file"]
            file = os.path.basename(filepath)
            new_file, instance_id = data_utils.generate_hashname(filepath)
            matched_id = self._try_instance_id(file, index)
            if matched_id is not None:
                if matched_id != instance_id:
                    # TODO: not sure if this should be an exception or how we should really handle it
                    raise Exception(
                        f"{filepath} uses a duplicate filename corresponding to different media! Please rename this file."
                    )
                    # TODO: we may also want to check for duplicate instance_ids to get a one-to-one relationship
            else:
                new_index["instance_id"].append(instance_id)
                new_index["file"].append(file)
            cmd = f"cp {filepath} {os.path.join(self.media_path, new_file)}"
            subprocess.check_output(cmd, shell=True)

            # Add annotation
            data_utils.check_annotation(data)
            # TODO: ability to get instance_id also from bucket
            data["instance_id"] = instance_id
            data[
                "file"
            ] = file  # TODO: determine if we actually need to store this on the annotation level
            data["value_type"] = type(data["value"]).__name__
            if isinstance(data["value"], list):
                data["value"] = json.dumps(data["value"])  # convert lists to string
            else:
                data["value"] = str(data["value"])

            self.pfm.write(data)

        # TODO: copy everything at the end with multithread instead of within for loop
        # similar logic can be used for buckets

        self.pfm.close()
        self._write_index(index, new_index)

    def make_splits(
        self, ratios: List[float] = [0.8, 0.1, 0.1], definitions: Optional[Dict] = None
    ) -> None:
        """
        Saves a splits.json file that specified the train/val/test split.
        For use in OFFLINE mode only.

        ratios [List[float]] : length 3 list of train/val/test ratios in that order used for a random split.
            If no definitions are provided, this is used to generate a random split.
        definitions [Optional[Dict]] : dictionary specifying split keys to lists of filepath values.
            Note that this assumes unique filenames
            (e.g. {"train": ["/path/to/cat.jpg", "/path/to/dog.jpg"], "val": [...], "test": [...]})

        WARNING: this will overwrite any previously saved splits.
        """

        splits = {}

        if definitions is None:  # random split
            df = self._load_df_offline()
            ids = list(set(df["instance_id"]))
            np.random.shuffle(ids)
            N = len(ids)
            b1 = round(N * ratios[0])
            b2 = round(N * ratios[0]) + round(N * ratios[1])
            splits["train"] = ids[:b1]
            splits["val"] = ids[b1:b2]
            splits["test"] = ids[b2:]
        else:  # provided split
            index = self._get_file_index()
            if index is None:
                raise Exception("File index not found")
            if set(definitions.keys()) != set("train", "val", "test"):
                raise Exception("Must specify train, val, and test and those keys only")
            for split in "train", "val", "test":
                files = definitions[split]
                if not isinstance(files, list):
                    raise Exception("Must provide splits as a list of str")
                files = [os.path.basename(file) for file in files]
                ids = [self._try_instance_id(file, index) for file in files]
                splits[split] = ids

        with open(os.path.join(self.path, "splits.json"), "w") as file:
            json.dump(splits, file, indent=4)
