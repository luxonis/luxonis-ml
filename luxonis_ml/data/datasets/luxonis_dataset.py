import json
import logging
import os
import os.path as osp
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rich.progress

import luxonis_ml.data.utils.data_utils as data_utils
from luxonis_ml.utils import LuxonisFileSystem, environ

from ..utils.constants import LABEL_TYPES, LDF_VERSION
from ..utils.enums import BucketStorage, BucketType, ImageType, MediaType
from ..utils.parquet import ParquetFileManager
from .base_dataset import BaseDataset, DatasetGeneratorFunction
from .source import LuxonisComponent, LuxonisSource


class LuxonisDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        team_id: Optional[str] = None,
        team_name: Optional[str] = None,
        bucket_type: BucketType = BucketType.INTERNAL,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
    ) -> None:
        """Luxonis Dataset Format (LDF) is used to define datasets in the Luxonis MLOps
        ecosystem.

        @type dataset_name: Optional[str]
        @param dataset_name: Name of the dataset
        @type team_id: Optional[str]
        @param team_id: Optional unique team identifier for the cloud
        @type team_name: Optional[str]
        @param team_name: Optional team name for the cloud
        @type dataset_id: Optional[str]
        @param dataset_id: Optional dataset ID unique identifier
        @type bucket_type: BucketType
        @param bucket_type: Whether to use external cloud buckets
        @type bucket_storage: BucketStorage
        @param bucket_storage: Underlying bucket storage from local, S3, or GCS
        @raise ValueError: If C{dataset_name} is not a string or if neither
            C{dataset_name} nor C{dataset_id} are provided.
        """

        if dataset_name is not None and not isinstance(dataset_name, str):
            raise ValueError("`dataset_name` argument must be a string")

        if dataset_name is None and dataset_id is None:
            raise ValueError(
                "Must provide either dataset_name or dataset_id when initializing LuxonisDataset"
            )

        self.base_path = environ.LUXONISML_BASE_PATH
        os.makedirs(self.base_path, exist_ok=True)

        credentials_cache_file = osp.join(self.base_path, "credentials.json")
        if osp.exists(credentials_cache_file):
            with open(credentials_cache_file) as file:
                self.config = json.load(file)
        else:
            self.config = {}

        if team_id is None:
            team_id = self._get_config("LUXONISML_TEAM_ID")
        if team_name is None:
            team_name = self._get_config("LUXONISML_TEAM_NAME")

        self.bucket = self._get_config("LUXONISML_BUCKET")

        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.team_id = team_id
        self.team_name = team_name
        self.bucket_type = bucket_type
        self.bucket_storage = bucket_storage
        if not isinstance(self.bucket_type, BucketType):
            raise Exception("Must use a valid BucketType!")
        if not isinstance(self.bucket_storage, BucketStorage):
            raise Exception("Must use a valid BucketStorage!")

        if self.bucket_storage != BucketStorage.LOCAL and self.bucket is None:
            raise Exception("Must set LUXONISML_BUCKET environment variable!")

        self.datasets_cache_file = osp.join(self.base_path, "datasets.json")
        if osp.exists(self.datasets_cache_file):
            with open(self.datasets_cache_file) as file:
                self.datasets = json.load(file)
        else:
            self.datasets = {}

        if self.dataset_name is None:
            raise Exception("Must provide a dataset_name for offline mode")
        if self.dataset_name in self.datasets:
            self.dataset_doc = self.datasets[self.dataset_name]

        else:
            self.source = LuxonisSource("default")
            self.datasets[self.dataset_name] = {
                "source": self._source_to_document(self.source),
                "ldf_version": LDF_VERSION,
                "classes": {},
                "skeletons": {},
            }
            self._write_datasets()

        self._init_path()

        if self.bucket_storage == BucketStorage.LOCAL:
            self.fs = LuxonisFileSystem(f"file://{self.path}")
        elif self.bucket_storage == BucketStorage.AZURE_BLOB:
            raise NotImplementedError
        else:
            self.tmp_dir = ".luxonis_tmp"
            self.fs = LuxonisFileSystem(self.path)

        self.logger = logging.getLogger(__name__)

    @property
    def identifier(self) -> str:
        if self.dataset_name is not None:
            return self.dataset_name
        assert self.dataset_id is not None
        return self.dataset_id

    def __len__(self) -> int:
        """Returns the number of instances in the dataset."""

        df = self._load_df_offline()
        if df is not None:
            return len(set(df["instance_id"]))
        else:
            return 0

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

    def _get_config(self, key: str) -> str:
        """Gets secret credentials from credentials file or ENV variables."""

        if key in self.config.keys():
            return self.config[key]
        else:
            if not hasattr(environ, key):
                raise Exception(f"Must set {key} in ENV variables")
            return getattr(environ, key)

    def _init_path(self) -> None:
        """Configures local path or bucket directory."""

        self.local_path = osp.join(
            self.base_path,
            "data",
            self.team_id,
            "datasets",
            self.dataset_name,
        )
        self.media_path = osp.join(self.local_path, "media")
        self.annotations_path = osp.join(self.local_path, "annotations")
        self.metadata_path = osp.join(self.local_path, "metadata")
        self.masks_path = osp.join(self.local_path, "masks")

        if self.bucket_storage == BucketStorage.LOCAL:
            self.path = self.local_path
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.media_path, exist_ok=True)
            os.makedirs(self.annotations_path, exist_ok=True)
            os.makedirs(self.metadata_path, exist_ok=True)
        else:
            self.path = f"{self.bucket_storage.value}://{self.bucket}/{self.team_id}/datasets/{self.dataset_name}"

    def _load_df_offline(self, sync_mode: bool = False) -> Optional[pd.DataFrame]:
        dfs = []
        if self.bucket_storage == BucketStorage.LOCAL or sync_mode:
            annotations_path = self.annotations_path
        else:
            annotations_path = osp.join(self.tmp_dir, "annotations")
        for file in os.listdir(annotations_path):
            if osp.splitext(file)[1] == ".parquet":
                dfs.append(pd.read_parquet(osp.join(annotations_path, file)))
        if len(dfs):
            return pd.concat(dfs)
        else:
            return None

    def _try_instance_id(
        self, file: str, index: Optional[pd.DataFrame]
    ) -> Optional[str]:
        if index is None:
            return None

        if file in list(index["file"]):
            matched = index[index["file"] == file]
            if len(matched):
                return list(matched["instance_id"])[0]
        else:
            return None

    def _get_file_index(self) -> Optional[pd.DataFrame]:
        index = None
        if self.bucket_storage == BucketStorage.LOCAL:
            file_index_path = osp.join(self.metadata_path, "file_index.parquet")
        else:
            file_index_path = osp.join(self.tmp_dir, "file_index.parquet")
            try:
                self.fs.get_file("metadata/file_index.parquet", file_index_path)
            except Exception:
                pass
        if osp.exists(file_index_path):
            index = pd.read_parquet(file_index_path)
        return index

    def _write_index(
        self,
        index: Optional[pd.DataFrame],
        new_index: Dict,
        override_path: Optional[str] = None,
    ) -> None:
        if override_path:
            file_index_path = override_path
        else:
            file_index_path = osp.join(self.metadata_path, "file_index.parquet")
        df = pd.DataFrame(new_index)
        if index is not None:
            df = pd.concat([index, df])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_index_path)

    def _start_time(self) -> None:
        self.t0 = time.time()

    def _end_time(self) -> None:
        self.t1 = time.time()
        self.logger.info(f"Took {self.t1 - self.t0} seconds")

    def _make_temp_dir(self) -> None:
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=False)

    def _remove_temp_dir(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def update_source(self, source: LuxonisSource) -> None:
        """Updates underlying source of the dataset with a new LuxonisSource.

        @type source: L{LuxonisSource}
        @param source: The new L{LuxonisSource} to replace the old one.
        """

        self.datasets[self.dataset_name]["source"] = self._source_to_document(source)
        self._write_datasets()
        self.source = source

    def set_classes(self, classes: List[str], task: Optional[str] = None) -> None:
        """Sets the names of classes for the dataset. This can be across all CV tasks or
        certain tasks.

        @type classes: List[str]
        @param classes: List of class names to set.
        @type task: Optional[str]
        @param task: Optionally specify the LabelType where these classes apply.
        """

        if task is not None:
            if task not in LABEL_TYPES:
                raise Exception(f"Task {task} is not a supported task")
            self.datasets[self.dataset_name]["classes"][task] = classes
        else:
            for task in LABEL_TYPES:
                self.datasets[self.dataset_name]["classes"][task] = classes
        self._write_datasets()

        if self.bucket_storage != BucketStorage.LOCAL:
            classes_json = self.datasets[self.dataset_name]["classes"]
            self._make_temp_dir()
            local_file = osp.join(self.tmp_dir, "classes.json")
            with open(local_file, "w") as file:
                json.dump(classes_json, file, indent=4)
            self.fs.put_file(local_file, "metadata/classes.json")
            self._remove_temp_dir()

    # TODO: method to auto-set classes per-task using pandas

    def set_skeletons(self, skeletons: Dict[str, Dict]) -> None:
        """Sets the semantic structure of keypoint skeletons for the classes that use
        keypoints.

        @type skeletons: Dict[str, Dict]
        @param skeletons: A dict mapping class name to keypoint "labels" and "edges"
            between keypoints.
            The length of the "labels" determines the official number of keypoints.
            The inclusion of "edges" is optional.

            Example::

                {
                    "person": {
                        "labels": ["right hand", "right shoulder", ...]
                        "edges" [[0, 1], [4, 5], ...]
                    }
                }
        """

        self.datasets[self.dataset_name]["skeletons"] = skeletons
        self._write_datasets()

    def sync_from_cloud(self) -> None:
        """Downloads data from a remote cloud bucket."""

        if self.bucket_storage == BucketStorage.LOCAL:
            self.logger.warning("This is a local dataset! Cannot sync")
        else:
            if not hasattr(self, "is_synced") or not self.is_synced:
                local_dir = osp.join(self.base_path, "data", self.team_id, "datasets")
                if not osp.exists(local_dir):
                    os.makedirs(local_dir, exist_ok=True)

                self.fs.get_dir(remote_paths="", local_dir=local_dir)

                self.is_synced = True

    def get_classes(
        self, sync_mode: bool = False
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """Gets overall classes in the dataset and classes according to computer vision
        task.

        @type sync_mode: bool
        @param sync_mode: If C{True}, reads classes from remote storage. If C{False},
            classes are read locally.
        @rtype: Tuple[List[str], Dict[str, List[str]]
        @return: A combined list of classes for all tasks and a dictionary mapping tasks
            to the classes used in each task.
        """

        classes = set()
        classes_by_task = {}
        if sync_mode:
            local_file = osp.join(self.metadata_path, "classes.json")
            self.fs.get_file("metadata/classes.json", local_file)
            with open(local_file) as file:
                classes_json = json.load(file)
        else:
            classes_json = self.datasets[self.dataset_name]["classes"]
        for task in classes_json:
            task_classes = classes_json[task]
            if len(task_classes):
                classes_by_task[task] = task_classes
                for cls in task_classes:
                    classes.add(cls)
        classes = list(classes)
        classes.sort()

        return classes, classes_by_task

    def get_skeletons(self) -> Dict[str, Dict]:
        """Returns the dictionary defining the semantic skeleton for each class using
        keypoints.

        @rtype: Dict[str, Dict]
        @return: A dictionary mapping classes to their skeleton definitions.
        """

        return self.datasets[self.dataset_name]["skeletons"]

    def delete_dataset(self) -> None:
        """Deletes all local files belonging to the dataset."""

        del self.datasets[self.dataset_name]
        self._write_datasets()
        if self.bucket_storage == BucketStorage.LOCAL:
            shutil.rmtree(self.path)

    def add(
        self,
        generator: DatasetGeneratorFunction,
        batch_size: int = 1000000,
    ) -> None:
        """Write annotations to parquet files.

        @type generator: L{DatasetGeneratorFunction}
        @param generator: A Python iterator that yields dictionaries of data
            with the key described by the C{ANNOTATIONS_SCHEMA} but also listed below:
                - file (C{str}) : path to file on local disk or object storage
                - class (C{str}): string specifying the class name or label name
                - type (C{str}) : the type of label or annotation
                - value (C{Union[str, list, int, float, bool]}): the actual annotation value.
                The function will check to ensure `value` matches this for each annotation type

                    - value (classification) [bool] : Marks whether the class is present or not
                        (e.g. True/False)
                    - value (box) [List[float]] : the normalized (0-1) x, y, w, and h of a bounding box
                        (e.g. [0.5, 0.4, 0.1, 0.2])
                    - value (polyline) [List[List[float]]] : an ordered list of [x, y] polyline points
                        (e.g. [[0.2, 0.3], [0.4, 0.5], ...])
                    - value (segmentation) [Tuple[int, int, List[int]]]: an RLE representation of (height, width, counts) based on the COCO convention
                    - value (keypoints) [List[List[float]]] : an ordered list of [x, y, visibility] keypoints for a keypoint skeleton instance
                        (e.g. [[0.2, 0.3, 2], [0.4, 0.5, 2], ...])
                    - value (array) [str]: path to a numpy .npy file

        @type batch_size: int
        @param batch_size: The number of annotations generated before processing.
            This can be set to a lower value to reduce memory usage.
        """

        def _add_process_batch(batch_data: List[Dict]) -> None:
            paths = list(set([data["file"] for data in batch_data]))
            self.logger.info("Generating UUIDs...")
            self._start_time()
            uuid_dict = self.fs.get_file_uuids(
                paths, local=True
            )  # TODO: support from bucket
            self._end_time()
            if self.bucket_storage != BucketStorage.LOCAL:
                self.logger.info("Uploading media...")
                # TODO: support from bucket (likely with a self.fs.copy_dir)
                self._start_time()
                self.fs.put_dir(
                    local_paths=paths, remote_dir="media", uuid_dict=uuid_dict
                )
                self._end_time()

            array_paths = list(
                set([data["value"] for data in batch_data if data["type"] == "array"])
            )
            progress = rich.progress.Progress(
                rich.progress.TextColumn("[progress.description]{task.description}"),
                rich.progress.BarColumn(),
                rich.progress.TaskProgressColumn(),
                rich.progress.MofNCompleteColumn(),
                rich.progress.TimeRemainingColumn(),
            )
            task = progress.add_task(
                "[magenta]Processing data...", total=len(batch_data)
            )
            if len(array_paths):
                self.logger.info("Checking arrays...")
                self._start_time()
                data_utils.check_arrays(array_paths)
                self._end_time()
                self.logger.info("Generating array UUIDs...")
                self._start_time()
                array_uuid_dict = self.fs.get_file_uuids(
                    array_paths, local=True
                )  # TODO: support from bucket
                self._end_time()
                if self.bucket_storage != BucketStorage.LOCAL:
                    self.logger.info("Uploading arrays...")
                    # TODO: support from bucket (likely with a self.fs.copy_dir)
                    self._start_time()
                    mask_upload_dict = self.fs.put_dir(
                        local_paths=array_paths,
                        remote_dir="arrays",
                        uuid_dict=array_uuid_dict,
                    )
                    self._end_time()
                self.logger.info("Finalizing paths...")
                progress.start()
                for data in batch_data:
                    if data["type"] == "array":
                        if self.bucket_storage != BucketStorage.LOCAL:
                            remote_path = mask_upload_dict[data["value"]]
                            remote_path = f"{self.fs.protocol}://{osp.join(self.fs.path, remote_path)}"
                            data["value"] = remote_path
                        else:
                            data["value"] = osp.abspath(data["value"])
                        progress.update(task, advance=1)
                progress.stop()

            self.logger.info("Saving annotations...")
            self._start_time()
            progress.reset(task)
            progress.start()
            for data in batch_data:
                filepath = data["file"]
                file = osp.basename(filepath)
                instance_id = uuid_dict[filepath]
                matched_id = self._try_instance_id(file, index)
                if matched_id is not None:
                    if matched_id != instance_id:
                        # TODO: not sure if this should be an exception or how we should really handle it
                        raise Exception(
                            f"{filepath} uses a duplicate filename corresponding to different media! Please rename this file."
                        )
                        # TODO: we may also want to check for duplicate instance_ids to get a one-to-one relationship
                elif instance_id not in new_index["instance_id"]:
                    new_index["instance_id"].append(instance_id)
                    new_index["file"].append(file)
                    new_index["original_filepath"].append(osp.abspath(filepath))

                data_utils.check_annotation(data)
                data["instance_id"] = instance_id
                data["file"] = file
                data["value_type"] = type(data["value"]).__name__
                if data["type"] == "segmentation":  # handles RLE
                    data["value"] = data_utils.transform_segmentation_value(
                        data["value"]
                    )
                if isinstance(data["value"], (list, tuple)):
                    data["value"] = json.dumps(data["value"])  # convert lists to string
                else:
                    data["value"] = str(data["value"])
                data["created_at"] = datetime.utcnow()

                self.pfm.write(data)
                progress.update(task, advance=1)
            progress.stop()
            self._end_time()

        if self.bucket_storage == BucketStorage.LOCAL:
            self.pfm = ParquetFileManager(self.annotations_path)
        else:
            self._make_temp_dir()
            annotations_dir = osp.join(self.tmp_dir, "annotations")
            os.makedirs(annotations_dir, exist_ok=True)
            self.pfm = ParquetFileManager(annotations_dir)

        index = self._get_file_index()
        new_index = {"instance_id": [], "file": [], "original_filepath": []}

        batch_data = []

        for i, data in enumerate(generator()):
            batch_data.append(data)
            if (i + 1) % batch_size == 0:
                _add_process_batch(batch_data)
                batch_data = []

        _add_process_batch(batch_data)

        self.pfm.close()

        if self.bucket_storage == BucketStorage.LOCAL:
            self._write_index(index, new_index)
        else:
            file_index_path = osp.join(".luxonis_tmp", "file_index.parquet")
            self._write_index(index, new_index, override_path=file_index_path)
            self.fs.put_dir(Path(annotations_dir), "annotations")
            self.fs.put_file(file_index_path, "metadata/file_index.parquet")
            self._remove_temp_dir()

    def make_splits(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        definitions: Optional[Dict] = None,
    ) -> None:
        """Saves a splits json file that specified the train/val/test split. For use in
        OFFLINE mode only.

        @type ratios: Tuple[float, float, float]
        @param ratios: A tuple of rations for train/val/test used for a random split.
            Defaults to (0.8, 0.1, 0.1).

        @type definitions: Optional[Dict]
        @param definitions [Optional[Dict]]: Dictionary specifying split keys to lists
            of filepath values. Note that this assumes unique filenames.
            Example::

                {
                    "train": ["/path/to/cat.jpg", "/path/to/dog.jpg"],
                    "val": [...],
                    "test": [...]
                }

            Only overrides splits that are present in the dictionary.
        """

        new_splits = {"train": {}, "val": {}, "test": {}}
        splits_to_update = []

        if definitions is None:  # random split
            if self.bucket_storage != BucketStorage.LOCAL:
                self._make_temp_dir()
                self.fs.get_dir("annotations", osp.join(self.tmp_dir, "annotations"))

            df = self._load_df_offline()
            ids = list(set(df["instance_id"]))
            np.random.shuffle(ids)
            N = len(ids)
            b1 = round(N * ratios[0])
            b2 = round(N * ratios[0]) + round(N * ratios[1])
            new_splits["train"] = ids[:b1]
            new_splits["val"] = ids[b1:b2]
            new_splits["test"] = ids[b2:]
            splits_to_update = ["train", "val", "test"]
        else:  # provided split
            index = self._get_file_index()
            if index is None:
                raise Exception("File index not found")
            for split in "train", "val", "test":
                if split not in definitions:
                    continue
                splits_to_update.append(split)
                files = definitions[split]
                if not isinstance(files, list):
                    raise Exception("Must provide splits as a list of str")
                files = [osp.basename(file) for file in files]
                ids = [self._try_instance_id(file, index) for file in files]
                new_splits[split] = ids

        if self.bucket_storage == BucketStorage.LOCAL:
            splits_path = os.path.join(self.metadata_path, "splits.json")
            if os.path.exists(splits_path):
                with open(splits_path, "r") as file:
                    splits = json.load(file)
                for split in splits_to_update:
                    splits[split] = new_splits[split]
            else:
                splits = new_splits
            with open(os.path.join(self.metadata_path, "splits.json"), "w") as file:
                json.dump(splits, file, indent=4)
        else:
            remote_splits_path = "metadata/splits.json"
            local_splits_path = os.path.join(self.tmp_dir, "splits.json")
            if self.fs.exists(remote_splits_path):
                self.fs.get_file(remote_splits_path, local_splits_path)
                with open(local_splits_path, "r") as file:
                    splits = json.load(file)
                for split in splits_to_update:
                    splits[split] = new_splits[split]
            else:
                splits = new_splits
            with open(local_splits_path, "w") as file:
                json.dump(splits, file, indent=4)
            self.fs.put_file(local_splits_path, "metadata/splits.json")
            self._remove_temp_dir()

    @staticmethod
    def exists(dataset_name: str) -> bool:
        """Checks whether a dataset exists.

        @warning: For offline mode only.
        @type dataset_name: str
        @param dataset_name: Name of the dataset
        @rtype: bool
        @return: Whether the dataset exists
        """
        return dataset_name in LuxonisDataset.list_datasets()

    @staticmethod
    def list_datasets() -> Dict:
        """Returns a dictionary of all datasets.

        @rtype: Dict
        @return: Dictionary of all datasets
        """
        base_path = environ.LUXONISML_BASE_PATH
        datasets_cache_file = osp.join(base_path, "datasets.json")
        if osp.exists(datasets_cache_file):
            with open(datasets_cache_file) as file:
                datasets = json.load(file)
        else:
            datasets = {}

        return datasets
