import json
import logging
import os
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rich.progress

import luxonis_ml.data.utils.data_utils as data_utils
from luxonis_ml.utils import LuxonisFileSystem, environ

from ..utils.constants import LDF_VERSION
from ..utils.enums import BucketStorage, BucketType
from ..utils.parquet import ParquetFileManager
from .annotation import ArrayAnnotation, DatasetRecord
from .base_dataset import BaseDataset, DatasetIterator
from .source import LuxonisSource


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

        self.base_path = environ.LUXONISML_BASE_PATH
        self.base_path.mkdir(exist_ok=True)

        credentials_cache_file = self.base_path / "credentials.json"
        if credentials_cache_file.exists():
            with open(credentials_cache_file) as file:
                self.config = json.load(file)
        else:
            self.config = {}

        team_id = team_id or self._get_config("LUXONISML_TEAM_ID")
        team_name = team_name or self._get_config("LUXONISML_TEAM_NAME")

        self.bucket = self._get_config("LUXONISML_BUCKET")

        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.team_id = team_id
        self.team_name = team_name
        self.bucket_type = bucket_type
        self.bucket_storage = bucket_storage

        if self.bucket_storage != BucketStorage.LOCAL and self.bucket is None:
            raise Exception("Must set LUXONISML_BUCKET environment variable!")

        self.datasets_cache_file = self.base_path / "datasets.json"
        if self.datasets_cache_file.exists():
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
                "source": self.source.to_document(),
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
            self.tmp_dir = Path(".luxonis_tmp")
            self.fs = LuxonisFileSystem(self.path)

        self.logger = logging.getLogger(__name__)

        self.progress = rich.progress.Progress(
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn(),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TimeRemainingColumn(),
        )

    @property
    def identifier(self) -> str:
        if self.dataset_name is not None:
            return self.dataset_name
        assert (
            self.dataset_id is not None
        ), "At least one of dataset_name or dataset_id must be provided."
        return self.dataset_id

    def __len__(self) -> int:
        """Returns the number of instances in the dataset."""

        df = self._load_df_offline(self.bucket_storage != BucketStorage.LOCAL)
        if df is not None:
            return len(set(df["instance_id"]))
        else:
            return 0

    def _write_datasets(self) -> None:
        with open(self.datasets_cache_file, "w") as file:
            json.dump(self.datasets, file, indent=4)

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

        self.local_path = (
            self.base_path / "data" / self.team_id / "datasets" / self.identifier
        )
        self.media_path = self.local_path / "media"
        self.annotations_path = self.local_path / "annotations"
        self.metadata_path = self.local_path / "metadata"
        self.masks_path = self.local_path / "masks"

        if self.bucket_storage == BucketStorage.LOCAL:
            self.path = str(self.local_path)
            for path in [
                self.media_path,
                self.annotations_path,
                self.metadata_path,
            ]:
                path.mkdir(exist_ok=True, parents=True)
        else:
            self.path = (
                f"{self.bucket_storage.value}://{self.bucket}/"
                f"{self.team_id}/datasets/{self.dataset_name}"
            )

    def _load_df_offline(self, sync_mode: bool = False) -> Optional[pd.DataFrame]:
        dfs = []
        if self.bucket_storage == BucketStorage.LOCAL or sync_mode:
            annotations_path = self.annotations_path
        else:
            annotations_path = self.tmp_dir / "annotations"
        if not annotations_path.exists():
            return None
        for file in annotations_path.iterdir():
            if file.suffix == ".parquet":
                dfs.append(pd.read_parquet(annotations_path / file))
        if len(dfs):
            return pd.concat(dfs)
        else:
            return None

    def _find_filepath_instance_id(
        self, filepath: Path, index: Optional[pd.DataFrame]
    ) -> Optional[str]:
        if index is None:
            return None

        abs_path = str(filepath.absolute())
        if abs_path in list(index["original_filepath"]):
            matched = index[index["original_filepath"] == abs_path]
            if len(matched):
                return list(matched["instance_id"])[0]
        else:
            return None

    def _get_file_index(self) -> Optional[pd.DataFrame]:
        index = None
        if self.bucket_storage == BucketStorage.LOCAL:
            file_index_path = self.metadata_path / "file_index.parquet"
        else:
            file_index_path = self.tmp_dir / "file_index.parquet"
            try:
                self.fs.get_file("metadata/file_index.parquet", file_index_path)
            except Exception:
                pass
        if file_index_path.exists():
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
            file_index_path = self.metadata_path / "file_index.parquet"
        df = pd.DataFrame(new_index)
        if index is not None:
            df = pd.concat([index, df])
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_index_path)

    @contextmanager
    def _log_time(self):
        t = time.time()
        yield
        self.logger.info(f"Took {time.time() - t} seconds")

    def _make_temp_dir(self) -> None:
        if self.tmp_dir.exists():
            self._remove_temp_dir()
        os.makedirs(self.tmp_dir, exist_ok=False)

    def _remove_temp_dir(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def update_source(self, source: LuxonisSource) -> None:
        """Updates underlying source of the dataset with a new LuxonisSource.

        @type source: L{LuxonisSource}
        @param source: The new L{LuxonisSource} to replace the old one.
        """

        self.datasets[self.dataset_name]["source"] = source.to_document()
        self._write_datasets()
        self.source = source

    def set_classes(self, classes: List[str], task: Optional[str] = None) -> None:
        """Sets the names of classes for the dataset. This can be across all CV tasks or
        certain tasks.

        @type classes: List[str]
        @param classes: List of class names to set.
        @type task: Optional[str]
        @param task: Optionally specify the task where these classes apply.
        """
        if task is not None:
            self.datasets[self.dataset_name]["classes"][task] = classes
        else:
            raise NotImplementedError(
                "Setting classes for all tasks not yet supported. "
                "Set classes individually for each task"
            )

        self._write_datasets()

        if self.bucket_storage != BucketStorage.LOCAL:
            classes_json = self.datasets[self.dataset_name]["classes"]
            self._make_temp_dir()
            local_file = self.tmp_dir / "classes.json"
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

        if self.bucket_storage != BucketStorage.LOCAL:
            skeletons_json = self.datasets[self.dataset_name]["skeletons"]
            self._make_temp_dir()
            local_file = self.tmp_dir / "skeletons.json"
            with open(local_file, "w") as file:
                json.dump(skeletons_json, file, indent=4)
            self.fs.put_file(local_file, "metadata/skeletons.json")
            self._remove_temp_dir()

    def sync_from_cloud(self) -> None:
        """Downloads data from a remote cloud bucket."""

        if self.bucket_storage == BucketStorage.LOCAL:
            self.logger.warning("This is a local dataset! Cannot sync")
        else:
            if not getattr(self, "is_synced", False):
                local_dir = self.base_path / "data" / self.team_id / "datasets"
                if not local_dir.exists():
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
            local_file = self.metadata_path / "classes.json"
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

    def get_skeletons(self, sync_mode: bool = False) -> Dict[str, Dict]:
        """Returns the dictionary defining the semantic skeleton for each class using
        keypoints.

        @rtype: Dict[str, Dict]
        @return: A dictionary mapping classes to their skeleton definitions.
        """
        if sync_mode:
            local_file = self.metadata_path / "skeletons.json"
            if not os.path.exists(local_file):
                self.logger.warning("Skeletons file not found at %s", local_file)
                return {}
            self.fs.get_file("metadata/skeletons.json", local_file)
            with open(local_file) as file:
                skeletons = json.load(file)
        else:
            if "skeletons" not in self.datasets[self.dataset_name]:
                self.logger.warning(
                    "No skeletons data available for dataset %s", self.dataset_name
                )
                return {}
            skeletons = self.datasets[self.dataset_name]["skeletons"]

        return skeletons

    def delete_dataset(self) -> None:
        """Deletes all local files belonging to the dataset."""

        del self.datasets[self.dataset_name]
        self._write_datasets()
        if self.bucket_storage == BucketStorage.LOCAL:
            shutil.rmtree(self.path)

    def add(
        self,
        generator: DatasetIterator,
        batch_size: int = 1_000_000,
    ) -> None:
        """Write annotations to parquet files.

        @type generator: L{DatasetGenerator}
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

        def _process_arrays(batch_data: List[DatasetRecord]) -> None:
            array_paths = set(
                ann.path for ann in batch_data if isinstance(ann, ArrayAnnotation)
            )
            if array_paths:
                task = self.progress.add_task(
                    "[magenta]Processing arrays...", total=len(batch_data)
                )
                self.logger.info("Checking arrays...")
                with self._log_time():
                    data_utils.check_arrays(array_paths)
                self.logger.info("Generating array UUIDs...")
                with self._log_time():
                    array_uuid_dict = self.fs.get_file_uuids(
                        array_paths, local=True
                    )  # TODO: support from bucket
                if self.bucket_storage != BucketStorage.LOCAL:
                    self.logger.info("Uploading arrays...")
                    # TODO: support from bucket (likely with a self.fs.copy_dir)
                    with self._log_time():
                        mask_upload_dict = self.fs.put_dir(
                            local_paths=array_paths,
                            remote_dir="arrays",
                            uuid_dict=array_uuid_dict,
                        )
                self.logger.info("Finalizing paths...")
                self.progress.start()
                for ann in batch_data:
                    if isinstance(ann, ArrayAnnotation):
                        if self.bucket_storage != BucketStorage.LOCAL:
                            remote_path = mask_upload_dict[str(ann.path)]
                            remote_path = (
                                f"{self.fs.protocol}://{self.fs.path / remote_path}"
                            )
                            ann.path = remote_path
                        else:
                            ann.path = ann.path.absolute()
                        self.progress.update(task, advance=1)
                self.progress.stop()

        def _add_process_batch(batch_data: List[DatasetRecord]) -> None:
            paths = list(set(data.file for data in batch_data))
            self.logger.info("Generating UUIDs...")
            with self._log_time():
                uuid_dict = self.fs.get_file_uuids(
                    paths, local=True
                )  # TODO: support from bucket
            if self.bucket_storage != BucketStorage.LOCAL:
                self.logger.info("Uploading media...")
                # TODO: support from bucket (likely with a self.fs.copy_dir)

                with self._log_time():
                    self.fs.put_dir(
                        local_paths=paths, remote_dir="media", uuid_dict=uuid_dict
                    )

            task = self.progress.add_task(
                "[magenta]Processing data...", total=len(batch_data)
            )

            _process_arrays(batch_data)

            self.logger.info("Saving annotations...")
            with self._log_time():
                self.progress.start()
                for ann in batch_data:
                    filepath = ann.file
                    file = filepath.name
                    uuid = uuid_dict[str(filepath)]
                    matched_id = self._find_filepath_instance_id(filepath, index)
                    if matched_id is not None:
                        if matched_id != uuid:
                            # TODO: not sure if this should be an exception or how we should really handle it
                            raise Exception(
                                f"{filepath} already added to the dataset! Please skip or rename the file."
                            )
                            # TODO: we may also want to check for duplicate instance_ids to get a one-to-one relationship
                    elif uuid not in new_index["instance_id"]:
                        new_index["instance_id"].append(uuid)
                        new_index["file"].append(file)
                        new_index["original_filepath"].append(str(filepath.absolute()))

                    self.pfm.write({"instance_id": uuid, **ann.to_parquet()})
                    self.progress.update(task, advance=1)
                self.progress.stop()

        if self.bucket_storage == BucketStorage.LOCAL:
            self.pfm = ParquetFileManager(str(self.annotations_path))
        else:
            self._make_temp_dir()
            annotations_dir = self.tmp_dir / "annotations"
            annotations_dir.mkdir(exist_ok=True, parents=True)
            self.pfm = ParquetFileManager(str(annotations_dir))

        index = self._get_file_index()
        new_index = {"instance_id": [], "file": [], "original_filepath": []}

        batch_data: list[DatasetRecord] = []

        classes_per_tasks: Dict[str, Set[str]] = defaultdict(set)

        for i, data in enumerate(generator, start=1):
            record = data if isinstance(data, DatasetRecord) else DatasetRecord(**data)
            if record.annotation is not None:
                classes_per_tasks[record.annotation.task].add(record.annotation.class_)

            batch_data.append(record)
            if i % batch_size == 0:
                _add_process_batch(batch_data)
                batch_data = []

        _add_process_batch(batch_data)

        _, curr_classes = self.get_classes()
        if not curr_classes:
            for task, classes in classes_per_tasks.items():
                self.set_classes(list(classes), task)
                self.logger.info(f"Detected classes for task {task}: {list(classes)}")

        self.pfm.close()

        if self.bucket_storage == BucketStorage.LOCAL:
            self._write_index(index, new_index)
        else:
            file_index_path = str(self.tmp_dir / "file_index.parquet")
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
        I{OFFLINE} mode only.

        @type ratios: Tuple[float, float, float]
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

        new_splits = {"train": [], "val": [], "test": []}
        splits_to_update = []

        if definitions is None:  # random split
            if self.bucket_storage != BucketStorage.LOCAL:
                self._make_temp_dir()
                self.fs.get_dir("annotations", self.tmp_dir / "annotations")

            df = self._load_df_offline()
            assert df is not None
            ids: list[str] = list(set(df["instance_id"]))
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
                filepaths = definitions[split]
                if not isinstance(filepaths, list):
                    raise Exception("Must provide splits as a list of str")
                ids = [
                    self._find_filepath_instance_id(Path(filepath), index)
                    for filepath in filepaths
                ]
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
        datasets_cache_file = base_path / "datasets.json"
        if datasets_cache_file.exists():
            with open(datasets_cache_file) as file:
                datasets = json.load(file)
        else:
            datasets = {}

        return datasets

    def get_tasks(self) -> List[str]:
        return list(self.get_classes()[1].keys())
