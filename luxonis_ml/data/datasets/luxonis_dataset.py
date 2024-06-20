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
import polars as pl
import pyarrow.parquet as pq
import rich.progress

import luxonis_ml.data.utils.data_utils as data_utils
from luxonis_ml.utils import LuxonisFileSystem, environ
from luxonis_ml.utils.filesystem import PathType

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
        if dataset_name is None and dataset_id is None:
            raise ValueError(
                "Must provide either dataset_name or dataset_id when initializing LuxonisDataset"
            )

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
        return len(df.select("uuid").unique()) if df is not None else 0

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

    def _load_df_offline(self, sync_mode: bool = False) -> Optional[pl.DataFrame]:
        dfs = []
        if self.bucket_storage == BucketStorage.LOCAL or sync_mode:
            annotations_path = self.annotations_path
        else:
            annotations_path = self.tmp_dir / "annotations"
        if not annotations_path.exists():
            return None
        for file in annotations_path.iterdir():
            if file.suffix == ".parquet":
                dfs.append(pl.read_parquet(file))
        if dfs:
            return pl.concat(dfs)
        else:
            return None

    def _find_filepath_uuid(
        self,
        filepath: Path,
        index: Optional[pl.DataFrame],
        *,
        raise_on_missing: bool = False,
    ) -> Optional[str]:
        if index is None:
            return None

        abs_path = str(filepath.absolute())
        matched = index.filter(pl.col("original_filepath") == abs_path)

        if len(matched):
            return list(matched.select("uuid"))[0][0]
        elif raise_on_missing:
            raise ValueError(f"File {abs_path} not found in index")
        return None

    def _get_file_index(self) -> Optional[pl.DataFrame]:
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
            index = pl.read_parquet(file_index_path).select(
                pl.all().exclude("^__index_level_.*$")
            )
        return index

    def _write_index(
        self,
        index: Optional[pl.DataFrame],
        new_index: Dict[str, List[str]],
        override_path: Optional[str] = None,
    ) -> None:
        if override_path:
            file_index_path = override_path
        else:
            file_index_path = self.metadata_path / "file_index.parquet"
        df = pl.DataFrame(new_index)
        if index is not None:
            df = pl.concat([index, df])
        pq.write_table(df.to_arrow(), file_index_path)

    @contextmanager
    def _log_time(self):
        t = time.time()
        yield
        self.logger.info(f"Took {time.time() - t} seconds")

    def _make_temp_dir(self, remove_previous: bool = True) -> None:
        if self.tmp_dir.exists() and remove_previous:
            self._remove_temp_dir()
        self.tmp_dir.mkdir(exist_ok=True)

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

    def set_classes(
        self,
        classes: List[str],
        task: Optional[str] = None,
        *,
        _remove_tmp_dir: bool = True,
    ) -> None:
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
            self._make_temp_dir(remove_previous=_remove_tmp_dir)
            local_file = self.tmp_dir / "classes.json"
            with open(local_file, "w") as file:
                json.dump(classes_json, file, indent=4)
            self.fs.put_file(local_file, "metadata/classes.json")
            if _remove_tmp_dir:
                self._remove_temp_dir()

    def get_classes(
        self, sync_mode: bool = False
    ) -> Tuple[List[str], Dict[str, List[str]]]:
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
                for cls_ in task_classes:
                    classes.add(cls_)
        classes = sorted(list(classes))

        return classes, classes_by_task

    def set_skeletons(
        self, skeletons: Dict[str, Dict], task: Optional[str] = None
    ) -> None:
        if task is None:
            raise NotImplementedError("Skeletons must be set for a specific task")

        if "skeletons" not in self.datasets[self.dataset_name]:
            self.datasets[self.dataset_name]["skeletons"] = {}
        self.datasets[self.dataset_name]["skeletons"][task] = skeletons
        self._write_datasets()

    def get_skeletons(self) -> Dict[str, Dict]:
        if "skeletons" not in self.datasets[self.dataset_name]:
            self.datasets[self.dataset_name]["skeletons"] = {}
        return self.datasets[self.dataset_name]["skeletons"]

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

    def delete_dataset(self) -> None:
        del self.datasets[self.dataset_name]
        self._write_datasets()
        if self.bucket_storage == BucketStorage.LOCAL:
            shutil.rmtree(self.path)

    def _process_arrays(self, batch_data: List[DatasetRecord]) -> None:
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
                    arrays_upload_dict = self.fs.put_dir(
                        local_paths=array_paths,
                        remote_dir="arrays",
                        uuid_dict=array_uuid_dict,
                    )
            self.logger.info("Finalizing paths...")
            self.progress.start()
            for ann in batch_data:
                if isinstance(ann, ArrayAnnotation):
                    if self.bucket_storage != BucketStorage.LOCAL:
                        remote_path = arrays_upload_dict[str(ann.path)]  # type: ignore
                        remote_path = (
                            f"{self.fs.protocol}://{self.fs.path / remote_path}"
                        )
                        ann.path = remote_path  # type: ignore
                    else:
                        ann.path = ann.path.absolute()
                    self.progress.update(task, advance=1)
            self.progress.stop()
            self.progress.remove_task(task)

    def _add_process_batch(
        self,
        batch_data: List[DatasetRecord],
        pfm: ParquetFileManager,
        index: Optional[pl.DataFrame],
        new_index: Dict[str, List[str]],
        processed_uuids: Set[str],
    ) -> None:
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

        self._process_arrays(batch_data)

        self.logger.info("Saving annotations...")
        with self._log_time():
            self.progress.start()
            for ann in batch_data:
                filepath = ann.file
                file = filepath.name
                uuid = uuid_dict[str(filepath)]
                matched_id = self._find_filepath_uuid(filepath, index)
                if matched_id is not None:
                    if matched_id != uuid:
                        # TODO: not sure if this should be an exception or how we should really handle it
                        raise Exception(
                            f"{filepath} already added to the dataset! Please skip or rename the file."
                        )
                        # TODO: we may also want to check for duplicate uuids to get a one-to-one relationship
                elif uuid not in processed_uuids:
                    new_index["uuid"].append(uuid)
                    new_index["file"].append(file)
                    new_index["original_filepath"].append(str(filepath.absolute()))
                    processed_uuids.add(uuid)

                pfm.write({"uuid": uuid, **ann.to_parquet_dict()})
                self.progress.update(task, advance=1)
            self.progress.stop()
            self.progress.remove_task(task)

    def add(self, generator: DatasetIterator, batch_size: int = 1_000_000) -> None:
        if self.bucket_storage == BucketStorage.LOCAL:
            annotations_dir = self.annotations_path
        else:
            self._make_temp_dir()
            annotations_dir = self.tmp_dir / "annotations"
            annotations_dir.mkdir(exist_ok=True, parents=True)

        index = self._get_file_index()
        new_index = {"uuid": [], "file": [], "original_filepath": []}
        processed_uuids = set()

        batch_data: list[DatasetRecord] = []

        classes_per_task: Dict[str, Set[str]] = defaultdict(set)
        num_kpts_per_task: Dict[str, int] = {}

        with ParquetFileManager(annotations_dir) as pfm:
            for i, data in enumerate(generator, start=1):
                record = (
                    data if isinstance(data, DatasetRecord) else DatasetRecord(**data)
                )
                if record.annotation is not None:
                    classes_per_task[record.annotation.task].add(
                        record.annotation.class_
                    )
                    if record.annotation.type_ == "keypoints":
                        num_kpts_per_task[record.annotation.task] = len(
                            record.annotation.keypoints
                        )

                batch_data.append(record)
                if i % batch_size == 0:
                    self._add_process_batch(
                        batch_data, pfm, index, new_index, processed_uuids
                    )
                    batch_data = []

            self._add_process_batch(batch_data, pfm, index, new_index, processed_uuids)

        _, curr_classes = self.get_classes()
        for task, classes in classes_per_task.items():
            old_classes = set(curr_classes.get(task, []))
            new_classes = list(classes - old_classes)
            self.logger.info(f"Detected new classes for task {task}: {new_classes}")
            self.set_classes(list(classes | old_classes), task, _remove_tmp_dir=False)

        if self.bucket_storage == BucketStorage.LOCAL:
            self._write_index(index, new_index)
        else:
            file_index_path = str(self.tmp_dir / "file_index.parquet")
            self._write_index(index, new_index, override_path=file_index_path)
            self.fs.put_dir(Path(annotations_dir), "annotations")  # type: ignore
            self.fs.put_file(file_index_path, "metadata/file_index.parquet")
            self._remove_temp_dir()

    def make_splits(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        definitions: Optional[Dict[str, List[PathType]]] = None,
    ) -> None:
        new_splits = {"train": [], "val": [], "test": []}
        splits_to_update = []

        if definitions is None:  # random split
            if self.bucket_storage != BucketStorage.LOCAL:
                self._make_temp_dir()
                self.fs.get_dir("annotations", self.tmp_dir / "annotations")

            df = self._load_df_offline()
            assert df is not None
            ids = df.select("uuid").unique().get_column("uuid").to_list()
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
                    self._find_filepath_uuid(
                        Path(filepath), index, raise_on_missing=True
                    )
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
