import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
)

import cv2
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from bidict import bidict
from filelock import FileLock
from loguru import logger
from semver.version import Version
from typing_extensions import Self, override

from luxonis_ml.data.utils import (
    BucketStorage,
    BucketType,
    ParquetFileManager,
    UpdateMode,
    infer_task,
    warn_on_duplicates,
)
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.data.utils.task_utils import get_task_type
from luxonis_ml.enums.enums import DatasetType
from luxonis_ml.typing import PathType
from luxonis_ml.utils import (
    LuxonisFileSystem,
    deprecated,
    environ,
    make_progress_bar,
)

from .annotation import Category, DatasetRecord, Detection
from .base_dataset import BaseDataset, DatasetIterator
from .metadata import Metadata
from .migration import migrate_dataframe, migrate_metadata
from .source import LuxonisSource
from .utils import find_filepath_uuid, get_dir, get_file


class LuxonisDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        team_id: Optional[str] = None,
        bucket_type: Union[
            BucketType, Literal["internal", "external"]
        ] = BucketType.INTERNAL,
        bucket_storage: Union[
            BucketStorage, Literal["local", "gcs", "s3", "azure"]
        ] = BucketStorage.LOCAL,
        *,
        delete_existing: bool = False,
        delete_remote: bool = False,
    ) -> None:
        """Luxonis Dataset Format (LDF) is used to define datasets in
        the Luxonis MLOps ecosystem.

        @type dataset_name: str
        @param dataset_name: Name of the dataset
        @type team_id: Optional[str]
        @param team_id: Optional unique team identifier for the cloud
        @type bucket_type: BucketType
        @param bucket_type: Whether to use external cloud buckets
        @type bucket_storage: BucketStorage
        @param bucket_storage: Underlying bucket storage. Can be one of
            C{local}, C{S3}, or C{GCS}.
        @type delete_existing: bool
        @param delete_existing: Whether to delete a dataset with the
            same name if it exists
        @type delete_remote: bool
        @param delete_remote: Whether to delete the dataset from the
            cloud as well
        """

        self.dataset_name = dataset_name
        self.base_path = environ.LUXONISML_BASE_PATH
        self.base_path.mkdir(exist_ok=True)

        self._credentials = self._init_credentials()
        self._is_synced = False

        # What is this for?
        self.bucket_type = BucketType(bucket_type)

        self.bucket_storage = BucketStorage(bucket_storage)

        if self.bucket_storage == BucketStorage.AZURE_BLOB:
            raise NotImplementedError("Azure Blob Storage not yet supported")

        self.bucket = self._get_credential("LUXONISML_BUCKET")

        if self.is_remote and self.bucket is None:
            raise ValueError(
                "The `LUXONISML_BUCKET` environment variable "
                "must be set for remote datasets"
            )

        self.team_id = team_id or self._get_credential("LUXONISML_TEAM_ID")

        self._init_paths()

        self.fs = LuxonisFileSystem(self.path)

        if delete_existing:
            if self.exists(
                self.dataset_name,
                self.team_id,
                self.bucket_storage,
                self.bucket,
            ):
                self.delete_dataset(delete_remote=delete_remote)

            self._init_paths()

        # For DDP GCS training - multiple processes
        with FileLock(self.base_path / ".metadata.lock"):
            self._metadata = self._get_metadata()

        if self.version != LDF_VERSION:
            logger.warning(
                f"LDF versions do not match. The current `luxonis-ml` "
                f"installation supports LDF v{LDF_VERSION}, but the "
                f"`{self.identifier}` dataset is in v{self._metadata.ldf_version}. "
                "Internal migration will be performed. Note that some parts "
                "and new features might not work correctly unless you "
                "manually re-create the dataset using the latest version "
                "of `luxonis-ml`."
            )
        self.progress = make_progress_bar()

    @property
    def metadata(self) -> Metadata:
        """Returns a copy of the dataset metadata.

        The metadata is a pydantic model with the following fields:
            - source: L{LuxonisSource}
            - ldf_version: str
            - classes: Dict[task_name, Dict[class_name, class_id]]
            - tasks: Dict[task_name, List[task_type]]
            - skeletons: Dict[task_name, Skeletons]
              - Skeletons is a dictionary with keys 'labels' and 'edges'
                - labels: List[str]
                - edges: List[Tuple[int, int]]
            - categorical_encodings: Dict[task_name, Dict[metadata_name, Dict[metadata_value, int]]]
              - Encodings for string metadata values
              - Example::

                    {
                        "vehicle": {
                            "color": {"red": 0, "green": 1, "blue": 2},
                            "brand": {"audi": 0, "bmw": 1, "mercedes": 2},
                        }
                    }

        @type: L{Metadata}
        """
        return self._metadata.model_copy(deep=True)

    @cached_property
    def version(self) -> Version:
        return Version.parse(
            self._metadata.ldf_version, optional_minor_and_patch=True
        )

    @property
    def source(self) -> LuxonisSource:
        if self._metadata.source is None:
            raise ValueError("Source not found in metadata")
        return self._metadata.source

    @property
    @override
    def identifier(self) -> str:
        return self.dataset_name

    def __len__(self) -> int:
        """Returns the number of instances in the dataset."""
        if self.is_remote:
            return len(list(self.fs.walk_dir("media")))

        df = self._load_df_offline()
        return len(df.select("uuid").unique()) if df is not None else 0

    def _get_credential(self, key: str) -> str:
        """Gets secret credentials from credentials file or ENV
        variables."""

        if key in self._credentials:
            return self._credentials[key]
        if not hasattr(environ, key):
            raise RuntimeError(f"Must set {key} in ENV variables")
        return getattr(environ, key)

    def _init_paths(self) -> None:
        """Configures local path or bucket directory."""

        self.local_path = (
            self.base_path
            / "data"
            / self.team_id
            / "datasets"
            / self.dataset_name
        )
        self.media_path = self.local_path / "media"
        self.annotations_path = self.local_path / "annotations"
        self.metadata_path = self.local_path / "metadata"
        self.arrays_path = self.local_path / "arrays"

        for path in [
            self.media_path,
            self.annotations_path,
            self.metadata_path,
        ]:
            path.mkdir(exist_ok=True, parents=True)

        if not self.is_remote:
            self.path = str(self.local_path)
        else:
            self.path = self._construct_url(
                self.bucket_storage,
                self.bucket,
                self.team_id,
                self.dataset_name,
            )

    def _save_df_offline(self, pl_df: pl.DataFrame) -> None:
        """Saves the given Polars DataFrame into multiple Parquet files
        using ParquetFileManager. Ensures the same structure as the
        original dataset.

        @type pl_df: pl.DataFrame
        @param pl_df: The Polars DataFrame to save.
        """
        annotations_path = Path(self.annotations_path)

        for old_file in annotations_path.glob("*.parquet"):
            old_file.unlink()

        rows = pl_df.to_dicts()

        with ParquetFileManager(annotations_path) as pfm:
            for row in rows:
                uuid_val = row.get("uuid")
                if uuid_val is None:
                    raise ValueError("Missing 'uuid' in row!")

                data_dict = dict(row)
                data_dict.pop("uuid", None)

                pfm.write(uuid_val, data_dict)  # type: ignore

        logger.info(
            f"Saved merged DataFrame to Parquet files in '{annotations_path}'."
        )

    def _merge_metadata_with(self, other: "LuxonisDataset") -> None:
        """Merges relevant metadata from `other` into `self`."""
        self._metadata = self._metadata.merge_with(other._metadata)
        self._write_metadata()

    def clone(
        self, new_dataset_name: str, push_to_cloud: bool = True
    ) -> "LuxonisDataset":
        """Create a new LuxonisDataset that is a local copy of the
        current dataset. Cloned dataset will overwrite the existing
        dataset with the same name.

        @type new_dataset_name: str
        @param new_dataset_name: Name of the newly created dataset.
        @type push_to_cloud: bool
        @param push_to_cloud: Whether to push the new dataset to the
            cloud. Only if the current dataset is remote.
        """

        new_dataset = LuxonisDataset(
            dataset_name=new_dataset_name,
            team_id=self.team_id,
            bucket_type=self.bucket_type,
            bucket_storage=self.bucket_storage,
            delete_existing=True,
            delete_remote=True,
        )

        if self.is_remote:
            self.sync_from_cloud()

        new_dataset_path = Path(new_dataset.local_path)
        new_dataset_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self.local_path, new_dataset.local_path, dirs_exist_ok=True
        )

        new_dataset._init_paths()
        new_dataset._metadata = self._get_metadata()

        new_dataset._metadata.parent_dataset = self.dataset_name

        if self.is_remote and push_to_cloud:
            new_dataset.sync_to_cloud()

        path = self.metadata_path / "metadata.json"
        path.write_text(self._metadata.model_dump_json(indent=4))

        return new_dataset

    def sync_to_cloud(self) -> None:
        """Uploads data to a remote cloud bucket."""
        if not self.is_remote:
            logger.warning("This is a local dataset! Cannot sync")
            return

        logger.info("Syncing to cloud...")
        self.fs.put_dir(
            local_paths=self.local_path, remote_dir="", copy_contents=True
        )

    def merge_with(
        self,
        other: "LuxonisDataset",
        inplace: bool = True,
        new_dataset_name: Optional[str] = None,
    ) -> "LuxonisDataset":
        """Merge all data from `other` LuxonisDataset into the current
        dataset (in-place or in a new dataset).

        @type other: LuxonisDataset
        @param other: The dataset to merge into the current dataset.
        @type inplace: bool
        @param inplace: Whether to merge into the current dataset (True)
            or create a new dataset (False).
        @type new_dataset_name: str
        @param new_dataset_name: The name of the new dataset to create
            if inplace is False.
        """
        if inplace:
            target_dataset = self
        elif new_dataset_name:
            target_dataset = self.clone(new_dataset_name, push_to_cloud=False)
        else:
            raise ValueError(
                "You must specify a name for the new dataset "
                "when inplace is False"
            )

        if self.is_remote:
            other.sync_from_cloud(update_mode=UpdateMode.ALWAYS)
            self.sync_from_cloud(
                update_mode=UpdateMode.ALWAYS
                if inplace
                else UpdateMode.IF_EMPTY
            )

        df_self = self._load_df_offline(raise_when_empty=True)
        df_other = other._load_df_offline(raise_when_empty=True)
        duplicate_uuids = set(df_self["uuid"]).intersection(df_other["uuid"])
        if duplicate_uuids:
            df_other = df_other.filter(
                ~df_other["uuid"].is_in(duplicate_uuids)
            )

        df_merged = pl.concat([df_self, df_other])
        target_dataset._save_df_offline(df_merged)

        file_index_self = self._get_file_index(raise_when_empty=True)
        file_index_other = other._get_file_index(raise_when_empty=True)
        file_index_duplicates = set(file_index_self["uuid"]).intersection(
            file_index_other["uuid"]
        )
        if file_index_duplicates:
            file_index_other = file_index_other.filter(
                ~file_index_other["uuid"].is_in(file_index_duplicates)
            )

        merged_file_index = pl.concat([file_index_self, file_index_other])
        if merged_file_index is not None:
            file_index_path = (
                target_dataset.metadata_path / "file_index.parquet"
            )
            merged_file_index.write_parquet(file_index_path)

        splits_self = self._load_splits(self.metadata_path)
        splits_other = self._load_splits(other.metadata_path)
        self._merge_splits(splits_self, splits_other)
        target_dataset._save_splits(splits_self)

        if self.is_remote:
            shutil.copytree(
                other.media_path, target_dataset.media_path, dirs_exist_ok=True
            )
            target_dataset.sync_to_cloud()

        target_dataset._merge_metadata_with(other)

        return target_dataset

    def _load_splits(self, path: Path) -> Dict[str, List[str]]:
        splits_path = path / "splits.json"
        with open(splits_path) as f:
            return json.load(f)

    def _merge_splits(
        self,
        splits_self: Dict[str, List[str]],
        splits_other: Dict[str, List[str]],
    ) -> None:
        for split_name, uuids_other in splits_other.items():
            if split_name not in splits_self:
                splits_self[split_name] = []
            combined_uuids = set(splits_self[split_name]).union(uuids_other)
            splits_self[split_name] = list(combined_uuids)

    def _save_splits(self, splits: Dict[str, List[str]]) -> None:
        splits_path_self = self.metadata_path / "splits.json"
        with open(splits_path_self, "w") as f:
            json.dump(splits, f, indent=4)

    @overload
    def _load_df_offline(
        self,
        lazy: Literal[False] = ...,
        raise_when_empty: Literal[False] = ...,
        attempt_migration: bool = ...,
    ) -> Optional[pl.DataFrame]: ...

    @overload
    def _load_df_offline(
        self,
        lazy: Literal[False] = ...,
        raise_when_empty: Literal[True] = ...,
        attempt_migration: bool = ...,
    ) -> pl.DataFrame: ...

    @overload
    def _load_df_offline(
        self,
        lazy: Literal[True] = ...,
        raise_when_empty: Literal[False] = ...,
        attempt_migration: bool = ...,
    ) -> Optional[pl.LazyFrame]: ...

    @overload
    def _load_df_offline(
        self,
        lazy: Literal[True] = ...,
        raise_when_empty: Literal[True] = ...,
        attempt_migration: bool = ...,
    ) -> pl.LazyFrame: ...

    def _load_df_offline(
        self,
        lazy: bool = False,
        raise_when_empty: bool = False,
        attempt_migration: bool = True,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        """Loads the dataset DataFrame **always** from the local
        storage."""
        path = (
            self.base_path
            / "data"
            / self.team_id
            / "datasets"
            / self.dataset_name
            / "annotations"
        )

        if not path.exists():
            if raise_when_empty:
                raise FileNotFoundError(
                    f"Dataset '{self.dataset_name}' is empty."
                )
            return None

        if lazy:
            dfs = [pl.scan_parquet(file) for file in path.glob("*.parquet")]
            df = pl.concat(dfs) if dfs else None
        else:
            dfs = [pl.read_parquet(file) for file in path.glob("*.parquet")]
            df = pl.concat(dfs) if dfs else None

        if df is None and raise_when_empty:
            raise FileNotFoundError(f"Dataset '{self.dataset_name}' is empty.")

        if not attempt_migration or self.version == LDF_VERSION or df is None:
            return df
        return migrate_dataframe(df)  # pragma: no cover

    @overload
    def _get_file_index(
        self,
        lazy: Literal[False] = ...,
        sync_from_cloud: bool = ...,
        raise_when_empty: Literal[False] = ...,
    ) -> Optional[pl.DataFrame]: ...
    @overload
    def _get_file_index(
        self,
        lazy: Literal[False] = ...,
        sync_from_cloud: bool = ...,
        raise_when_empty: Literal[True] = ...,
    ) -> pl.DataFrame: ...

    @overload
    def _get_file_index(
        self,
        lazy: Literal[True] = ...,
        sync_from_cloud: bool = ...,
        raise_when_empty: Literal[False] = ...,
    ) -> Optional[pl.LazyFrame]: ...
    @overload
    def _get_file_index(
        self,
        lazy: Literal[True] = ...,
        sync_from_cloud: bool = ...,
        raise_when_empty: Literal[True] = ...,
    ) -> pl.LazyFrame: ...

    def _get_file_index(
        self,
        lazy: bool = False,
        sync_from_cloud: bool = False,
        raise_when_empty: bool = False,
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        """Loads the file index DataFrame from the local storage or the
        cloud if sync_from_cloud.

        @type lazy: bool
        @param lazy: Whether to return a LazyFrame instead of a
            DataFrame
        @type sync_from_cloud: bool
        @param sync_from_cloud: Whether to sync from cloud before
            loading the index
        """
        if sync_from_cloud:
            get_file(
                self.fs, "metadata/file_index.parquet", self.metadata_path
            )

        path = self.metadata_path / "file_index.parquet"
        if path is not None and path.exists():
            if not lazy:
                df = pl.read_parquet(path)
            else:
                df = pl.scan_parquet(path)

            return df.select(pl.all().exclude("^__index_level_.*$"))

        if raise_when_empty:
            raise FileNotFoundError(
                f"File index for dataset '{self.dataset_name}' is empty."
            )
        return None

    def _write_index(
        self,
        index: Optional[pl.DataFrame],
        new_index: Dict[str, List[str]],
        path: Optional[PathType] = None,
    ) -> None:
        path = Path(path or self.metadata_path / "file_index.parquet")
        df = pl.DataFrame(new_index)
        if index is not None:
            df = pl.concat([index, df])
        pq.write_table(df.to_arrow(), path)

    def _write_metadata(self) -> None:
        path = self.metadata_path / "metadata.json"
        path.write_text(self._metadata.model_dump_json(indent=4))
        with suppress(shutil.SameFileError):
            self.fs.put_file(path, "metadata/metadata.json")

    @staticmethod
    def _construct_url(
        bucket_storage: BucketStorage,
        bucket: str,
        team_id: str,
        dataset_name: str,
    ) -> str:
        """Constructs a URL for a remote dataset."""
        return f"{bucket_storage.value}://{bucket}/{team_id}/datasets/{dataset_name}"

    # TODO: Is the cache used anywhere at all?
    def _init_credentials(self) -> Dict[str, Any]:
        credentials_cache_file = self.base_path / "credentials.json"
        if credentials_cache_file.exists():
            return json.loads(credentials_cache_file.read_text())
        return {}

    def _get_metadata(self) -> Metadata:
        """Loads metadata from local storage or cloud, depending on the
        BucketStorage type.

        If loads from cloud it always downloads before loading.
        """
        if self.fs.exists("metadata/metadata.json"):
            path = get_file(
                self.fs,
                "metadata/metadata.json",
                self.metadata_path,
                default=self.metadata_path / "metadata.json",
            )
            metadata_json = json.loads(path.read_text())
            version = Version.parse(metadata_json.get("ldf_version", "1.0.0"))
            if version != LDF_VERSION:  # pragma: no cover
                return migrate_metadata(
                    metadata_json,
                    self._load_df_offline(lazy=True, attempt_migration=False),
                )
            return Metadata(**metadata_json)
        return Metadata(
            source=LuxonisSource(),
            ldf_version=str(LDF_VERSION),
            classes={},
            tasks={},
            skeletons={},
            categorical_encodings={},
            metadata_types={},
        )

    @property
    def is_remote(self) -> bool:
        return self.bucket_storage != BucketStorage.LOCAL

    @override
    def update_source(self, source: LuxonisSource) -> None:
        self._metadata.source = source
        self._write_metadata()

    @override
    def set_classes(
        self,
        classes: Union[List[str], Dict[str, int]],
        task: Optional[str] = None,
    ) -> None:
        if task is None:
            tasks = self.get_task_names()
        else:
            tasks = [task]

        for t in tasks:
            self._metadata.set_classes(classes, t)

        self._write_metadata()

    @override
    def get_classes(self) -> Dict[str, Dict[str, int]]:
        return self._metadata.classes

    def get_n_classes(self) -> Dict[str, int]:
        """Returns a mapping of task names to number of classes.

        @rtype: Dict[str, int]
        @return: A mapping from task names to number of classes.
        """
        return {
            task_name: len(classes)
            for task_name, classes in self.get_classes().items()
        }

    @override
    def set_skeletons(
        self,
        labels: Optional[List[str]] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        task: Optional[str] = None,
    ) -> None:
        if labels is None and edges is None:
            raise ValueError("Must provide either keypoint names or edges")

        if task is None:
            tasks = self.get_task_names()
        else:
            tasks = [task]
        for t in tasks:
            self._metadata.skeletons[t] = {
                "labels": labels or [],
                "edges": sorted(edges or []),
            }
        self._write_metadata()

    @override
    def get_skeletons(
        self,
    ) -> Dict[str, Tuple[List[str], List[Tuple[int, int]]]]:
        return {
            task: (skel["labels"], skel["edges"])
            for task, skel in self._metadata.skeletons.items()
        }

    @override
    def get_tasks(self) -> Dict[str, List[str]]:
        return self._metadata.tasks

    @override
    def set_tasks(self, tasks: Mapping[str, Iterable[str]]) -> None:
        self._metadata.tasks = {
            task_name: sorted(task_types)
            for task_name, task_types in tasks.items()
        }
        self._write_metadata()

    def get_categorical_encodings(
        self,
    ) -> Dict[str, Dict[str, int]]:
        return self._metadata.categorical_encodings

    def get_metadata_types(
        self,
    ) -> Dict[str, Literal["float", "int", "str", "Category"]]:
        return self._metadata.metadata_types

    def sync_from_cloud(
        self, update_mode: UpdateMode = UpdateMode.IF_EMPTY
    ) -> None:
        """Synchronizes the dataset from a remote cloud bucket to the
        local directory.

        This method performs the download only if local data is empty, or always downloads
        depending on the provided update_mode.

        @type update_mode: UpdateMode
        @param update_mode: Specifies the update behavior.
            - UpdateMode.IF_EMPTY: Downloads data only if the local dataset is empty.
            - UpdateMode.ALWAYS: Always downloads and overwrites the local dataset.
        """
        if not self.is_remote:
            logger.warning("This is a local dataset! Cannot sync from cloud.")
            return

        local_dir = self.base_path / "data" / self.team_id / "datasets"
        local_dir.mkdir(exist_ok=True, parents=True)

        lock_path = local_dir / ".sync.lock"

        with FileLock(str(lock_path)):  # DDP GCS training - multiple processes
            any_subfolder_empty = any(
                subfolder.is_dir() and not any(subfolder.iterdir())
                for subfolder in (local_dir / self.dataset_name).iterdir()
                if subfolder.is_dir()
            )
            if update_mode == UpdateMode.IF_EMPTY and not any_subfolder_empty:
                logger.info(
                    "Local dataset directory already exists. Skipping download."
                )
                return
            if update_mode == UpdateMode.ALWAYS or not self._is_synced:
                logger.info("Syncing from cloud...")
                self.fs.get_dir(remote_paths="", local_dir=local_dir)
                self._is_synced = True
            else:
                logger.warning(
                    "Already synced. Use update_mode=ALWAYS to resync."
                )

    @override
    def delete_dataset(self, *, delete_remote: bool = False) -> None:
        """Deletes the dataset from local storage and optionally from
        the cloud.

        @type delete_remote: bool
        @param delete_remote: Whether to delete the dataset from the
            cloud.
        """
        if not self.is_remote:
            shutil.rmtree(self.path)
            logger.info(f"Deleted dataset '{self.dataset_name}'")

        if self.is_remote and delete_remote:
            logger.info(f"Deleting dataset '{self.dataset_name}' from cloud")
            assert self.path
            assert self.dataset_name
            assert self.local_path
            if self.local_path.exists():
                shutil.rmtree(self.local_path)
            self.fs.delete_dir(allow_delete_parent=True)

    def _process_arrays(self, data_batch: List[DatasetRecord]) -> None:
        logger.info("Checking arrays...")
        task = self.progress.add_task(
            "[magenta]Processing arrays...", total=len(data_batch)
        )
        self.progress.start()
        uuid_dict = {}
        for record in data_batch:
            self.progress.update(task, advance=1)
            if record.annotation is None or record.annotation.array is None:
                continue
            ann = record.annotation.array
            if self.is_remote:
                uuid = self.fs.get_file_uuid(
                    ann.path, local=True
                )  # TODO: support from bucket
                uuid_dict[str(ann.path)] = uuid
                ann.path = Path(uuid).with_suffix(ann.path.suffix)
            else:
                ann.path = ann.path.absolute().resolve()
        self.progress.stop()
        self.progress.remove_task(task)
        if self.is_remote:
            logger.info("Uploading arrays...")
            # TODO: support from bucket (likely with a self.fs.copy_dir)
            self.fs.put_dir(
                local_paths=uuid_dict.keys(),
                remote_dir="arrays",
                uuid_dict=uuid_dict,
            )

    def _add_process_batch(
        self,
        data_batch: List[DatasetRecord],
        pfm: ParquetFileManager,
        index: Optional[pl.DataFrame],
        new_index: Dict[str, List[str]],
        processed_uuids: Set[str],
    ) -> None:
        paths = {data.file for data in data_batch}
        logger.info("Generating UUIDs...")
        # TODO: support from bucket
        uuid_dict = self.fs.get_file_uuids(paths, local=True)
        if self.is_remote:
            logger.info("Uploading media...")

            # TODO: support from bucket (likely with a self.fs.copy_dir)
            self.fs.put_dir(
                local_paths=paths, remote_dir="media", uuid_dict=uuid_dict
            )
            logger.info("Media uploaded")

        self._process_arrays(data_batch)

        task = self.progress.add_task(
            "[magenta]Processing data...", total=len(data_batch)
        )

        logger.info("Saving annotations...")
        with self.progress:
            for record in data_batch:
                filepath = record.file
                file = filepath.name
                uuid = uuid_dict[str(filepath)]
                matched_id = find_filepath_uuid(filepath, index)
                if matched_id is not None:
                    if matched_id != uuid:
                        # TODO: not sure if this should be an exception or how we should really handle it
                        raise ValueError(
                            f"{filepath} already added to the dataset! Please skip or rename the file."
                        )
                        # TODO: we may also want to check for duplicate uuids to get a one-to-one relationship
                elif uuid not in processed_uuids:
                    new_index["uuid"].append(uuid)
                    new_index["file"].append(file)
                    new_index["original_filepath"].append(
                        str(filepath.absolute().resolve())
                    )
                    processed_uuids.add(uuid)

                for row in record.to_parquet_rows():
                    pfm.write(uuid, row)

                self.progress.update(task, advance=1)
        self.progress.remove_task(task)

    def add(
        self, generator: DatasetIterator, batch_size: int = 1_000_000
    ) -> Self:
        logger.info(f"Adding data to dataset '{self.dataset_name}'...")
        index = self._get_file_index(sync_from_cloud=True)
        new_index = {"uuid": [], "file": [], "original_filepath": []}
        processed_uuids = set()

        data_batch: list[DatasetRecord] = []

        classes_per_task: Dict[str, Set[str]] = defaultdict(set)
        tasks: Dict[str, Set[str]] = defaultdict(set)
        categorical_encodings = defaultdict(dict)
        metadata_types = {}
        num_kpts_per_task: Dict[str, int] = {}

        annotations_path = get_dir(
            self.fs,
            "annotations",
            self.local_path,
            default=self.annotations_path,
        )
        assert annotations_path is not None

        with ParquetFileManager(annotations_path) as pfm:
            for i, record in enumerate(generator, start=1):
                if not isinstance(record, DatasetRecord):
                    record = DatasetRecord(**record)
                ann = record.annotation
                if ann is not None:
                    if not record.task_name:
                        record.task_name = infer_task(
                            record.task_name,
                            ann.class_name,
                            self.get_classes(),
                        )

                    def update_state(task_name: str, ann: Detection) -> None:
                        if ann.class_name is not None:
                            classes_per_task[task_name].add(ann.class_name)
                        elif not classes_per_task[task_name]:
                            classes_per_task[task_name] = set()

                        tasks[task_name] |= ann.get_task_types()

                        if ann.keypoints is not None:
                            num_kpts_per_task[task_name] = len(
                                ann.keypoints.keypoints
                            )
                        for name, value in ann.metadata.items():
                            task = f"{task_name}/metadata/{name}"
                            typ = type(value).__name__
                            if (
                                task in metadata_types
                                and metadata_types[task] != typ
                            ):
                                if {typ, metadata_types[task]} == {
                                    "int",
                                    "float",
                                }:
                                    metadata_types[task] = "float"
                                else:
                                    raise ValueError(
                                        f"Metadata type mismatch for {task}: {metadata_types[task]} and {typ}"
                                    )
                            else:
                                metadata_types[task] = typ

                            if not isinstance(value, Category):
                                continue
                            if value not in categorical_encodings[task]:
                                categorical_encodings[task][value] = len(
                                    categorical_encodings[task]
                                )
                        for name, sub_detection in ann.sub_detections.items():
                            update_state(f"{task_name}/{name}", sub_detection)

                    update_state(record.task_name, ann)

                data_batch.append(record)
                if i % batch_size == 0:
                    self._add_process_batch(
                        data_batch, pfm, index, new_index, processed_uuids
                    )
                    data_batch = []

            self._add_process_batch(
                data_batch, pfm, index, new_index, processed_uuids
            )

        with suppress(shutil.SameFileError):
            self.fs.put_dir(annotations_path, "")

        curr_classes = self.get_classes()
        for task, classes in classes_per_task.items():
            old_classes = set(curr_classes.get(task, []))
            new_classes = list(classes - old_classes)
            if new_classes or task not in curr_classes:
                logger.info(
                    f"Detected new classes for task group '{task}': {new_classes}"
                )

                self.set_classes(list(classes | old_classes), task=task)

        for task, num_kpts in num_kpts_per_task.items():
            self.set_skeletons(
                labels=[str(i) for i in range(num_kpts)],
                edges=[(i, i + 1) for i in range(num_kpts - 1)],
                task=task,
            )

        self._metadata.categorical_encodings = dict(categorical_encodings)
        self._metadata.metadata_types = metadata_types

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            self._write_index(index, new_index, path=tmp_file.name)

        self.fs.put_file(tmp_file.name, "metadata/file_index.parquet")
        self.set_tasks(tasks)
        self._write_metadata()
        self._warn_on_duplicates()
        return self

    def _warn_on_duplicates(self) -> None:
        df = self._load_df_offline(lazy=True)
        index = self._get_file_index(lazy=True, sync_from_cloud=self.is_remote)
        if df is None or index is None:
            return
        df = df.join(index, on="uuid").drop("file_right")
        warn_on_duplicates(df)

    def get_splits(self) -> Optional[Dict[str, List[str]]]:
        splits_path = get_file(
            self.fs, "metadata/splits.json", self.metadata_path
        )
        if splits_path is None:
            return None

        with open(splits_path) as file:
            return json.load(file)

    @deprecated(
        "ratios",
        "definitions",
        suggest={"ratios": "splits", "definitions": "splits"},
    )
    @override
    def make_splits(
        self,
        splits: Optional[
            Union[
                Mapping[str, Sequence[PathType]],
                Mapping[str, float],
                Tuple[float, float, float],
            ]
        ] = None,
        *,
        ratios: Optional[
            Union[Dict[str, float], Tuple[float, float, float]]
        ] = None,
        definitions: Optional[Dict[str, List[PathType]]] = None,
        replace_old_splits: bool = False,
    ) -> None:
        if ratios is not None and definitions is not None:
            raise ValueError("Cannot provide both ratios and definitions")

        if splits is None and ratios is None and definitions is None:
            splits = {"train": 0.8, "val": 0.1, "test": 0.1}

        if splits is not None:
            if ratios is not None or definitions is not None:
                raise ValueError(
                    "Cannot provide both splits and ratios/definitions"
                )
            if isinstance(splits, tuple):
                ratios = splits
            elif isinstance(splits, dict):
                value = next(iter(splits.values()))
                if isinstance(value, float):
                    ratios = splits  # type: ignore
                elif isinstance(value, list):
                    definitions = splits  # type: ignore

        if ratios is not None:
            if isinstance(ratios, tuple):
                if not len(ratios) == 3:
                    raise ValueError(
                        "Ratios must be a tuple of 3 floats for train, val, and test splits"
                    )
                ratios = {
                    "train": ratios[0],
                    "val": ratios[1],
                    "test": ratios[2],
                }
            sum_ = sum(ratios.values())
            if not math.isclose(sum_, 1.0):
                raise ValueError(f"Ratios must sum to 1.0, got {sum_:0.4f}")

        if definitions is not None:
            n_files = sum(map(len, definitions.values()))
            if n_files > len(self):
                raise ValueError(
                    "Dataset size is smaller than the total number of files in the definitions. "
                    f"Dataset size: {len(self)}, Definitions: {n_files}."
                )

        splits_to_update: List[str] = []
        new_splits: Dict[str, List[str]] = {}
        old_splits: Dict[str, List[str]] = defaultdict(list)

        splits_path = get_file(
            self.fs,
            "metadata/splits.json",
            self.metadata_path,
            default=self.metadata_path / "splits.json",
        )
        if splits_path.exists():
            with open(splits_path) as file:
                old_splits = defaultdict(list, json.load(file))

        defined_uuids = {
            uuid for uuids in old_splits.values() for uuid in uuids
        }

        if definitions is None:
            ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
            df = self._load_df_offline(raise_when_empty=True)
            ids = (
                df.filter(~pl.col("uuid").is_in(defined_uuids))
                .select("uuid")
                .unique()
                .sort("uuid")
                .get_column("uuid")
                .to_list()
            )
            if not ids:
                if not replace_old_splits:
                    raise ValueError(
                        "No new files to add to splits. "
                        "If you want to generate new splits, set `replace_old_splits=True`"
                    )
                ids = df.select("uuid").unique().get_column("uuid").to_list()
                old_splits = defaultdict(list)

            np.random.shuffle(ids)
            N = len(ids)
            lower_bound = 0
            for split, ratio in ratios.items():
                upper_bound = lower_bound + math.ceil(N * ratio)
                new_splits[split] = ids[lower_bound:upper_bound]
                splits_to_update.append(split)
                lower_bound = upper_bound

        else:
            index = self._get_file_index(sync_from_cloud=True)
            if index is None:
                raise FileNotFoundError("File index not found")
            for split, filepaths in definitions.items():
                splits_to_update.append(split)
                if not isinstance(filepaths, list):
                    raise TypeError(
                        "Must provide splits as a list of filepaths"
                    )
                ids = [
                    find_filepath_uuid(filepath, index, raise_on_missing=True)
                    for filepath in filepaths
                ]
                new_splits[split] = ids

        for split, uuids in new_splits.items():
            old_splits[split].extend(uuids)

        splits_path.write_text(json.dumps(old_splits, indent=4))

        with suppress(shutil.SameFileError):
            self.fs.put_file(splits_path, "metadata/splits.json")

    @staticmethod
    @override
    def exists(
        dataset_name: str,
        team_id: Optional[str] = None,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
        bucket: Optional[str] = None,
    ) -> bool:
        """Checks if a dataset exists.

        @type dataset_name: str
        @param dataset_name: Name of the dataset to check
        @type team_id: Optional[str]
        @param team_id: Optional team identifier
        @type bucket_storage: BucketStorage
        @param bucket_storage: Underlying bucket storage from C{local},
            C{S3}, or C{GCS}. Default is C{local}.
        @type bucket: Optional[str]
        @param bucket: Name of the bucket. Default is C{None}.
        @rtype: bool
        @return: Whether the dataset exists.
        """
        return dataset_name in LuxonisDataset.list_datasets(
            team_id, bucket_storage, bucket
        )

    @staticmethod
    def list_datasets(
        team_id: Optional[str] = None,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
        bucket: Optional[str] = None,
    ) -> List[str]:
        """Returns a list of all datasets.

        @type team_id: Optional[str]
        @param team_id: Optional team identifier
        @type bucket_storage: BucketStorage
        @param bucket_storage: Underlying bucket storage (local, S3, or
            GCS). Default is local.
        @type bucket: Optional[str]
        @param bucket: Name of the bucket. Default is None.
        @rtype: List[str]
        @return: List of all dataset names.
        """
        base_path = environ.LUXONISML_BASE_PATH
        team_id = team_id or environ.LUXONISML_TEAM_ID

        if bucket_storage == BucketStorage.LOCAL:
            fs = LuxonisFileSystem(
                f"file://{base_path}/data/{team_id}/datasets"
            )
        else:
            bucket = bucket or environ.LUXONISML_BUCKET
            if bucket is None:
                raise ValueError(
                    "Must set LUXONISML_BUCKET environment variable!"
                )
            fs = LuxonisFileSystem(
                LuxonisDataset._construct_url(
                    bucket_storage, bucket, team_id, ""
                )
            )
        if not fs.exists():
            return []

        def process_directory(path: PurePosixPath) -> Optional[str]:
            metadata_path = path / "metadata" / "metadata.json"
            if fs.exists(metadata_path):
                return path.name
            return None

        paths = (
            PurePosixPath(path)
            for path in fs.walk_dir("", recursive=False, typ="directory")
        )
        with ThreadPoolExecutor() as executor:
            return sorted(
                name for name in executor.map(process_directory, paths) if name
            )

    def export(
        self,
        output_path: PathType,
        dataset_type: DatasetType = DatasetType.COCO,
    ) -> Path:
        """Exportes the dataset into on of the supported formats.

        @type output_path: PathType
        @param output_path: Path to the directory where the dataset will
            be exported.
        @type dataset_type: DatasetType
        @param dataset_type: To what format to export the dataset.
            Currently only DatasetType.COCO is supported.
        @rtype: Path
        @return: Path to the zip file containing the exported dataset.
        """
        if dataset_type is not DatasetType.COCO:
            raise NotImplementedError(
                "Only COCO dataset export is supported at the moment"
            )
        logger.info(f"Exporting '{self.identifier}' to COCO format")
        from luxonis_ml.data import LuxonisLoader

        splits = self.get_splits()
        if splits is None:
            raise ValueError("Cannot export dataset without splits")
        if len(self.get_tasks()) > 1:
            raise NotImplementedError(
                "Only single-task datasets are supported for export"
            )
        output_path = Path(output_path)
        if output_path.exists():
            raise ValueError(
                f"Export path '{output_path}' already exists. Please remove it first."
            )
        output_path.mkdir(parents=True)
        classes = bidict(
            self.get_classes()[next(iter(self.get_tasks()))]
        ).inverse
        for split in splits:
            loader = LuxonisLoader(self, view=[split])
            if split == "val":
                split = "valid"
            split_path = output_path / self.identifier / split
            data_path = split_path / "data"
            data_path.mkdir(parents=True, exist_ok=True)
            coco_annotations = []
            coco_images = []
            coco_categories = [
                {"id": i, "name": name} for i, name in classes.items()
            ]
            skeletons = self.get_skeletons()
            if skeletons:
                labels, edges = next(iter(skeletons.values()))
                for cat in coco_categories:
                    cat["skeleton"] = [
                        [in_edge + 1, out_edge + 1]
                        for in_edge, out_edge in edges
                    ]
                    cat["keypoints"] = labels

            for image_id, (img, labels) in enumerate(loader):
                img_h, img_w = img.shape[:2]
                img_path = data_path / f"{image_id:06}.png"
                cv2.imwrite(
                    str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                )
                instances = defaultdict(dict)
                for task, arrays in labels.items():
                    task_type = get_task_type(task)
                    if task_type == "boundingbox":
                        for instance_id, arr in enumerate(arrays):
                            class_id = arr[0].astype(int).item()
                            instances[instance_id].update(
                                {"image_id": image_id, "category_id": class_id}
                            )
                            bbox = arr[1:]
                            bbox[::2] *= img_w
                            bbox[1::2] *= img_h
                            instances[instance_id]["bbox"] = bbox.tolist()

                    elif task_type == "keypoints":
                        for instance_id, arr in enumerate(arrays):
                            arr = arr.reshape(-1, 3)
                            arr[:, 0] *= img_w
                            arr[:, 1] *= img_h
                            instances[instance_id]["keypoints"] = arr.tolist()

                coco_annotations.extend(instances.values())

                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": img_path.name,
                        "width": img_w,
                        "height": img_h,
                    }
                )

            with open(split_path / "labels.json", "w") as f:
                json.dump(
                    {
                        "annotations": coco_annotations,
                        "images": coco_images,
                        "categories": coco_categories,
                    },
                    f,
                    indent=4,
                )

        zip_path = output_path / f"{self.identifier}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_path / self.identifier):
                for file in files:
                    file = Path(root, file)
                    zipf.write(file, file.relative_to(output_path))
        logger.info(f"Dataset '{self.identifier}' exported to '{zip_path}'")
        return zip_path
