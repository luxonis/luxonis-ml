import json
import math
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import cached_property
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, Literal, overload

import numpy as np
import polars as pl
from filelock import FileLock
from loguru import logger
from semver.version import Version
from typing_extensions import Self, override

from luxonis_ml.data.utils import (
    BucketStorage,
    BucketType,
    ParquetFileManager,
    UpdateMode,
    get_class_distributions,
    get_duplicates_info,
    get_heatmaps,
    get_missing_annotations,
    infer_task,
    merge_uuids,
    warn_on_duplicates,
)
from luxonis_ml.data.utils.constants import LDF_VERSION
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
from .source import LuxonisComponent, LuxonisSource
from .utils import (
    find_filepath_group_id,
    find_filepath_uuid,
    get_dir,
    get_file,
)


class LuxonisDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        team_id: str | None = None,
        bucket_type: BucketType
        | Literal["internal", "external"] = BucketType.INTERNAL,
        bucket_storage: BucketStorage
        | Literal["local", "gcs", "s3", "azure"] = BucketStorage.LOCAL,
        *,
        delete_local: bool = False,
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
        @type delete_local: bool
        @param delete_local: Whether to delete a dataset with the same
            name if it exists
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

        if delete_local or delete_remote:
            if self.exists(
                self.dataset_name,
                self.team_id,
                self.bucket_storage,
                self.bucket,
            ):
                self.delete_dataset(
                    delete_remote=delete_remote, delete_local=delete_local
                )

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
                group_id_val = row.get("group_id")
                if uuid_val is None:
                    raise ValueError("Missing 'uuid' in row!")

                data_dict = dict(row)
                data_dict.pop("uuid", None)
                data_dict.pop("group_id", None)

                pfm.write(uuid_val, data_dict, group_id_val)  # type: ignore

        logger.info(
            f"Saved DataFrame to Parquet files in '{annotations_path}'."
        )

    def _merge_metadata_with(self, other: "LuxonisDataset") -> None:
        """Merges relevant metadata from `other` into `self`."""
        self._metadata = self._metadata.merge_with(other._metadata)
        self._write_metadata()

    def clone(
        self,
        new_dataset_name: str,
        push_to_cloud: bool = True,
        splits_to_clone: list[str] | None = None,
        team_id: str | None = None,
    ) -> "LuxonisDataset":
        """Create a new LuxonisDataset that is a local copy of the
        current dataset. Cloned dataset will overwrite the existing
        dataset with the same name.

        @type new_dataset_name: str
        @param new_dataset_name: Name of the newly created dataset.
        @type push_to_cloud: bool
        @param push_to_cloud: Whether to push the new dataset to the
            cloud. Only if the current dataset is remote.
        @param splits_to_clone: list[str] | None
        @type splits_to_clone: Optional list of split names to clone. If
            None, all data will be cloned.
        @type team_id: str | None
        @param team_id: Optional team identifier.
        """
        if team_id is None:
            team_id = self.team_id

        new_dataset = LuxonisDataset(
            dataset_name=new_dataset_name,
            team_id=team_id,
            bucket_type=self.bucket_type,
            bucket_storage=self.bucket_storage,
            delete_local=True,
            delete_remote=True,
        )

        if self.is_remote:
            self.pull_from_cloud(update_mode=UpdateMode.MISSING)

        new_dataset_path = Path(new_dataset.local_path)
        new_dataset_path.mkdir(parents=True, exist_ok=True)

        if splits_to_clone is not None:
            df_self = self._load_df_offline(raise_when_empty=True)
            splits_self = self._load_splits(self.metadata_path)
            uuids_to_clone = {
                uid
                for split in splits_to_clone
                for uid in splits_self.get(split, [])
            }
            df_self = df_self.filter(df_self["uuid"].is_in(uuids_to_clone))
            splits_self = {
                k: v for k, v in splits_self.items() if k in splits_to_clone
            }

        shutil.copytree(
            self.local_path,
            new_dataset.local_path,
            dirs_exist_ok=True,
            ignore=lambda d, n: self._ignore_files_not_in_uuid_set(
                d, n, uuids_to_clone if splits_to_clone else set()
            ),
        )

        if splits_to_clone is not None:
            new_dataset._save_df_offline(df_self)
            new_dataset._save_splits(splits_self)

        new_dataset._init_paths()
        new_dataset._metadata = self._get_metadata()

        new_dataset._metadata.parent_dataset = self.dataset_name

        if push_to_cloud:
            if self.is_remote:
                new_dataset.push_to_cloud(
                    update_mode=UpdateMode.MISSING,
                    bucket_storage=self.bucket_storage,
                )
            else:
                logger.warning(
                    f"Cannot push to cloud. The cloned dataset '{new_dataset.dataset_name}' is local. "
                )

        new_dataset._write_metadata()

        return new_dataset

    def merge_with(
        self,
        other: "LuxonisDataset",
        inplace: bool = True,
        new_dataset_name: str | None = None,
        splits_to_merge: list[str] | None = None,
        team_id: str | None = None,
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
        @type splits_to_merge: list[str] | None
        @param splits_to_merge: Optional list of split names to merge.
        @type team_id: str | None
        @param team_id: Optional team identifier.
        """
        if inplace:
            target_dataset = self
        elif new_dataset_name:
            if self.bucket_storage != other.bucket_storage:
                raise ValueError(
                    "Cannot merge datasets with different bucket storage types."
                )
            target_dataset = self.clone(
                new_dataset_name, push_to_cloud=False, team_id=team_id
            )
        else:
            raise ValueError(
                "You must specify a name for the new dataset "
                "when inplace is False"
            )

        if self.is_remote:
            other.pull_from_cloud(UpdateMode.MISSING)
            self.pull_from_cloud(UpdateMode.MISSING)

        df_self = self._load_df_offline(raise_when_empty=True)
        df_other = other._load_df_offline(raise_when_empty=True)
        duplicate_group_ids = set(df_self["group_id"]).intersection(
            df_other["group_id"]
        )
        if duplicate_group_ids:
            logger.warning(
                f"Found {len(duplicate_group_ids)} duplicate group ID's in the datasets. "
                "Merging will remove these duplicates from the incoming dataset."
            )
            df_other = df_other.filter(
                ~df_other["group_id"].is_in(duplicate_group_ids)
            )

        splits_self = self._load_splits(self.metadata_path)
        splits_other = self._load_splits(other.metadata_path)
        if splits_to_merge is not None:
            uuids_to_merge = {
                uuid
                for split_name in splits_to_merge
                for uuid in splits_other.get(split_name, [])
            }
            df_other = df_other.filter(df_other["uuid"].is_in(uuids_to_merge))
            splits_other = {
                k: v for k, v in splits_other.items() if k in splits_to_merge
            }

        df_merged = pl.concat([df_self, df_other])
        target_dataset._save_df_offline(df_merged)

        splits_other = {
            split_name: [
                group_id
                for group_id in group_ids
                if group_id not in duplicate_group_ids
            ]
            for split_name, group_ids in splits_other.items()
        }
        self._merge_splits(splits_self, splits_other)
        target_dataset._save_splits(splits_self)

        if self.is_remote:
            shutil.copytree(
                other.media_path,
                target_dataset.media_path,
                dirs_exist_ok=True,
                ignore=lambda d, n: self._ignore_files_not_in_uuid_set(
                    d, n, uuids_to_merge if splits_to_merge else set()
                ),
            )
            target_dataset.push_to_cloud(
                bucket_storage=target_dataset.bucket_storage,
                update_mode=UpdateMode.MISSING,
            )

        for entry in (
            df_other.select(["uuid", "file"])
            .unique(subset=["uuid"])
            .to_dicts()
        ):
            uid, rel_file = entry["uuid"], entry["file"]
            src_path = other.media_path / f"{uid}{Path(rel_file).suffix}"
            dst_path = target_dataset.media_path / src_path.name
            if src_path.exists() and not dst_path.exists():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_path, dst_path)

        target_dataset._merge_metadata_with(other)

        return target_dataset

    def _load_splits(self, path: Path) -> dict[str, list[str]]:
        splits_path = path / "splits.json"
        with open(splits_path) as f:
            return json.load(f)

    def _ignore_files_not_in_uuid_set(
        self,
        dir_path: PathLike[str] | str,
        names: list[str],
        uuids_to_keep: set[str],
    ) -> set[str]:
        if not uuids_to_keep:
            return set()
        ignored: set[str] = set()
        for name in names:
            full = Path(dir_path) / name
            if full.is_file() and full.stem not in uuids_to_keep:
                ignored.add(name)
        return ignored

    def _merge_splits(
        self,
        splits_self: dict[str, list[str]],
        splits_other: dict[str, list[str]],
    ) -> None:
        for split_name, group_ids_other in splits_other.items():
            if split_name not in splits_self:
                splits_self[split_name] = []
            combined_group_ids = set(splits_self[split_name]).union(
                group_ids_other
            )
            splits_self[split_name] = list(combined_group_ids)

    def _save_splits(self, splits: dict[str, list[str]]) -> None:
        splits_path_self = self.metadata_path / "splits.json"
        with open(splits_path_self, "w") as f:
            json.dump(splits, f, indent=4)

    @overload
    def _load_df_offline(
        self,
        lazy: Literal[False] = ...,
        raise_when_empty: Literal[False] = ...,
        attempt_migration: bool = ...,
    ) -> pl.DataFrame | None: ...

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
    ) -> pl.LazyFrame | None: ...

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
    ) -> pl.DataFrame | pl.LazyFrame | None:
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
        files = list(path.glob("*.parquet"))
        if not files:
            if raise_when_empty:
                raise FileNotFoundError(
                    f"Dataset '{self.dataset_name}' is empty."
                )
            return None

        lazy_df = pl.scan_parquet([str(f) for f in files])

        if lazy:
            return lazy_df

        df = lazy_df.collect()
        if df.is_empty() and raise_when_empty:
            raise FileNotFoundError(f"Dataset '{self.dataset_name}' is empty.")

        if attempt_migration and self.version != LDF_VERSION:
            df = migrate_dataframe(df)

        return df

    @overload
    def _get_index(
        self,
        lazy: Literal[False] = ...,
        raise_when_empty: Literal[False] = ...,
    ) -> pl.DataFrame | None: ...

    @overload
    def _get_index(
        self,
        lazy: Literal[False] = ...,
        raise_when_empty: Literal[True] = ...,
    ) -> pl.DataFrame: ...

    @overload
    def _get_index(
        self,
        lazy: Literal[True] = ...,
        raise_when_empty: Literal[False] = ...,
    ) -> pl.LazyFrame | None: ...

    def _get_index(
        self,
        lazy: bool = False,
        raise_when_empty: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Loads unique file entries from annotation data."""
        df = self._load_df_offline(
            lazy=True, raise_when_empty=raise_when_empty
        )
        if df is None:
            return None

        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        unique_files = df.select(pl.col("file")).unique().collect()
        files: list[str] = unique_files["file"].to_list()

        def resolve_path(p: str) -> str:
            return str(Path(p).resolve())

        with ThreadPoolExecutor() as pool:
            resolved_paths = list(pool.map(resolve_path, files))
        mapping: dict[str, str] = dict(zip(files, resolved_paths, strict=True))

        processed = (
            df.with_columns(
                [
                    pl.col("uuid"),
                    pl.col("file")
                    .map_dict(mapping, default=None)
                    .alias("original_filepath"),
                ]
            )
            .unique(
                subset=["uuid", "original_filepath", "group_id"],
                maintain_order=False,
            )
            .select(["uuid", "original_filepath", "group_id"])
        )

        if not lazy:
            processed = processed.collect()

        return processed

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
    def _init_credentials(self) -> dict[str, Any]:
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

            # TODO: Remove this LuxonisSource and df migration in the future
            if "source" in metadata_json:
                source_data = metadata_json["source"]
                if isinstance(source_data, dict):
                    components = source_data.get("components", {})
                    if isinstance(components, dict) and len(components) == 1:
                        new_components = {
                            "image": LuxonisComponent(
                                **next(iter(components.values()))
                            )
                        }
                        metadata_json["source"]["components"] = new_components
                        metadata_json["source"]["main_component"] = "image"
            df = self._load_df_offline(lazy=False, attempt_migration=False)
            if df is not None and "group_id" not in df.columns:
                df = df.with_columns(pl.col("uuid").alias("group_id"))
                self._save_df_offline(df)

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
    def get_source_names(self) -> list[str]:
        return list(self.source.components.keys())

    @override
    def set_classes(
        self,
        classes: list[str] | dict[str, int],
        task: str | None = None,
    ) -> None:
        if task is None:
            tasks = self.get_task_names()
        else:
            tasks = [task]

        for t in tasks:
            self._metadata.set_classes(classes, t)

        self._write_metadata()

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        return self._metadata.classes

    def get_n_classes(self) -> dict[str, int]:
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
        labels: list[str] | None = None,
        edges: list[tuple[int, int]] | None = None,
        task: str | None = None,
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
    ) -> dict[str, tuple[list[str], list[tuple[int, int]]]]:
        return {
            task: (skel["labels"], skel["edges"])
            for task, skel in self._metadata.skeletons.items()
        }

    @override
    def get_tasks(self) -> dict[str, list[str]]:
        return self._metadata.tasks

    @override
    def set_tasks(self, tasks: Mapping[str, Iterable[str]]) -> None:
        if len(tasks) == 0:
            return
        self._metadata.tasks = {
            task_name: sorted(task_types)
            for task_name, task_types in tasks.items()
        }
        self._write_metadata()

    def get_categorical_encodings(
        self,
    ) -> dict[str, dict[str, int]]:
        return self._metadata.categorical_encodings

    def get_metadata_types(
        self,
    ) -> dict[str, Literal["float", "int", "str", "Category"]]:
        return self._metadata.metadata_types

    def pull_from_cloud(
        self, update_mode: UpdateMode = UpdateMode.MISSING
    ) -> None:
        """Synchronizes the dataset from a remote cloud bucket to the
        local directory.

        This method performs the download only if some local dataset media files are missing, or always downloads
        depending on the provided update_mode.

        @type update_mode: UpdateMode
        @param update_mode: Specifies the update behavior.
            - UpdateMode.MISSING: Downloads only the missing media files for the dataset.
            - UpdateMode.ALL: Always downloads and overwrites all media files in the local dataset.
        """
        if not self.is_remote:
            logger.warning("This is a local dataset! Cannot sync from cloud.")
            return

        local_dir = self.base_path / "data" / self.team_id / "datasets"
        local_dir.mkdir(exist_ok=True, parents=True)

        lock_path = local_dir / ".sync.lock"

        with FileLock(str(lock_path)):  # DDP GCS training - multiple processes
            logger.info(
                "Pulling remote's dataset annotations and metadata to local dataset ..."
            )
            for dir_name in ["annotations", "metadata"]:
                _ = get_dir(self.fs, dir_name, self.local_path)

            index = self._get_index(lazy=False)
            missing_media_paths = []
            if index is not None:
                uuids = index["uuid"].to_list()
                origps = index["original_filepath"].to_list()

                media_root = Path(local_dir) / self.dataset_name / "media"

                missing_media_paths = [
                    f"media/{uid}{Path(orig).suffix}"
                    for uid, orig in zip(uuids, origps, strict=True)
                    if not Path(orig).exists()
                    and not (media_root / f"{uid}{Path(orig).suffix}").exists()
                ]

            if update_mode == UpdateMode.ALL:
                logger.info("Force-pulling all media files...")
                self.fs.get_dir(remote_paths="", local_dir=local_dir)
            elif update_mode == UpdateMode.MISSING and missing_media_paths:
                logger.info(
                    f"Pulling {len(missing_media_paths)} missing files..."
                )
                self.fs.get_dir(
                    remote_paths=missing_media_paths,
                    local_dir=local_dir / f"{self.dataset_name}" / "media",
                )
            else:
                logger.info("Media already synced")

    def push_to_cloud(
        self,
        bucket_storage: BucketStorage,
        update_mode: UpdateMode = UpdateMode.MISSING,
    ) -> None:
        """Pushes the local dataset to a remote cloud bucket.

        This method performs the pushing only for missing cloud media files or always pushes
        depending on the provided update_mode.

        @type update_mode: UpdateMode
        @param update_mode: Specifies the update behavior. Annotations and metadata are always pushed.
            - UpdateMode.MISSING: Pushes only the missing media files for the dataset.
            - UpdateMode.ALL:  Always pushes and overwrites all media files in the cloud dataset.
        """
        index = self._get_index(lazy=False)

        if index is None:
            raise ValueError(
                "Cannot push to cloud. The dataset is empty or not initialized."
            )

        dataset = LuxonisDataset(
            dataset_name=self.dataset_name,
            team_id=self.team_id,
            bucket_type=self.bucket_type,
            bucket_storage=bucket_storage,
            delete_local=False,
            delete_remote=False,
        )

        bucket_uuids = (
            [
                PurePosixPath(path).stem
                for path in dataset.fs.walk_dir(
                    "media", recursive=False, typ="file"
                )
            ]
            if dataset.fs.exists("media")
            else []
        )

        missing_df = index.filter(~pl.col("uuid").is_in(bucket_uuids))

        missing_uuid_dict = {}
        for original_path, uuid in zip(
            missing_df["original_filepath"].to_list(),
            missing_df["uuid"].to_list(),
            strict=True,
        ):
            if not Path(original_path).exists():
                suffix = Path(original_path).suffix
                fallback_path = self.local_path / "media" / f"{uuid}{suffix}"
                if fallback_path.exists():
                    missing_uuid_dict[str(fallback_path)] = uuid
                else:
                    raise FileNotFoundError(
                        f"File {original_path} and {fallback_path} do not exist!"
                    )
            else:
                missing_uuid_dict[original_path] = uuid

        for dir_name in ["annotations", "metadata"]:
            dataset.fs.put_dir(
                local_paths=self.local_path / dir_name,
                remote_dir=dir_name,
                copy_contents=True,
            )

        if update_mode == UpdateMode.ALL:
            logger.info("Force-pushing all media files...")
            dataset.fs.put_dir(
                local_paths=self.local_path / "media", remote_dir="media"
            )
        elif update_mode == UpdateMode.MISSING and missing_uuid_dict:
            logger.info(
                f"Pushing {len(missing_uuid_dict)} missing files to cloud..."
            )
            dataset.fs.put_dir(
                local_paths=missing_uuid_dict.keys(),
                remote_dir="media",
                uuid_dict=missing_uuid_dict,
            )
        else:
            logger.info("Media already synced")

    @override
    def delete_dataset(
        self, *, delete_remote: bool = False, delete_local: bool = False
    ) -> None:
        """Deletes the dataset from local storage and optionally from
        the cloud.

        @type delete_remote: bool
        @param delete_remote: Whether to delete the dataset from the
            cloud.
        @type delete_local: bool
        @param delete_local: Whether to delete the dataset from local
            storage.
        """
        if not (delete_remote or delete_local):
            raise ValueError(
                "Must set delete_remote=True and/or delete_local=True when calling delete_dataset()"
            )

        if not self.is_remote and delete_local:
            logger.info(
                f"Deleting local dataset '{self.dataset_name}' from local storage"
            )
            shutil.rmtree(self.path)

        if self.is_remote and delete_remote:
            logger.info(
                f"Deleting remote dataset '{self.dataset_name}' from cloud storage"
            )
            assert self.path
            assert self.dataset_name
            assert self.local_path
            self.fs.delete_dir(allow_delete_parent=True)

        if self.is_remote and delete_local:
            logger.info(
                f"Deleting remote dataset '{self.dataset_name}' from local storage"
            )
            if self.local_path.exists():
                shutil.rmtree(self.local_path)

    def _process_arrays(self, data_batch: list[DatasetRecord]) -> None:
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
                uuid = self.fs.get_file_uuid(ann.path, local=True)
                uuid_dict[str(ann.path)] = uuid
                ann.path = Path(uuid).with_suffix(ann.path.suffix)
            else:
                ann.path = ann.path.absolute().resolve()
        self.progress.stop()
        self.progress.remove_task(task)
        if self.is_remote:
            logger.info("Uploading arrays...")
            self.fs.put_dir(
                local_paths=uuid_dict.keys(),
                remote_dir="arrays",
                uuid_dict=uuid_dict,
            )

    def _add_process_batch(
        self,
        data_batch: list[DatasetRecord],
        pfm: ParquetFileManager,
        index: pl.DataFrame | None,
    ) -> None:
        paths = {path for data in data_batch for path in data.all_file_paths}
        logger.info("Generating UUIDs...")
        uuid_dict = self.fs.get_file_uuids(paths, local=True)

        overwrite_uuids = set()
        for file_path in paths:
            matched_id = find_filepath_uuid(file_path, index)
            if matched_id is not None:
                overwrite_uuids.add(matched_id)
                logger.warning(
                    f"File {file_path} with UUID: {matched_id} already existed in the dataset from previous dataset.add() call. "
                    "Old data will be overwritten with the new data."
                )

        if overwrite_uuids:
            pfm.remove_duplicate_uuids(overwrite_uuids)

        if self.is_remote:
            logger.info("Uploading media...")

            self.fs.put_dir(
                local_paths=paths,
                remote_dir="media",
                uuid_dict=dict(uuid_dict),
            )
            logger.info("Media uploaded")

        self._process_arrays(data_batch)

        task = self.progress.add_task(
            "[magenta]Processing data...", total=len(data_batch)
        )

        logger.info("Saving annotations...")
        with self.progress:
            for record in data_batch:
                file_paths = record.all_file_paths
                uuid_list = [
                    uuid_dict[str(file_path)] for file_path in file_paths
                ]
                group_id = (
                    str(merge_uuids(uuid_list))
                    if len(uuid_list) > 1
                    else str(uuid_list[0])
                )
                for row in record.to_parquet_rows():
                    pfm.write(uuid_dict[row["file"]], row, group_id)
                self.progress.update(task, advance=1)
        self.progress.remove_task(task)

    def add(
        self, generator: DatasetIterator, batch_size: int = 1_000_000
    ) -> Self:
        logger.info(f"Adding data to dataset '{self.dataset_name}'...")

        data_batch: list[DatasetRecord] = []

        classes_per_task: dict[str, set[str]] = defaultdict(set)
        tasks: dict[str, set[str]] = defaultdict(set)
        categorical_encodings = defaultdict(dict)
        metadata_types = {}
        num_kpts_per_task: dict[str, int] = {}
        sources: set[str] = set()

        annotations_path = get_dir(
            self.fs,
            "annotations",
            self.local_path,
            default=self.annotations_path,
        )

        index = self._get_index()

        assert annotations_path is not None

        with ParquetFileManager(annotations_path, batch_size) as pfm:
            for i, record in enumerate(generator, start=1):
                if not isinstance(record, DatasetRecord):
                    record = DatasetRecord(**record)
                sources.update(record.files.keys())
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
                    self._add_process_batch(data_batch, pfm, index)
                    data_batch = []

            self._add_process_batch(data_batch, pfm, index)

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
        self.set_tasks(tasks)
        if sources:
            components = {
                source_name: LuxonisComponent(
                    name=source_name,
                )
                for source_name in sources
            }
            source = LuxonisSource(
                components=components,
                main_component=next(iter(components.keys())),
            )
            self.update_source(source)
        self._warn_on_duplicates()
        return self

    def _warn_on_duplicates(self) -> None:
        df = self._load_df_offline(lazy=True)
        if df is None:
            return
        warn_on_duplicates(df)

    def get_splits(self) -> dict[str, list[str]] | None:
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
        splits: Mapping[str, Sequence[PathType]]
        | Mapping[str, float]
        | tuple[float, float, float]
        | None = None,
        *,
        ratios: dict[str, float] | tuple[float, float, float] | None = None,
        definitions: dict[str, list[PathType]] | None = None,
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
                if not splits:
                    raise ValueError("Splits cannot be empty")
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

        splits_to_update: list[str] = []
        new_splits: dict[str, list[str]] = {}
        old_splits: dict[str, list[str]] = defaultdict(list)

        splits_path = get_file(
            self.fs,
            "metadata/splits.json",
            self.metadata_path,
            default=self.metadata_path / "splits.json",
        )
        if splits_path.exists():
            with open(splits_path) as file:
                old_splits = defaultdict(list, json.load(file))

        defined_group_ids = {
            group_id
            for group_ids in old_splits.values()
            for group_id in group_ids
        }

        if definitions is None:
            ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
            df = self._load_df_offline(raise_when_empty=True)
            ids = (
                df.filter(~pl.col("group_id").is_in(defined_group_ids))
                .select("group_id")
                .unique()
                .sort("group_id")
                .get_column("group_id")
                .to_list()
            )
            if not ids:
                if not replace_old_splits:
                    raise ValueError(
                        "No new files to add to splits. "
                        "If you want to generate new splits, set `replace_old_splits=True`"
                    )
                ids = (
                    df.select("group_id")
                    .unique()
                    .get_column("group_id")
                    .to_list()
                )
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
            index = self._get_index()
            if index is None:
                raise FileNotFoundError("File index not found")
            for split, filepaths in definitions.items():
                splits_to_update.append(split)
                if not isinstance(filepaths, list):
                    raise TypeError(
                        "Must provide splits as a list of filepaths"
                    )
                ids = [
                    find_filepath_group_id(
                        filepath, index, raise_on_missing=True
                    )
                    for filepath in filepaths
                ]
                new_splits[split] = list(set(ids))

        for split, group_ids in new_splits.items():
            old_splits[split].extend(group_ids)

        splits_path.write_text(json.dumps(old_splits, indent=4))

        with suppress(shutil.SameFileError):
            self.fs.put_file(splits_path, "metadata/splits.json")

    @staticmethod
    @override
    def exists(
        dataset_name: str,
        team_id: str | None = None,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
        bucket: str | None = None,
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
        team_id: str | None = None,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
        bucket: str | None = None,
    ) -> list[str]:
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

        def process_directory(path: PurePosixPath) -> str | None:
            metadata_path = path / "metadata"
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
        dataset_type: DatasetType = DatasetType.NATIVE,
        max_partition_size_gb: float | None = None,
        zip_output: bool = False,
    ) -> Path | list[Path]:
        """Exports the dataset into one of the supported formats.

        @type output_path: PathType
        @param output_path: Path to the directory where the dataset will
            be exported.
        @type dataset_type: DatasetType
        @param dataset_type: To what format to export the dataset.
            Currently only DatasetType.NATIVE is supported.
        @type max_partition_size_gb: Optional[float]
        @param max_partition_size_gb: Maximum size of each partition in
            GB. If the dataset exceeds this size, it will be split into
            multiple partitions named
            {dataset_name}_part{partition_number}. Default is None,
            meaning the dataset will be exported as a single partition
            named {dataset_name}.
        @type zip_output: bool
        @param zip_output: Whether to zip the exported dataset (or each
            partition) after export. Default is False.
        @rtype: Union[Path, List[Path]]
        @return: Path(s) to the ZIP file(s) containing the exported
            dataset.
        """

        def _dump_annotations(
            annotations: dict[str, list[Any]],
            output_path: Path,
            identifier: str,
            part: int | None = None,
        ) -> None:
            for split_name, annotation_data in annotations.items():
                if part is not None:
                    split_path = (
                        output_path / f"{identifier}_part{part}" / split_name
                    )
                else:
                    split_path = output_path / identifier / split_name
                split_path.mkdir(parents=True, exist_ok=True)
                with open(split_path / "annotations.json", "w") as f:
                    json.dump(annotation_data, f, indent=4)

        def resolve_path(
            img_path: str | Path, uuid: str, media_path: str
        ) -> str:
            img_path = Path(img_path)
            if img_path.exists():
                return str(img_path)

            ext = img_path.suffix.lstrip(".")
            fallback = Path(media_path) / f"{uuid}.{ext}"
            if not fallback.exists():
                raise FileNotFoundError(f"Missing image: {fallback}")
            return str(fallback)

        if dataset_type is not DatasetType.NATIVE:
            raise NotImplementedError(
                "Only 'NATIVE' dataset export is supported at the moment"
            )
        logger.info(
            f"Exporting '{self.identifier}' to '{dataset_type.name}' format"
        )

        splits = self.get_splits()
        if splits is None:
            raise ValueError("Cannot export dataset without splits")

        output_path = Path(output_path)
        if output_path.exists():
            raise ValueError(
                f"Export path '{output_path}' already exists. Please remove it first."
            )
        output_path.mkdir(parents=True)
        image_indices = {}
        annotations = {"train": [], "val": [], "test": []}
        df = self._load_df_offline(raise_when_empty=True)

        # Capture the original order. Assume annotations are ordered if instance_id's were not specified.
        df = df.with_row_count("row_idx").with_columns(
            pl.col("row_idx").min().over("file").alias("first_occur")
        )

        # Resolve file paths to ensure they are absolute and exist
        df = df.with_columns(
            pl.struct(["file", "uuid"])
            .map_elements(
                lambda row: resolve_path(
                    row["file"], row["uuid"], str(self.media_path)
                ),
                return_dtype=pl.Utf8,
            )
            .alias("file")
        )

        grouped_image_sources = df.select(
            "group_id", "source_name", "file"
        ).unique()

        # Filter out rows without annotations and ensure we have at least one row per group_id (images without annotations)
        df = (
            df.with_columns(
                [
                    pl.col("annotation").is_not_null().alias("has_annotation"),
                    pl.col("group_id")
                    .cumcount()
                    .over("group_id")
                    .alias("first_occur"),
                ]
            )
            .pipe(
                lambda df: (
                    df.filter(pl.col("has_annotation")).vstack(
                        df.filter(
                            ~pl.col("group_id").is_in(
                                df.filter(pl.col("has_annotation"))
                                .select("group_id")
                                .unique()["group_id"]
                            )
                        ).unique(subset=["group_id"], keep="first")
                    )
                )
            )
            .sort(["row_idx"])
            .select(
                [
                    col
                    for col in df.columns
                    if col not in ["has_annotation", "row_idx", "first_occur"]
                ]
            )
        )

        splits = self.get_splits()
        assert splits is not None

        current_size = 0
        part = 0 if max_partition_size_gb else None
        max_partition_size = (
            max_partition_size_gb * 1024**3 if max_partition_size_gb else None
        )

        # Group the full dataframe by group_id
        df = df.group_by("group_id", maintain_order=True)
        copied_files = set()

        for group_id, group_df in df:
            matched_df = grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            group_source_names = matched_df.get_column("source_name").to_list()

            split = next(
                (
                    s
                    for s, group_ids in splits.items()
                    if group_id in group_ids
                ),
                None,
            )
            assert split is not None

            group_total_size = sum(Path(f).stat().st_size for f in group_files)
            annotation_records = []

            for row in group_df.iter_rows(named=True):
                task_name = row["task_name"]
                class_name = row["class_name"]
                instance_id = row["instance_id"]
                task_type = row["task_type"]
                ann_str = row["annotation"]

                source_to_file = {
                    name: str(
                        (
                            Path("images")
                            / f"{image_indices.setdefault(Path(f), len(image_indices))}{Path(f).suffix}"
                        ).as_posix()
                    )
                    for name, f in zip(
                        group_source_names, group_files, strict=True
                    )
                }

                record = {
                    "files" if len(group_source_names) > 1 else "file": (
                        source_to_file
                        if len(group_source_names) > 1
                        else source_to_file[group_source_names[0]]
                    ),
                    "task_name": task_name,
                }

                if ann_str is not None:
                    data = json.loads(ann_str)
                    annotation_base = {
                        "instance_id": instance_id,
                        "class": class_name,
                    }
                    if task_type in {
                        "instance_segmentation",
                        "segmentation",
                        "boundingbox",
                        "keypoints",
                    }:
                        annotation_base[task_type] = data
                    elif task_type.startswith("metadata/"):
                        annotation_base["metadata"] = {task_type[9:]: data}
                    record["annotation"] = annotation_base

                annotation_records.append(record)

            annotations_size = sum(
                sys.getsizeof(r) for r in annotation_records
            )

            if (
                max_partition_size
                and part is not None
                and current_size + group_total_size + annotations_size
                > max_partition_size
            ):
                _dump_annotations(
                    annotations, output_path, self.identifier, part
                )
                current_size = 0
                part += 1
                annotations = {"train": [], "val": [], "test": []}

            if max_partition_size:
                data_path = (
                    output_path
                    / f"{self.identifier}_part{part}"
                    / split
                    / "images"
                )
            else:
                data_path = output_path / self.identifier / split / "images"
            data_path.mkdir(parents=True, exist_ok=True)

            for file in group_files:
                file_path = Path(file)
                if file_path not in copied_files:
                    copied_files.add(file_path)
                    image_index = image_indices[file_path]
                    dest_file = data_path / f"{image_index}{file_path.suffix}"
                    shutil.copy(file_path, dest_file)
                    current_size += file_path.stat().st_size

            annotations[split].extend(annotation_records)
            current_size += annotations_size

        _dump_annotations(annotations, output_path, self.identifier, part)

        if zip_output:
            archives = []
            if max_partition_size:
                assert part is not None
                for i in range(part + 1):
                    folder = output_path / f"{self.identifier}_part{i}"
                    if folder.exists():
                        archive_file = shutil.make_archive(
                            str(folder), "zip", root_dir=folder
                        )
                        archives.append(Path(archive_file))
            else:
                folder = output_path / self.identifier
                if folder.exists():
                    archive_file = shutil.make_archive(
                        str(folder), "zip", root_dir=folder
                    )
                    archives.append(Path(archive_file))
            if len(archives) > 1:
                logger.info(
                    f"Dataset successfully exported to: {[str(p) for p in archives]}"
                )
                return archives
            logger.info(f"Dataset successfully exported to: {archives[0]}")
            return archives[0]

        logger.info(f"Dataset successfully exported to: {output_path}")
        return output_path

    def get_statistics(
        self, sample_size: int | None = None, view: str | None = None
    ) -> dict[str, Any]:
        """Returns comprehensive dataset statistics as a structured
        dictionary for the given view or the entire dataset.

        The returned dictionary contains:

            - "duplicates": Analysis of duplicated content
                - "duplicate_uuids": List of {"uuid": str, "files": List[str]} for images with same UUID
                - "duplicate_annotations": List of repeated annotations with file_name, task_name, task_type,
                    annotation content, and count

            - "class_distributions": Nested dictionary of class frequencies organized by task_name and task_type
            (excludes classification tasks)

            - "missing_annotations": List of file paths that exist in the dataset but lack annotations

            - "heatmaps": Spatial distribution of annotations as 15x15 grid matrices organized by task_name and task_type

        @type sample_size: Optional[int]
        @param sample_size: Number of samples to use for heatmap generation
        @type view: Optional[str]
        @param view: Name of the view to analyze. If None, the entire dataset is analyzed.
        @rtype: Dict[str, Any]
        @return: Dataset statistics dictionary as described above
        """
        df = self._load_df_offline(lazy=True)

        stats = {
            "duplicates": {},
            "missing_annotations": 0,
            "heatmaps": {},
        }

        if df is None:
            return stats

        splits = self.get_splits()
        if splits is not None and view and view in splits:
            df = df.filter(pl.col("uuid").is_in(splits[view]))  # type: ignore

        stats["duplicates"] = get_duplicates_info(df)

        stats["class_distributions"] = get_class_distributions(df)

        stats["missing_annotations"] = get_missing_annotations(df)

        stats["heatmaps"] = get_heatmaps(df, sample_size)

        return stats

    def remove_duplicates(self) -> None:
        """Removes duplicate files and annotations from the dataset."""
        df = self._load_df_offline(lazy=True)
        if df is None:
            raise ValueError(
                "Dataset index or dataframe with annotations is not available."
            )
        duplicate_info = get_duplicates_info(df)

        duplicate_files_to_remove = [
            file
            for duplicates in duplicate_info["duplicate_uuids"]
            for file in duplicates["files"][1:]
        ]
        df = df.filter(~pl.col("file").is_in(duplicate_files_to_remove))

        df = df.unique(subset=["file", "annotation"], maintain_order=True)

        self._save_df_offline(df.collect())

        if self.is_remote:
            self.fs.put_dir(
                local_paths=self.local_path / "annotations",
                remote_dir="annotations",
                copy_contents=True,
            )
