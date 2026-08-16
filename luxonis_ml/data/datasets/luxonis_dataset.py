import json
import math
import shutil
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import cached_property
from os import PathLike
from pathlib import Path, PurePosixPath
from types import NotImplementedType
from typing import Any, Literal, overload

import numpy as np
import polars as pl
import rich.progress
from filelock import FileLock
from loguru import logger
from rich.progress import Progress
from semver.version import Version
from typing_extensions import Self, override

from luxonis_ml.data.exporters import (
    BaseExporter,
    ClassificationDirectoryExporter,
    CocoExporter,
    CreateMLExporter,
    DarknetExporter,
    FiftyOneClassificationExporter,
    NativeExporter,
    PreparedLDF,
    SegmentationMaskDirectoryExporter,
    TensorflowCSVExporter,
    UltralyticsNDJSONExporter,
    VOCExporter,
    YoloV4Exporter,
    YoloV6Exporter,
    YoloV8Exporter,
    YoloV8InstanceSegmentationExporter,
    YoloV8KeypointsExporter,
)
from luxonis_ml.data.exporters.exporter_utils import (
    ExporterSpec,
    create_zip_output,
)
from luxonis_ml.data.exporters.ldf_downgrade import resolve_export_version
from luxonis_ml.data.utils import (
    BucketStorage,
    BucketType,
    COCOFormat,
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
from luxonis_ml.data.utils.ldf_equivalence import ldf_equivalent
from luxonis_ml.data.utils.parquet import DEFAULT_METADATA
from luxonis_ml.enums.enums import DatasetType
from luxonis_ml.ldf import (
    Category,
    DatasetRecord,
    Detection,
    KeypointMetadata,
)
from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisFileSystem, deprecated, environ

from .base_dataset import BaseDataset, DatasetIterator, KeypointPair
from .metadata import Metadata
from .migration import migrate_dataframe, migrate_metadata
from .source import LuxonisComponent, LuxonisSource
from .utils import (
    find_filepath_group_id,
    find_filepath_uuid,
    get_dir,
    get_file,
)


class LuxonisDataset(BaseDataset):  # noqa: PLW1641
    """Luxonis Dataset Format (LDF) dataset handle.

    LDF is a flexible and feature-rich dataset format
    designed for use within the Luxonis MLOps ecosystem.

    Attributes:
        dataset_name: Name of the dataset.
        bucket_storage: Underlying storage backend for the dataset.
        bucket_type: Whether the dataset uses internal or external buckets.
        team_id: Optional cloud team identifier.
        version: The version of the underlying LDF that the dataset adheres to.

    """

    def __init__(
        self,
        dataset_name: str,
        team_id: str | None = None,
        bucket_type: BucketType
        | Literal["internal", "external"] = BucketType.INTERNAL,
        bucket_storage: (
            BucketStorage | Literal["local", "gcs", "s3", "azure"]
        ) = BucketStorage.LOCAL,
        *,
        delete_local: bool = False,
        delete_remote: bool = False,
    ) -> None:
        """Create a Luxonis Dataset Format dataset handle.

        Args:
            dataset_name: Dataset name.
            team_id: Optional cloud team identifier.
            bucket_type: Whether the dataset uses internal or external
                buckets.
            bucket_storage: Underlying storage backend.
            delete_local: Whether to delete a local dataset with the same
                name before initialization.
            delete_remote: Whether to delete the remote dataset as well.

        Raises:
            ValueError: If the dataset exists and deletion flags are not set.
            ValueError: If the dataset is remote but no bucket is configured.
            NotImplementedError: If Azure Blob Storage is selected as the
                bucket storage.

        """

        self._dataset_name = dataset_name
        self._base_path = environ.LUXONISML_BASE_PATH
        self._base_path.mkdir(exist_ok=True)

        self._credentials = self._init_credentials()
        self._is_synced = False

        # What is this for?
        self._bucket_type = BucketType(bucket_type)

        self._bucket_storage = BucketStorage(bucket_storage)

        if self._bucket_storage == BucketStorage.AZURE_BLOB:
            raise NotImplementedError("Azure Blob Storage not yet supported")

        self._bucket = self._get_credential("LUXONISML_BUCKET")

        if self.is_remote and self._bucket is None:
            raise ValueError(
                "The `LUXONISML_BUCKET` environment variable "
                "must be set for remote datasets"
            )

        self._team_id = team_id or self._get_credential("LUXONISML_TEAM_ID")

        self._init_paths()

        self._fs = LuxonisFileSystem(self._path)

        if delete_local or delete_remote:
            if self.exists(
                self._dataset_name,
                self._team_id,
                self._bucket_storage,
                self._bucket,
            ):
                self.delete_dataset(
                    delete_remote=delete_remote, delete_local=delete_local
                )

            self._init_paths()

        # For DDP GCS training - multiple processes
        with FileLock(self._base_path / ".metadata.lock"):
            self._metadata = self._get_metadata()

        if self.version.major != LDF_VERSION.major:
            logger.warning(
                f"LDF versions do not match. The current `luxonis-ml` "
                f"installation supports LDF v{LDF_VERSION}, but the "
                f"`{self.identifier}` dataset is in v{self._metadata.ldf_version}. "
                "Internal migration will be performed. Note that some parts "
                "and new features might not work correctly unless you "
                "manually re-create the dataset using the latest version "
                "of `luxonis-ml`."
            )

    @cached_property
    def _progress(self) -> Progress:
        return Progress(
            rich.progress.TextColumn(
                "[progress.description]{task.description}"
            ),
            rich.progress.BarColumn(),
            rich.progress.TaskProgressColumn(),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TimeRemainingColumn(),
        )

    @property
    def metadata(self) -> Metadata:
        """Get the dataset metadata.

        Returns:
            Deep copy of the dataset metadata.

        """
        return self._metadata.model_copy(deep=True)

    @cached_property
    def version(self) -> Version:
        """The version of the underlying LDF that the dataset adheres
        to.
        """
        return Version.parse(
            self._metadata.ldf_version, optional_minor_and_patch=True
        )

    @property
    def source(self) -> LuxonisSource:
        """Get the source information for the dataset.

        Returns:
            Dataset source metadata.

        Raises:
            ValueError: If source metadata is missing.

        """
        if self._metadata.source is None:
            raise ValueError("Source not found in metadata")
        return self._metadata.source

    @property
    @override
    def identifier(self) -> str:
        return self._dataset_name

    def __eq__(self, other: object) -> bool | NotImplementedType:
        """Compare datasets for equivalence."""
        if not isinstance(other, (LuxonisDataset, str)):
            return NotImplemented
        return ldf_equivalent(self, other)

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        if self.is_remote:
            return len(list(self._fs.walk_dir("media")))

        df = self._load_df_offline()
        return len(df.select("uuid").unique()) if df is not None else 0

    def _get_credential(self, key: str) -> str:
        """Get secret credentials from the credentials file or
        environment.
        """
        if key in self._credentials:
            return self._credentials[key]
        if not hasattr(environ, key):
            raise RuntimeError(f"'{key}' must be set in ENV variables")
        return getattr(environ, key)

    def _init_paths(self) -> None:
        """Configure local paths or a bucket directory."""
        self._local_path = (
            self._base_path
            / "data"
            / self._team_id
            / "datasets"
            / self._dataset_name
        )
        self._media_path = self._local_path / "media"
        self._annotations_path = self._local_path / "annotations"
        self._metadata_path = self._local_path / "metadata"
        self._arrays_path = self._local_path / "arrays"

        for path in [
            self._media_path,
            self._annotations_path,
            self._metadata_path,
        ]:
            path.mkdir(exist_ok=True, parents=True)

        if not self.is_remote:
            self._path = str(self._local_path)
        else:
            self._path = self._construct_url(
                self._bucket_storage,
                self._bucket,
                self._team_id,
                self._dataset_name,
            )

    def _save_df_offline(self, pl_df: pl.DataFrame) -> None:
        """Save annotations DataFrame into parquet files.

        Uses ``ParquetFileManager`` to preserve the same structure as the
        original dataset.

        Args:
            pl_df: DataFrame to save.

        Raises:
            ValueError: If any row in the DataFrame is missing a 'uuid' value.

        """
        annotations_path = Path(self._annotations_path)

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
        """Merge relevant metadata from ``other`` into ``self``."""
        self._metadata = self._metadata.merge_with(other._metadata)
        self._write_metadata()

    def clone(
        self,
        new_dataset_name: str,
        push_to_cloud: bool = True,
        splits_to_clone: list[str] | None = None,
        team_id: str | None = None,
    ) -> "LuxonisDataset":
        """Create a local copy of the current dataset.

        Warning:
            The cloned dataset overwrites any existing dataset with the same
            name.

        Args:
            new_dataset_name: Name of the cloned dataset.
            push_to_cloud: Whether to push the cloned dataset to the cloud
                when the current dataset is remote.
            splits_to_clone: Optional split names to clone. If omitted, all
                data is cloned.
            team_id: Optional team identifier for the cloned dataset.

        Returns:
            Cloned dataset handle.

        Raises:
            FileNotFoundError: If the current dataset is empty and
                ``splits_to_clone`` is specified.

        """
        if team_id is None:
            team_id = self._team_id

        new_dataset = LuxonisDataset(
            dataset_name=new_dataset_name,
            team_id=team_id,
            bucket_type=self._bucket_type,
            bucket_storage=self._bucket_storage,
            delete_local=True,
            delete_remote=True,
        )

        if self.is_remote:
            self.pull_from_cloud(update_mode=UpdateMode.MISSING)

        new_dataset_path = Path(new_dataset._local_path)
        new_dataset_path.mkdir(parents=True, exist_ok=True)

        if splits_to_clone is not None:
            df_self = self._load_df_offline(raise_when_empty=True)
            splits_self = self._load_splits(self._metadata_path)
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
            self._local_path,
            new_dataset._local_path,
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

        new_dataset._metadata.parent_dataset = self._dataset_name

        if push_to_cloud:
            if self.is_remote:
                new_dataset.push_to_cloud(
                    update_mode=UpdateMode.MISSING,
                    bucket_storage=self._bucket_storage,
                )
            else:
                logger.warning(
                    f"Cannot push to cloud. The cloned dataset '{new_dataset._dataset_name}' is local. "
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
        """Merge another dataset into this or a new dataset.

        Args:
            other: Dataset to merge into this dataset.
            inplace: Whether to merge into this dataset. If ``False``, a
                new dataset is created.
            new_dataset_name: Name of the new dataset when ``inplace`` is
                ``False``.
            splits_to_merge: Optional split names to merge.
            team_id: Optional team identifier for a newly created dataset.

        Returns:
            Dataset containing the merged data.

        Raises:
            ValueError: If the datasets have different bucket storage types.
            ValueError: If ``inplace`` is ``False`` but no name for the new
                dataset is provided.

        """
        if inplace:
            target_dataset = self
        elif new_dataset_name:
            if self._bucket_storage != other._bucket_storage:
                raise ValueError(
                    "Cannot merge datasets with different bucket storage types."
                )
            target_dataset = self.clone(
                new_dataset_name, push_to_cloud=False, team_id=team_id
            )
        else:
            raise ValueError(
                "You must specify a name for the new dataset "
                "when `inplace` is `False`"
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
                f"Found {len(duplicate_group_ids)} duplicate group IDs in the datasets. "
                "Merging will remove these duplicates from the incoming dataset."
            )
            df_other = df_other.filter(
                ~df_other["group_id"].is_in(duplicate_group_ids)
            )

        splits_self = self._load_splits(self._metadata_path)
        splits_other = self._load_splits(other._metadata_path)
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
                other._media_path,
                target_dataset._media_path,
                dirs_exist_ok=True,
                ignore=lambda d, n: self._ignore_files_not_in_uuid_set(
                    d, n, uuids_to_merge if splits_to_merge else set()
                ),
            )
            target_dataset.push_to_cloud(
                bucket_storage=target_dataset._bucket_storage,
                update_mode=UpdateMode.MISSING,
            )

        for entry in (
            df_other.select(["uuid", "file"])
            .unique(subset=["uuid"])
            .to_dicts()
        ):
            uid, rel_file = entry["uuid"], entry["file"]
            src_path = other._media_path / f"{uid}{Path(rel_file).suffix}"
            dst_path = target_dataset._media_path / src_path.name
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
        splits_path_self = self._metadata_path / "splits.json"
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
        """Load the dataset DataFrame from local storage.

        Args:
            lazy: Whether to return a LazyFrame that can be further processed
                before collecting.
            raise_when_empty: Whether to raise an error if the dataset is
                empty. If ``False``, returns ``None`` when the dataset is
                empty.
            attempt_migration: Whether to attempt internal migration of the
                DataFrame if the LDF version of the dataset does not match
                the supported LDF version of the current ``luxonis-ml``
                installation. If ``True``, will perform internal migration
                using `migrate_dataframe`. If ``False``, will return the
                DataFrame as-is without migration, which may lead to errors
                if the LDF versions do not match.

        Returns:
            The dataset annotations as a Polars DataFrame or LazyFrame, or
            ``None`` if the dataset is empty and ``raise_when_empty`` is
            ``False``.

        Raises:
            FileNotFoundError: If the dataset is empty and ``raise_when_empty``
                is ``True``.

        """
        path = (
            self._base_path
            / "data"
            / self._team_id
            / "datasets"
            / self._dataset_name
            / "annotations"
        )
        files = list(path.glob("*.parquet"))
        if not files:
            if raise_when_empty:
                raise FileNotFoundError(
                    f"Dataset '{self._dataset_name}' is empty."
                )
            return None

        lazy_df = self._scan_annotation_files(files)

        if lazy:
            return lazy_df

        df = lazy_df.collect()
        if df.is_empty() and raise_when_empty:
            raise FileNotFoundError(
                f"Dataset '{self._dataset_name}' is empty."
            )

        if attempt_migration and self.version.major != LDF_VERSION.major:
            df = migrate_dataframe(df)

        return df

    @staticmethod
    def _scan_annotation_files(files: list[Path]) -> pl.LazyFrame:
        lazy_frames = []
        for file in files:
            lazy_frame = pl.scan_parquet(str(file))
            if "sample_metadata" not in lazy_frame.schema:
                lazy_frame = lazy_frame.with_columns(
                    pl.lit(DEFAULT_METADATA).alias("sample_metadata")
                )
            lazy_frames.append(lazy_frame)

        if len(lazy_frames) == 1:
            return lazy_frames[0]
        return pl.concat(lazy_frames, how="diagonal_relaxed")

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

    @overload
    def _get_index(
        self,
        lazy: Literal[True] = ...,
        raise_when_empty: Literal[True] = ...,
    ) -> pl.LazyFrame: ...

    def _get_index(
        self,
        lazy: bool = False,
        raise_when_empty: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame | None:
        """Load unique file entries from annotation data.

        Args:
            lazy: Whether to return a LazyFrame that can be further processed
                before collecting.
            raise_when_empty: Whether to raise an error if the dataset is
                empty. If ``False``, returns ``None`` when the dataset is empty.

        Returns:
            DataFrame or LazyFrame containing the dataset records.

        Raises:
            FileNotFoundError: If the dataset is empty and ``raise_when_empty``
                is ``True``.

        """
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
                    pl.col("file").replace(mapping).alias("original_filepath"),
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
        # `keypoint_metadata` is new in LDF 2.2 and `Metadata` forbids
        # extra fields. A dataset without keypoints must not carry the
        # key, or an older luxonis-ml refuses to open it.
        exclude: set[str] = set()
        if not self._metadata.keypoint_metadata:
            exclude.add("keypoint_metadata")
        path = self._metadata_path / "metadata.json"
        path.write_text(
            self._metadata.model_dump_json(indent=4, exclude=exclude)
        )
        with suppress(shutil.SameFileError):
            self._fs.put_file(path, "metadata/metadata.json")

    @staticmethod
    def _construct_url(
        bucket_storage: BucketStorage,
        bucket: str,
        team_id: str,
        dataset_name: str,
    ) -> str:
        return f"{bucket_storage.value}://{bucket}/{team_id}/datasets/{dataset_name}"

    # TODO: Is the cache used anywhere at all?
    def _init_credentials(self) -> dict[str, Any]:
        credentials_cache_file = self._base_path / "credentials.json"
        if credentials_cache_file.exists():
            return json.loads(credentials_cache_file.read_text())
        return {}

    def _get_metadata(self) -> Metadata:
        """Load metadata from local storage or cloud.

        Cloud metadata is always downloaded before loading.
        """
        if self._fs.exists("metadata/metadata.json"):
            path = get_file(
                self._fs,
                "metadata/metadata.json",
                self._metadata_path,
                default=self._metadata_path / "metadata.json",
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
            if version.major != LDF_VERSION.major:  # pragma: no cover
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
            keypoint_metadata={},
            categorical_encodings={},
            metadata_types={},
        )

    @property
    def is_remote(self) -> bool:
        """Whether the dataset is stored remotely (in a cloud bucket) or
        locally.
        """
        return self._bucket_storage != BucketStorage.LOCAL

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
        rewrite_metadata: bool = True,
    ) -> None:
        tasks = self.get_task_names() if task is None else [task]

        for t in tasks:
            self._metadata.set_classes(classes, t)

        if rewrite_metadata:
            self._write_metadata()

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        return self._metadata.classes

    @override
    def set_keypoint_metadata(
        self,
        labels: list[str] | None = None,
        edges: list[KeypointPair] | None = None,
        task: str | None = None,
        *,
        flip_pairs: list[KeypointPair] | None = None,
        sigmas: list[float] | None = None,
        infer_flip_pairs: bool = True,
    ) -> None:
        updates = {
            "labels": labels,
            "edges": edges,
            "flip_pairs": flip_pairs,
            "sigmas": sigmas,
        }
        if all(value is None for value in updates.values()):
            raise ValueError(
                "Must provide either keypoint names, edges, flip pairs, "
                "or sigmas"
            )

        tasks = self.get_task_names() if task is None else [task]
        for t in tasks:
            current = self._metadata.keypoint_metadata.get(t)
            if current is not None and self._renames_keypoints(
                current, labels
            ):
                current = None
            keypoint_metadata = KeypointMetadata.model_validate(
                {
                    **(current.model_dump() if current else {}),
                    **{
                        field: value
                        for field, value in updates.items()
                        if value is not None
                    },
                }
            )
            self._metadata.keypoint_metadata[t] = self._fill_in_flip_pairs(
                keypoint_metadata, infer=infer_flip_pairs
            )

        self._write_metadata()

    @staticmethod
    def _fill_in_flip_pairs(
        keypoint_metadata: KeypointMetadata, *, infer: bool
    ) -> KeypointMetadata:
        """Infer flip pairs from the keypoint names when none are known.

        Only a write path infers them, never a read path. A read path
        would give flip pairs to a dataset that never asked for them.
        """
        labels = keypoint_metadata.labels
        if not infer or keypoint_metadata.flip_pairs or not labels:
            return keypoint_metadata
        flip_pairs = KeypointMetadata.infer_flip_pairs(labels)
        if not flip_pairs:
            return keypoint_metadata
        return keypoint_metadata.model_copy(update={"flip_pairs": flip_pairs})

    @staticmethod
    def _renames_keypoints(
        current: KeypointMetadata, labels: list[str] | None
    ) -> bool:
        """Whether new labels give the stored indices a new meaning.

        `edges`, `flip_pairs` and `sigmas` all address a keypoint by its
        position. New names put a different keypoint at a position, so
        the stored values describe the wrong keypoints. Placeholder names
        carry no identity. A rename that keeps their count therefore only
        names the keypoints that are already there.
        """
        if labels is None or not current.labels or current.labels == labels:
            return False
        return current.has_names or len(current.labels) != len(labels)

    @override
    def get_keypoint_metadata(self) -> dict[str, KeypointMetadata]:
        return dict(self._metadata.keypoint_metadata)

    def _update_keypoint_metadata(
        self,
        num_kpts_per_task: dict[str, set[int]],
        declared: dict[str, KeypointMetadata],
    ) -> None:
        """Promote the keypoints that the annotations describe.

        Only a keypoint task with no keypoint metadata gets placeholder
        names and chain edges. An explicit definition thus survives an
        `add`.
        """
        if not num_kpts_per_task:
            return

        for task, sizes in num_kpts_per_task.items():
            n_keypoints = max(sizes)
            if len(sizes) > 1:
                logger.warning(
                    f"Task '{task}' mixes annotations with different numbers "
                    f"of keypoints ({sorted(sizes)}). Storing keypoint "
                    f"metadata for {n_keypoints} keypoints."
                )
            stored = self._metadata.keypoint_metadata.get(task)
            task_keypoint_metadata = declared.get(task)
            if task_keypoint_metadata is not None:
                task_keypoint_metadata.validate_for(
                    n_keypoints, f"task '{task}'"
                )
                task_keypoint_metadata = self._merge_into_stored(
                    task_keypoint_metadata, stored, task, n_keypoints
                )
            elif stored is not None and not self._placeholder_is_outgrown(
                stored, n_keypoints
            ):
                continue
            else:
                task_keypoint_metadata = self._placeholder_keypoint_metadata(
                    n_keypoints
                )
            labels = task_keypoint_metadata.labels
            if not labels or (
                not task_keypoint_metadata.has_names
                and len(labels) < n_keypoints
            ):
                # A description of edges alone names nothing, and a
                # placeholder from an earlier `add` can be too narrow.
                # The stored entry has to carry the keypoint count.
                placeholder = self._placeholder_keypoint_metadata(n_keypoints)
                task_keypoint_metadata = task_keypoint_metadata.model_copy(
                    update={"labels": placeholder.labels}
                )
            self._metadata.keypoint_metadata[task] = self._fill_in_flip_pairs(
                task_keypoint_metadata, infer=True
            )
        self._write_metadata()

    @classmethod
    def _merge_into_stored(
        cls,
        declared: KeypointMetadata,
        stored: KeypointMetadata | None,
        task: str,
        n_keypoints: int,
    ) -> KeypointMetadata:
        """Combine described keypoints with the ones already stored.

        A described value wins. A field that the description leaves empty
        comes from the stored keypoint metadata. An `add` thus discards
        nothing. Stored labels are the exception. They name the columns of
        the rows already written, so they keep their order.
        """
        if stored is None:
            return declared

        # No warning for a stored value that `add` generated itself. The
        # annotations now describe a real one.
        placeholder = cls._placeholder_keypoint_metadata(n_keypoints)
        merged = declared.model_copy(deep=True)
        if stored.has_names:
            # `_alignment_keypoint_metadata` writes the new rows in the
            # stored order, so the stored labels name the stored columns.
            # A record may name a subset of them, or name them in another
            # order. Neither renames a column.
            merged.labels = list(stored.labels)
        for field in KeypointMetadata.model_fields:
            new, old = getattr(merged, field), getattr(stored, field)
            if not new:
                setattr(merged, field, old)
            elif old and old != new and old != getattr(placeholder, field):
                logger.warning(
                    f"The annotations of task '{task}' describe a different "
                    f"`{field}` than the one already stored. Using the "
                    f"described one. Stored: {old}, described: {new}."
                )
        return merged

    @staticmethod
    def _placeholder_keypoint_metadata(n_keypoints: int) -> KeypointMetadata:
        """Build what `add` writes for keypoints that name none."""
        return KeypointMetadata(
            labels=[str(i) for i in range(n_keypoints)],
            edges=[(i, i + 1) for i in range(n_keypoints - 1)],
        )

    @classmethod
    def _placeholder_is_outgrown(
        cls, stored: KeypointMetadata, n_keypoints: int
    ) -> bool:
        """Whether `add` may rewrite an entry it generated itself.

        A placeholder records only how many keypoints there are, so a
        later `add` with more of them replaces it. A definition survives,
        and so does a wider placeholder: the stored count sizes the empty
        keypoint arrays, so it has to cover the widest row on disk.
        """
        placeholder = cls._placeholder_keypoint_metadata(len(stored.labels))
        return stored == placeholder and n_keypoints > len(stored.labels)

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
        """Get the categorical encodings for the dataset grouped by
        task.

        Example output:

        .. python::

            {
                "vehicles": {
                    "color": {"red": 0, "green": 1, "blue": 2},
                    "brand": {"audi": 0, "bmw": 1, "mercedes": 2},
                }
            }
        """
        return self._metadata.categorical_encodings

    def get_metadata_types(
        self,
    ) -> dict[str, Literal["float", "int", "str", "Category"]]:
        """Get the metadata types for each metadata annotation in the
        dataset.

        Example output:

        .. python::

            {
                "id": "int",
                "time_of_day": "Category",
                "temperature": "float",
            }
        """
        return self._metadata.metadata_types

    def pull_from_cloud(
        self, update_mode: UpdateMode = UpdateMode.MISSING
    ) -> None:
        """Synchronize the dataset from a remote bucket to a local
        storage.

        Annotations and metadata are always pulled. Media files are pulled
        either when missing locally or always, depending on ``update_mode``.

        Args:
            update_mode: Media synchronization mode.

        """
        if not self.is_remote:
            logger.warning("This is a local dataset! Cannot sync from cloud.")
            return

        local_dir = self._base_path / "data" / self._team_id / "datasets"
        local_dir.mkdir(exist_ok=True, parents=True)

        lock_path = local_dir / ".sync.lock"

        with FileLock(str(lock_path)):  # DDP GCS training - multiple processes
            logger.info(
                "Pulling remote's dataset annotations and metadata to local dataset ..."
            )
            for dir_name in ["annotations", "metadata"]:
                _ = get_dir(self._fs, dir_name, self._local_path)

            index = self._get_index(lazy=False)
            missing_media_paths = []
            if index is not None:
                uuids = index["uuid"].to_list()
                origps = index["original_filepath"].to_list()

                media_root = Path(local_dir) / self._dataset_name / "media"

                missing_media_paths = [
                    f"media/{uid}{Path(orig).suffix}"
                    for uid, orig in zip(uuids, origps, strict=True)
                    if not Path(orig).exists()
                    and not (media_root / f"{uid}{Path(orig).suffix}").exists()
                ]

            if update_mode == UpdateMode.ALL:
                logger.info("Force-pulling all media files...")
                self._fs.get_dir(remote_paths="", local_dir=local_dir)
            elif update_mode == UpdateMode.MISSING and missing_media_paths:
                logger.info(
                    f"Pulling {len(missing_media_paths)} missing files..."
                )
                self._fs.get_dir(
                    remote_paths=missing_media_paths,
                    local_dir=local_dir / f"{self._dataset_name}" / "media",
                )
            else:
                logger.info("Media already synced")

    def push_to_cloud(
        self,
        bucket_storage: BucketStorage | None,
        update_mode: UpdateMode = UpdateMode.MISSING,
    ) -> None:
        """Push the local dataset to a remote bucket.

        Annotations and metadata are always pushed. Media files are pushed
        either when missing remotely or always, depending on ``update_mode``.

        Args:
            bucket_storage: Remote storage backend to push to.
                If unset, uses the dataset's current bucket storage type.
            update_mode: Media synchronization mode.

        Raises:
            ValueError: If the dataset is empty or not initialized.
            FileNotFoundError: If any media files are missing locally when attempting
                to push.

        """
        index = self._get_index(lazy=False)

        if index is None:
            raise ValueError(
                "Cannot push to cloud. The dataset is empty or not initialized."
            )

        dataset = LuxonisDataset(
            dataset_name=self._dataset_name,
            team_id=self._team_id,
            bucket_type=self._bucket_type,
            bucket_storage=bucket_storage or self._bucket_storage,
            delete_local=False,
            delete_remote=False,
        )

        bucket_uuids = (
            [
                PurePosixPath(path).stem
                for path in dataset._fs.walk_dir(
                    "media", recursive=False, typ="file"
                )
            ]
            if dataset._fs.exists("media")
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
                fallback_path = self._local_path / "media" / f"{uuid}{suffix}"
                if fallback_path.exists():
                    missing_uuid_dict[str(fallback_path)] = uuid
                else:
                    raise FileNotFoundError(
                        f"File {original_path} and {fallback_path} do not exist!"
                    )
            else:
                missing_uuid_dict[original_path] = uuid

        for dir_name in ["annotations", "metadata"]:
            dataset._fs.put_dir(
                local_paths=self._local_path / dir_name,
                remote_dir=dir_name,
                copy_contents=True,
            )

        if update_mode == UpdateMode.ALL:
            logger.info("Force-pushing all media files...")
            dataset._fs.put_dir(
                local_paths=self._local_path / "media", remote_dir="media"
            )
        elif update_mode == UpdateMode.MISSING and missing_uuid_dict:
            logger.info(
                f"Pushing {len(missing_uuid_dict)} missing files to cloud..."
            )
            dataset._fs.put_dir(
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
        """Delete the dataset from local storage and optionally the
        cloud.

        Args:
            delete_remote: Whether to delete the remote dataset.
            delete_local: Whether to delete the local dataset files.

        Raises:
            ValueError: If neither ``delete_remote`` nor ``delete_local`` is set to ``True``.

        """
        if not (delete_remote or delete_local):
            raise ValueError(
                "Must set delete_remote=True and/or delete_local=True when calling delete_dataset()"
            )

        if not self.is_remote and delete_local:
            logger.info(
                f"Deleting local dataset '{self._dataset_name}' from local storage"
            )
            shutil.rmtree(self._path)

        if self.is_remote and delete_remote:
            logger.info(
                f"Deleting remote dataset '{self._dataset_name}' from cloud storage"
            )
            assert self._path
            assert self._dataset_name
            assert self._local_path
            self._fs.delete_dir(allow_delete_parent=True)

        if self.is_remote and delete_local:
            logger.info(
                f"Deleting remote dataset '{self._dataset_name}' from local storage"
            )
            if self._local_path.exists():
                shutil.rmtree(self._local_path)

    def _process_arrays(self, data_batch: list[DatasetRecord]) -> None:
        logger.info("Checking arrays...")
        task = self._progress.add_task(
            "[magenta]Processing arrays...", total=len(data_batch)
        )
        self._progress.start()
        uuid_dict = {}
        for record in data_batch:
            self._progress.update(task, advance=1)
            for detections in record.annotation.values():
                for detection in _walk_detections(detections):
                    ann = detection.array
                    if ann is None:
                        continue
                    if isinstance(ann.path, np.ndarray):
                        raise NotImplementedError(
                            "An array annotation holds its data in memory, "
                            "which a dataset cannot store. Save it as a "
                            "'.npy' file and pass that path instead."
                        )
                    if self.is_remote:
                        uuid = self._fs.get_file_uuid(ann.path, local=True)
                        uuid_dict[str(ann.path)] = uuid
                        ann.path = Path(uuid).with_suffix(ann.path.suffix)
                    else:
                        ann.path = ann.path.absolute().resolve()
        self._progress.stop()
        self._progress.remove_task(task)
        if self.is_remote:
            logger.info("Uploading arrays...")
            self._fs.put_dir(
                local_paths=uuid_dict.keys(),
                remote_dir="arrays",
                uuid_dict=uuid_dict,
            )

    def _alignment_keypoint_metadata(
        self, declared: dict[str, KeypointMetadata]
    ) -> dict[str, KeypointMetadata]:
        """Return the keypoint metadata the payload is aligned against.

        The stored keypoint metadata owns the label set of a task, so a
        record can name a subset of it. Placeholder names that an earlier
        `add` generated own nothing, so the records win over those.
        """
        aligned = dict(declared)
        for task, stored in self._metadata.keypoint_metadata.items():
            if stored.has_names:
                aligned[task] = stored
        return aligned

    def _add_process_batch(
        self,
        data_batch: list[DatasetRecord],
        pfm: ParquetFileManager,
        index: pl.DataFrame | None,
        declared_keypoint_metadata: dict[str, KeypointMetadata],
    ) -> None:
        keypoint_metadata = self._alignment_keypoint_metadata(
            declared_keypoint_metadata
        )
        paths = {
            path for data in data_batch for path in data.file_paths.values()
        }
        logger.info("Generating UUIDs...")
        uuid_dict = self._fs.get_file_uuids(paths, local=True)

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

            self._fs.put_dir(
                local_paths=paths,
                remote_dir="media",
                uuid_dict=dict(uuid_dict),
            )
            logger.info("Media uploaded")

        self._process_arrays(data_batch)

        task = self._progress.add_task(
            "[magenta]Processing data...", total=len(data_batch)
        )

        logger.info("Saving annotations...")
        with self._progress:
            for record in data_batch:
                file_paths = record.file_paths.values()
                uuid_list = [
                    uuid_dict[str(file_path)] for file_path in file_paths
                ]
                group_id = (
                    str(merge_uuids(uuid_list))
                    if len(uuid_list) > 1
                    else str(uuid_list[0])
                )
                for row in record.to_parquet_rows(keypoint_metadata):
                    pfm.write(uuid_dict[row["file"]], row, group_id)
                self._progress.update(task, advance=1)
        self._progress.remove_task(task)

    def _infer_default_task(self, record: DatasetRecord) -> None:
        """Re-file the record's unnamed detections under an inferred task.

        A detection whose class belongs to a known task moves there. One
        whose class is unknown stays under the default task.
        """
        detections = record.annotation.get("")
        if not detections:
            return

        current_classes = self.get_classes()
        inferred: dict[str, list[Detection]] = defaultdict(list)
        for detection in detections:
            task_name = (
                infer_task("", detection.class_name, current_classes)
                if detection.class_name is not None
                else ""
            )
            inferred[task_name].append(detection)

        del record.annotation[""]
        for task_name, task_detections in inferred.items():
            record.annotation.setdefault(task_name, []).extend(task_detections)

    @override
    def add(
        self, generator: DatasetIterator, batch_size: int = 1_000_000
    ) -> Self:
        """Add data to the dataset from a generator of records.

        Args:
            generator: The generator should yield either
                dictionaries that can be converted to
                ``DatasetRecord`` objects or actual ``DatasetRecord``
                instances. Each record must contain at least a
                file path and can optionally include an annotation
                and a task name. Use ``sample_metadata`` for values
                attached to the whole sample rather than to one
                annotation.

                For example:

                .. python::

                    def record_generator():
                        yield {
                            "file": "/path/to/image.jpg",

                            "sample_metadata": {
                                "record_id": 123,
                                "camera": "left",
                                "tags": ["night", "warehouse"],
                            },

                            "annotation": {
                                "animals": [
                                    {
                                        "instance_id": 1,
                                        "class": "cat",
                                        "boundingbox": {
                                            "x": 0.10,
                                            "y": 0.20,
                                            "w": 0.30,
                                            "h": 0.40,
                                        },
                                        "keypoints": {
                                            "keypoints": [
                                                (0.15, 0.25, 1),
                                                (0.50, 0.60, 1),
                                                (0.70, 0.80, 0),
                                            ],
                                        },
                                        "instance_segmentation": {
                                            "mask": "/path/to/mask.png",
                                        },
                                    }
                                ],
                            },
                        }

                `LuxonisLoader` returns ``sample_metadata`` through
                `LoaderOutput.metadata`. Annotation metadata under
                ``annotation["metadata"]`` is different: it becomes a
                metadata label task.

            batch_size: The number of records to process in a batch before writing
                to storage. Larger batch sizes may be more efficient but will
                use more memory.

        Raises:
            ValueError: If the records yielded by the generator are not in the expected format.
            ValueError: If the dataset contains metadata annotations with conflicting types.

        """
        logger.info(f"Adding data to dataset '{self._dataset_name}'...")

        data_batch: list[DatasetRecord] = []

        classes_per_task: dict[str, set[str]] = defaultdict(set)
        tasks: dict[str, set[str]] = defaultdict(set)
        categorical_encodings = defaultdict(dict)
        metadata_types = {}
        num_kpts_per_task: dict[str, set[int]] = defaultdict(set)
        declared_keypoint_metadata: dict[str, KeypointMetadata] = {}
        sources: set[str] = set()

        annotations_path = get_dir(
            self._fs,
            "annotations",
            self._local_path,
            default=self._annotations_path,
        )

        index = self._get_index()

        assert annotations_path is not None

        def update_state(task_name: str, ann: Detection) -> None:
            if ann.class_name is not None:
                classes_per_task[task_name].add(ann.class_name)
            elif not classes_per_task[task_name]:
                classes_per_task[task_name] = set()

            tasks[task_name] |= ann.get_task_types()

            if ann.keypoints is not None:
                num_kpts_per_task[task_name].add(len(ann.keypoints.keypoints))
                declared = ann.keypoints.declared_metadata()
                if declared is not None:
                    current = declared_keypoint_metadata.get(task_name)
                    declared_keypoint_metadata[task_name] = (
                        declared
                        if current is None
                        else current.merge_with(
                            declared, f"task '{task_name}'"
                        )
                    )
            for name, value in ann.metadata.items():
                task = f"{task_name}/metadata/{name}"
                typ = type(value).__name__
                if task in metadata_types and metadata_types[task] != typ:
                    if {typ, metadata_types[task]} == {"int", "float"}:
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

        with ParquetFileManager(annotations_path, batch_size) as pfm:
            for i, record in enumerate(generator, start=1):
                if not isinstance(record, DatasetRecord):
                    record = DatasetRecord(**record)
                sources.update(record.files.keys())

                self._infer_default_task(record)
                for task_name, detections in record.annotation.items():
                    # A task with no detections is a negative that still
                    # declares the task, so it is registered either way.
                    tasks.setdefault(task_name, set())
                    for detection in detections:
                        update_state(task_name, detection)

                data_batch.append(record)
                if i % batch_size == 0:
                    self._add_process_batch(
                        data_batch, pfm, index, declared_keypoint_metadata
                    )
                    data_batch = []

            self._add_process_batch(
                data_batch, pfm, index, declared_keypoint_metadata
            )

        with suppress(shutil.SameFileError):
            self._fs.put_dir(annotations_path, "")

        curr_classes = self.get_classes()
        for task, classes in classes_per_task.items():
            old_classes = set(curr_classes.get(task, []))
            new_classes = list(classes - old_classes)
            if new_classes or task not in curr_classes:
                logger.info(
                    f"Detected new classes for task group '{task}': {new_classes}"
                )

                self.set_classes(list(classes | old_classes), task=task)

        self._update_keypoint_metadata(
            num_kpts_per_task, declared_keypoint_metadata
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
        """Get the dataset splits definitions.

        Returns:
            A mapping of split names to list of UUIDs,
            or ``None`` if no splits are defined.

        """
        splits_path = get_file(
            self._fs, "metadata/splits.json", self._metadata_path
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
        splits: (
            Mapping[str, Sequence[PathType]]
            | Mapping[str, float]
            | tuple[float, float, float]
            | None
        ) = None,
        *,
        ratios: dict[str, float] | tuple[float, float, float] | None = None,
        definitions: dict[str, list[PathType]] | None = None,
        replace_old_splits: bool = False,
    ) -> None:
        """Create dataset splits for training, validation, and testing.

        Note:
            Although ``"train"``, ``"val"``, and ``"test"``
            are the conventional split names, you can use any split names
            you want by providing a mapping to the ``splits`` argument.
            This can be useful for combining records from multiple
            sources (``"train_real"``, ``"train_synth"``) or for
            creating fully custom splits.

        Args:
            splits: A mapping defining the splits. Can be one of the following:

                - A mapping of split names to lists of file paths.
                - A mapping of split names to float ratios.
                - A tuple of three float ratios for train, val, and test splits.

            ratios: A mapping of split names to float ratios
                or a tuple of three float ratios for train, val, and test splits.

                .. deprecated:: 0.4.0
                    Use ``splits`` instead.

            definitions: A mapping of split names to lists of file paths.

                .. deprecated:: 0.4.0
                    Use ``splits`` instead.

            replace_old_splits: Whether to replace old splits with new ones. If ``False`
                (default), new splits will be added to old splits, and duplicate group IDs will be filtered out. If ``True``, old splits will be replaced with new splits.

        Raises:
            ValueError: If both ``ratios`` and ``definitions`` are provided.
            ValueError: If neither ``splits``, ``ratios``, nor ``definitions`` is provided.
            ValueError: If both ``splits`` and ``ratios``/``definitions`` are provided.
            ValueError: If ``splits`` is provided but is empty.
            ValueError: If ``ratios`` is provided but does not sum to 1.
            ValueError: If ``definitions`` is provided but the total number of files in definitions exceeds
                the dataset size.
            ValueError: If ``definitions`` are provided but all of them
                are already included in old splits, resulting in no new
                files to add to splits while ``replace_old_splits`` is ``False``.
            FileNotFoundError: If the dataset is empty.
            TypeError: If the splits definitions are not in the expected format.

        """
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
                logger.warning(
                    "Dataset size is smaller than the total number of files in the definitions. "
                    f"Dataset size: {len(self)}, Definitions: {n_files}. "
                    "Duplicate files will be filtered out and extra files in definitions will be ignored."
                )
                self.remove_duplicates()

        splits_to_update: list[str] = []
        new_splits: dict[str, list[str]] = {}
        old_splits: dict[str, list[str]] = defaultdict(list)

        splits_path = get_file(
            self._fs,
            "metadata/splits.json",
            self._metadata_path,
            default=self._metadata_path / "splits.json",
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
            index = self._get_index(raise_when_empty=True)
            for split, filepaths in definitions.items():
                splits_to_update.append(split)
                if not isinstance(filepaths, list):
                    raise TypeError(
                        "Must provide splits as a list of filepaths"
                    )
                ids: list[str] = []
                for filepath in filepaths:
                    group_id = find_filepath_group_id(
                        filepath, index, raise_on_missing=False
                    )

                    if group_id is None:
                        logger.warning(
                            f"No group ID found for '{filepath}' in definitions; skipping."
                        )
                        continue
                    ids.append(group_id)

                new_splits[split] = list(set(ids))

        for split, group_ids in new_splits.items():
            old_splits[split].extend(group_ids)

        splits_path.write_text(json.dumps(old_splits, indent=4))

        with suppress(shutil.SameFileError):
            self._fs.put_file(splits_path, "metadata/splits.json")

    @staticmethod
    @override
    def exists(
        dataset_name: str,
        team_id: str | None = None,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
        bucket: str | None = None,
    ) -> bool:
        """Check whether a dataset exists.

        Args:
            dataset_name: Dataset name to check.
            team_id: Optional team identifier.
            bucket_storage: Storage backend to inspect.
            bucket: Optional bucket name for remote storage.

        Returns:
            ``True`` if the dataset exists, ``False`` otherwise.

        Raises:
            ValueError: If bucket storage is remote but no bucket name is provided.

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
        """List available datasets.

        Args:
            team_id: Optional team identifier.
            bucket_storage: Storage backend to inspect.
            bucket: Optional bucket name for remote storage.

        Returns:
            List of dataset names.

        Raises:
            ValueError: If bucket storage is remote but no bucket name is provided.
            ValueError: If the dataset is stored remotely but no ``bucket``
                parameter is provided or no ``LUXONISML_BUCKET`` environment variable is set.

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
        ldf_version: str | None = None,
    ) -> Path | list[Path]:
        """Export the dataset into one of the supported formats.

        Args:
            output_path: Directory where the dataset should be exported.
            dataset_type: Export format.
            max_partition_size_gb: Optional maximum partition size. If the
                dataset exceeds this size, it is split into partitions named
                ``{dataset_name}_part{partition_number}``.
            zip_output: Whether to zip the exported dataset or each
                partition after export.
            ldf_version: LDF version to write, for example ``"2.0"``, so
                the export can be read by an older luxonis-ml. Native
                format only. Defaults to the version this installation
                writes. Downgrading is lossy and warns about what it drops.

        Returns:
            Export directory, or ZIP archive paths when ``zip_output`` is
            enabled.

        Raises:
            NotImplementedError: If the specified export format is not supported.
            ValueError: If the output path already exists, or if
                ``ldf_version`` is set for a non-native format or names a
                version this installation cannot write.

        """
        if ldf_version is not None and dataset_type != DatasetType.NATIVE:
            raise ValueError(
                f"'ldf_version' only applies to the native format, "
                f"not '{dataset_type}'."
            )
        target_version = resolve_export_version(ldf_version)
        keypoint_metadata = self.metadata.keypoint_metadata

        EXPORTER_MAP: dict[DatasetType, ExporterSpec] = {
            DatasetType.NATIVE: ExporterSpec(
                NativeExporter,
                {
                    "keypoint_metadata": keypoint_metadata,
                    "ldf_version": target_version,
                },
            ),
            DatasetType.COCO: ExporterSpec(
                CocoExporter,
                {
                    "format": COCOFormat.ROBOFLOW,
                    "keypoint_metadata": keypoint_metadata,
                },
            ),
            DatasetType.YOLOV8BOUNDINGBOX: ExporterSpec(YoloV8Exporter, {}),
            DatasetType.YOLOV8INSTANCESEGMENTATION: ExporterSpec(
                YoloV8InstanceSegmentationExporter, {}
            ),
            DatasetType.YOLOV8KEYPOINTS: ExporterSpec(
                YoloV8KeypointsExporter, {}
            ),
            DatasetType.YOLOV6: ExporterSpec(YoloV6Exporter, {}),
            DatasetType.YOLOV4: ExporterSpec(YoloV4Exporter, {}),
            DatasetType.DARKNET: ExporterSpec(DarknetExporter, {}),
            DatasetType.CLSDIR: ExporterSpec(
                ClassificationDirectoryExporter, {}
            ),
            DatasetType.FIFTYONECLS: ExporterSpec(
                FiftyOneClassificationExporter, {}
            ),
            DatasetType.SEGMASK: ExporterSpec(
                SegmentationMaskDirectoryExporter, {}
            ),
            DatasetType.VOC: ExporterSpec(VOCExporter, {}),
            DatasetType.CREATEML: ExporterSpec(CreateMLExporter, {}),
            DatasetType.TFCSV: ExporterSpec(TensorflowCSVExporter, {}),
            DatasetType.ULTRALYTICSNDJSON: ExporterSpec(
                UltralyticsNDJSONExporter,
                {
                    "dataset_type": DatasetType.ULTRALYTICSNDJSON,
                    "ndjson_task": "detect",
                },
            ),
            DatasetType.ULTRALYTICSNDJSONINSTANCESEGMENTATION: ExporterSpec(
                UltralyticsNDJSONExporter,
                {
                    "dataset_type": (
                        DatasetType.ULTRALYTICSNDJSONINSTANCESEGMENTATION
                    ),
                    "ndjson_task": "segment",
                },
            ),
            DatasetType.ULTRALYTICSNDJSONKEYPOINTS: ExporterSpec(
                UltralyticsNDJSONExporter,
                {
                    "dataset_type": DatasetType.ULTRALYTICSNDJSONKEYPOINTS,
                    "ndjson_task": "pose",
                },
            ),
        }
        spec = EXPORTER_MAP.get(dataset_type)
        if spec is None:
            raise NotImplementedError(
                f"Unsupported export format: {dataset_type}"
            )

        logger.info(
            f"Exporting '{self.identifier}' to '{dataset_type.name}' format"
        )

        out_path = Path(output_path)
        if out_path.exists():
            raise ValueError(
                f"Export path '{out_path}' already exists. Please remove it first."
            )
        out_path.mkdir(parents=True)

        prepared_ldf = PreparedLDF.from_dataset(self)

        exporter: BaseExporter = spec.cls(
            self.identifier, out_path, max_partition_size_gb, **spec.kwargs
        )

        exporter.export(prepared_ldf=prepared_ldf)

        # Detect whether partitioned export was produced and the max part index
        def _detect_last_part(base: Path, ds_id: str) -> int | None:
            max_idx: int | None = None
            prefix = f"{ds_id}_part"
            for p in base.iterdir():
                if p.is_dir() and p.name.startswith(prefix):
                    try:
                        idx = int(p.name[len(prefix) :])
                    except ValueError:
                        continue
                    max_idx = (
                        idx if (max_idx is None or idx > max_idx) else max_idx
                    )
            return max_idx

        last_part = _detect_last_part(out_path, self.identifier)
        if zip_output:
            archives = create_zip_output(
                max_partition_size=max_partition_size_gb,
                output_path=out_path,
                part=last_part,
                dataset_identifier=self.identifier,
            )
            if isinstance(archives, list):
                logger.info(
                    f"Dataset successfully exported to: {[str(p) for p in archives]}"
                )
                return archives
            logger.info(f"Dataset successfully exported to: {archives}")
            return archives

        logger.info(f"Dataset successfully exported to: {out_path}")
        return out_path

    def get_statistics(
        self, sample_size: int | None = None, view: str | None = None
    ) -> dict[str, Any]:
        """Return dataset statistics for a view or the full dataset.

        The returned statistics include:

            - ``"duplicates"``: Analysis of duplicated content.
            - ``"class_distributions"``: Class frequencies organized by
              task name and task type. Classification tasks are excluded.
            - ``"missing_annotations"``: File paths that lack annotations.
            - ``"heatmaps"``: Spatial annotation distributions.

        Args:
            sample_size: Optional number of samples used for heatmap
                generation.
            view: Optional split name to analyze. If omitted, the entire
                dataset is analyzed.

        Returns:
            Dataset statistics.

        """
        df = self._load_df_offline(lazy=True)

        stats = {
            "duplicates": {},
            "missing_annotations": 0,
            "heatmaps": {},
            "class_distributions": {},
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
        """Remove duplicate files and annotations from the dataset.

        Raises:
            FileNotFoundError: If the dataset is empty.

        """
        df = self._load_df_offline(lazy=True, raise_when_empty=True)
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
            self._fs.put_dir(
                local_paths=self._local_path / "annotations",
                remote_dir="annotations",
                copy_contents=True,
            )
        logger.info(
            "Successfully removed duplicate files and annotations from the dataset."
        )

    def set_class_order_per_task(
        self, class_order_per_task: dict[str, list[str]]
    ) -> None:
        """Set class order for specific tasks.

        Args:
            class_order_per_task: Mapping from task names to class names in
                the desired order.

        Raises:
            ValueError: If a task is missing or the provided class names do
                not match the dataset classes for that task.

        """
        for task_name, task_classes in class_order_per_task.items():
            if task_name not in self.get_tasks():
                raise ValueError(
                    f"Task {task_name} not found in dataset tasks. "
                    f"Available tasks: {list(self.get_tasks().keys())}"
                )
            if set(task_classes) != set(self.get_classes()[task_name].keys()):
                raise ValueError(
                    f"Classes for task {task_name} do not match "
                    f"the classes in the dataset. "
                    f"Expected: {set(self.get_classes()[task_name].keys())}, "
                    f"Got: {set(task_classes)}."
                )

            current_classes = list(self.get_classes()[task_name].keys())
            if task_classes != current_classes:
                logger.warning(
                    f"Reordering classes for task {task_name}. "
                    f"Original order: {current_classes}, "
                    f"New order: {task_classes}."
                )

                self.set_classes(
                    classes={
                        class_name: i
                        for i, class_name in enumerate(task_classes)
                    },
                    task=task_name,
                    rewrite_metadata=False,
                )


def _walk_detections(detections: Iterable[Detection]) -> Iterator[Detection]:
    """Yield every detection, and every detection nested under it."""
    for detection in detections:
        yield detection
        yield from _walk_detections(detection.sub_detections.values())
