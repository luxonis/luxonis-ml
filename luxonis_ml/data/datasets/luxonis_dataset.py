import json
import math
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
    ParquetRecord,
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
from luxonis_ml.data.utils.data_utils import get_keypoint_row_widths
from luxonis_ml.data.utils.ldf_equivalence import ldf_equivalent
from luxonis_ml.data.utils.parquet import DEFAULT_METADATA
from luxonis_ml.enums.enums import DatasetType
from luxonis_ml.ldf import (
    Category,
    DatasetRecord,
    Detection,
    KeypointMetadata,
    load_annotation,
)
from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisFileSystem, environ

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
    @override
    def version(self) -> Version:
        """The version of the underlying LDF that the dataset adheres
        to.
        """
        return self._metadata.version

    @property
    def metadata(self) -> Metadata:
        """Get the dataset metadata.

        Returns:
            Deep copy of the dataset metadata.

        """
        return self._metadata.model_copy(deep=True)

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
            ValueError: If the datasets have different major LDF versions.

        """
        if not (inplace or new_dataset_name):
            raise ValueError(
                "You must specify a name for the new dataset "
                "when `inplace` is `False`"
            )
        if not inplace and self._bucket_storage != other._bucket_storage:
            raise ValueError(
                "Cannot merge datasets with different bucket storage types."
            )
        # The metadata merge and the reads below can raise. The clone and
        # the writes come after them, so a failed merge changes nothing.
        merged_metadata = self._metadata.merge_with(other._metadata)

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

        # `_load_df_offline` appends the `sample_metadata` column that LDF
        # 2.0 lacks, so the two datasets can order their columns differently.
        df_merged = pl.concat([df_self, df_other], how="diagonal_relaxed")

        splits_other = {
            split_name: [
                group_id
                for group_id in group_ids
                if group_id not in duplicate_group_ids
            ]
            for split_name, group_ids in splits_other.items()
        }
        self._merge_splits(splits_self, splits_other)

        target_dataset = (
            self.clone(new_dataset_name, push_to_cloud=False, team_id=team_id)
            if new_dataset_name and not inplace
            else self
        )
        target_dataset._save_df_offline(df_merged)
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

        target_dataset._metadata = merged_metadata
        target_dataset._write_metadata()

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
        # key, or an older luxonis-ml refuses to open it. An empty entry
        # describes no keypoints.
        exclude: set[str] = set()
        version = self.version
        if all(
            entry == KeypointMetadata()
            for entry in self._metadata.keypoint_metadata.values()
        ):
            exclude.add("keypoint_metadata")
        elif version.major == LDF_VERSION.major and version < LDF_VERSION:
            # With the key, the file needs the current LDF version. Another
            # major version keeps its number, because `_load_df_offline`
            # migrates the rows by that number.
            self._metadata.ldf_version = str(LDF_VERSION)
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
        infer_flip_pairs: bool | None = None,
    ) -> None:
        updates = {
            field: value
            for field, value in (
                ("labels", labels),
                ("edges", edges),
                ("flip_pairs", flip_pairs),
                ("sigmas", sigmas),
            )
            if value is not None
        }
        if not updates:
            raise ValueError(
                "Must provide either keypoint names, edges, flip pairs, "
                "or sigmas"
            )

        tasks = self.get_task_names() if task is None else [task]
        # No task changes until every task passes the checks.
        updated: dict[str, KeypointMetadata] = {}
        for t in tasks:
            current = self._metadata.keypoint_metadata.get(t)
            kept = {}
            if current is not None and not _renames_keypoints(current, labels):
                # Names for placeholder keypoints keep the stored fields,
                # but not the chain edges that `add` invented.
                named = labels is not None and labels != current.labels
                kept = current.model_dump(
                    exclude=_placeholder_fields(current) if named else None
                )
            keypoint_metadata = KeypointMetadata.model_validate(
                {**kept, **updates}
            )
            # Stored names from an older luxonis-ml can repeat. A call that
            # gives no names keeps them.
            if labels is not None:
                keypoint_metadata.validate_labels(f"task '{t}'")
            # Stored edges from an older luxonis-ml can be out of range, so
            # only the edges of the call are checked.
            if keypoint_metadata.labels:
                keypoint_metadata.validate_for(
                    len(keypoint_metadata.labels),
                    f"task '{t}'",
                    check_edges=edges is not None,
                )
            updated[t] = _fill_in_flip_pairs(
                keypoint_metadata,
                current,
                infer=infer_flip_pairs if flip_pairs is None else False,
            )

        self._metadata.keypoint_metadata.update(updated)
        self._write_metadata()

    @override
    def get_keypoint_metadata(self) -> dict[str, KeypointMetadata]:
        return dict(self._metadata.keypoint_metadata)

    @override
    def get_n_keypoints(self) -> dict[str, int]:
        """Return the number of keypoints for each task.

        A task without labels counts the keypoints of its widest stored
        row. Edges cannot give the count, because they need not reach the
        last keypoint.

        Returns:
            Number of keypoints keyed by task name.

        """
        n_keypoints = self._keypoint_row_widths()
        for task, task_keypoints in self._metadata.keypoint_metadata.items():
            if task_keypoints.labels or task not in n_keypoints:
                n_keypoints[task] = len(task_keypoints.labels)
        return n_keypoints

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

        The pull overwrites the local ``annotations/`` and ``metadata/``
        folders with the remote copies on every call, so local-only edits
        to those folders are lost.

        With ``UpdateMode.MISSING``, a media file counts as missing only
        when both locations are absent:

            - the path in the ``file`` column of the Parquet shard;
            - ``media/<uuid><suffix>``, where the UUID comes from the
              ``uuid`` column of the same shard.

        With ``UpdateMode.ALL``, every media file is downloaded again and
        overwrites the local copy.

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
            if record.annotation is None or record.annotation.array is None:
                continue
            ann = record.annotation.array
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
        `add` generated own nothing, so the records win over those. An
        entry without names aligns nothing, so it is not in the result.
        Stored names that repeat are the exception. They name no keypoint,
        but they still set the number of keypoints of the task.
        """
        aligned = {
            task: entry for task, entry in declared.items() if entry.has_names
        }
        for task, stored in self._metadata.keypoint_metadata.items():
            if stored.has_names or stored.repeated_labels:
                aligned[task] = stored
        return aligned

    def _add_process_batch(
        self,
        data_batch: list[DatasetRecord],
        pfm: ParquetFileManager,
        index: pl.DataFrame | None,
        declared_keypoint_metadata: dict[str, KeypointMetadata],
    ) -> set[tuple[str, str, str]]:
        """Write the rows of a batch.

        Returns:
            The task, the UUID and the file of each keypoint row that the
            batch wrote without the names of its task.

        """
        keypoint_metadata = self._alignment_keypoint_metadata(
            declared_keypoint_metadata
        )
        paths = {path for data in data_batch for path in data.all_file_paths}
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

        # The rows store the array paths that this call sets.
        self._process_arrays(data_batch)

        task = self._progress.add_task(
            "[magenta]Processing data...", total=len(data_batch)
        )

        # The keypoint alignment and the media upload can raise. Both run
        # before `remove_duplicate_uuids` and the first write, so a failed
        # batch changes no row.
        rows: list[tuple[str, ParquetRecord, str]] = []
        with self._progress:
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
                rows.extend(
                    (uuid_dict[row["file"]], row, group_id)
                    for row in record.to_parquet_rows(keypoint_metadata)
                )
                self._progress.update(task, advance=1)
        self._progress.remove_task(task)

        if self.is_remote:
            logger.info("Uploading media...")

            self._fs.put_dir(
                local_paths=paths,
                remote_dir="media",
                uuid_dict=dict(uuid_dict),
            )
            logger.info("Media uploaded")

        if overwrite_uuids:
            pfm.remove_duplicate_uuids(overwrite_uuids)

        logger.info("Saving annotations...")
        for uuid, row, group_id in rows:
            pfm.write(uuid, row, group_id)

        return {
            (row["task_name"], uuid, row["file"])
            for uuid, row, _ in rows
            if row["task_type"] == "keypoints"
            and row["task_name"] not in keypoint_metadata
        }

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
                            "task_name": "animals",

                            "sample_metadata": {
                                "record_id": 123,
                                "camera": "left",
                                "tags": ["night", "warehouse"],
                            },

                            "annotation": {
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
        tasks_with_flip_pairs: set[str] = set()
        unnamed_keypoint_rows: set[tuple[str, str, str]] = set()
        sources: set[str] = set()

        annotations_path = get_dir(
            self._fs,
            "annotations",
            self._local_path,
            default=self._annotations_path,
        )

        index = self._get_index()
        row_widths = self._keypoint_row_widths()
        # The stored entries that every record of a task must fit.
        stored_alignment = self._alignment_keypoint_metadata({})

        assert annotations_path is not None

        with ParquetFileManager(annotations_path, batch_size) as pfm:
            for record in generator:
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
                            num_kpts_per_task[task_name].add(
                                len(ann.keypoints.keypoints)
                            )
                            # An empty list of flip pairs turns the
                            # inference off. Only the fields that the record
                            # sets tell it apart from an omitted list.
                            if "flip_pairs" in ann.keypoints.model_fields_set:
                                tasks_with_flip_pairs.add(task_name)
                            declared = ann.keypoints.declared_metadata()
                            if declared is not None:
                                declared = _in_stored_order(
                                    declared,
                                    self._metadata.keypoint_metadata.get(
                                        task_name
                                    ),
                                    task_name,
                                )
                                earlier = declared_keypoint_metadata.get(
                                    task_name
                                )
                                declared_keypoint_metadata[task_name] = (
                                    declared
                                    if earlier is None
                                    else earlier.merge_with(
                                        declared, f"task '{task_name}'"
                                    )
                                )
                            # `add` does not change stored names, so a later
                            # record cannot make a wider record fit. The
                            # alignment thus fails here, before `add` writes
                            # the batch in front of the record.
                            if task_name in stored_alignment:
                                stored_alignment[task_name].align(
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

                # A full batch waits for the next record. The last batch thus
                # always gets the full check after the loop. A check cannot
                # undo an earlier batch.
                if len(data_batch) == batch_size:
                    self._check_declared_keypoint_metadata(
                        num_kpts_per_task,
                        declared_keypoint_metadata,
                        row_widths,
                    )
                    unnamed_keypoint_rows |= self._add_process_batch(
                        data_batch, pfm, index, declared_keypoint_metadata
                    )
                    data_batch = []
                data_batch.append(record)

            # `add` has read every record, so the check covers the stored
            # fields too.
            resolved_keypoint_metadata = self._resolve_keypoint_metadata(
                num_kpts_per_task,
                declared_keypoint_metadata,
                tasks_with_flip_pairs,
                row_widths,
            )
            self._add_process_batch(
                data_batch, pfm, index, declared_keypoint_metadata
            )

        # A record can name the keypoints of a task after an earlier batch
        # wrote some of its rows. Those rows get the names now, so the rows
        # on disk do not depend on the batch size.
        self._align_keypoint_rows(
            annotations_path, unnamed_keypoint_rows, declared_keypoint_metadata
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

        if num_kpts_per_task:
            self._metadata.keypoint_metadata.update(resolved_keypoint_metadata)
            self._write_metadata()

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
        replace_old_splits: bool = False,
    ) -> None:
        if splits is None:
            splits = {"train": 0.8, "val": 0.1, "test": 0.1}

        ratios, definitions = _resolve_splits(splits)

        splits_path = get_file(
            self._fs,
            "metadata/splits.json",
            self._metadata_path,
            default=self._metadata_path / "splits.json",
        )
        old_splits: dict[str, list[str]] = defaultdict(list)
        if splits_path.exists():
            with open(splits_path) as file:
                old_splits = defaultdict(list, json.load(file))

        defined_group_ids: set[str] = set()
        if not replace_old_splits:
            for group_ids in old_splits.values():
                defined_group_ids.update(group_ids)

        if ratios is not None:
            new_splits = self._split_by_ratio(ratios, defined_group_ids)
        else:
            assert definitions is not None
            new_splits = self._split_by_definition(
                definitions, defined_group_ids
            )
            if not any(new_splits.values()):
                if not replace_old_splits:
                    logger.warning(
                        "No new files to add to splits. "
                        "The existing splits are left unchanged."
                    )
                    return
                logger.warning(
                    "No file from the definitions is in the dataset. "
                    "The new splits are empty."
                )

        if replace_old_splits:
            old_splits.clear()

        for split, group_ids in new_splits.items():
            old_splits[split].extend(group_ids)

        _write_json(splits_path, old_splits)

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
        n_rows = df.select(pl.len()).collect().item()
        df = df.filter(~pl.col("file").is_in(duplicate_files_to_remove))

        deduplicated = df.unique(
            subset=["file", "annotation"], maintain_order=True
        ).collect()

        # Both steps above only drop rows. An equal row count thus means
        # the data did not change, so the rewrite and the upload below
        # would only repeat the current contents.
        if deduplicated.height == n_rows:
            logger.info("The dataset has no duplicates.")
            return

        self._save_df_offline(deduplicated)

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

    def _split_by_ratio(
        self, ratios: Mapping[str, float], defined_group_ids: set[str]
    ) -> dict[str, list[str]]:
        """Divide the unassigned groups between the splits.

        Args:
            ratios: Split names mapped to ratios that sum to 1.
            defined_group_ids: Groups that already belong to a split.

        Returns:
            Split names mapped to the group ids they receive.

        Raises:
            ValueError: If every group already belongs to a split.

        """
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
            raise ValueError(
                "No new files to add to splits. "
                "If you want to generate new splits, set "
                "`replace_old_splits=True`"
            )

        np.random.shuffle(ids)
        new_splits: dict[str, list[str]] = {}
        lower_bound = 0
        for split, size in _split_sizes(len(ids), ratios).items():
            upper_bound = lower_bound + size
            new_splits[split] = ids[lower_bound:upper_bound]
            lower_bound = upper_bound

        for split, group_ids in new_splits.items():
            if not group_ids and ratios[split] > 0:
                logger.warning(
                    f"Split '{split}' got no data. The dataset has "
                    f"{len(ids)} new groups, which is too few for "
                    f"the ratio {ratios[split]}."
                )
        return new_splits

    def _split_by_definition(
        self,
        definitions: Mapping[str, Sequence[PathType]],
        defined_group_ids: set[str],
    ) -> dict[str, list[str]]:
        """Resolve filepath lists to group ids.

        The method adds each group that it takes to
        ``defined_group_ids``, so no group lands in two splits.

        Args:
            definitions: Split names mapped to filepath lists.
            defined_group_ids: Groups that already belong to a split.

        Returns:
            Split names mapped to the group ids they receive.

        """
        n_files = sum(map(len, definitions.values()))
        dataset_size = len(self)
        if n_files > dataset_size:
            logger.warning(
                "Dataset size is smaller than the total number of files in the definitions. "
                f"Dataset size: {dataset_size}, Definitions: {n_files}. "
                "Duplicate files will be filtered out and extra files in definitions will be ignored."
            )
            self.remove_duplicates()

        index = self._get_index(raise_when_empty=True)
        new_splits: dict[str, list[str]] = {}
        for split, filepaths in definitions.items():
            ids: list[str] = []
            for filepath in filepaths:
                if not isinstance(filepath, (str, Path)):
                    logger.warning(
                        f"Split '{split}' contains {filepath!r}, "
                        f"which is a {type(filepath).__name__} and "
                        "not a filepath; skipping."
                    )
                    continue

                group_id = find_filepath_group_id(
                    filepath, index, raise_on_missing=False
                )

                if group_id is None:
                    logger.warning(
                        f"No group ID found for '{filepath}' in definitions; skipping."
                    )
                    continue
                if group_id in defined_group_ids:
                    continue
                ids.append(group_id)
                defined_group_ids.add(group_id)

            new_splits[split] = ids
        return new_splits

    def _align_keypoint_rows(
        self,
        annotations_path: Path,
        rows: set[tuple[str, str, str]],
        declared_keypoint_metadata: dict[str, KeypointMetadata],
    ) -> None:
        """Write keypoint rows again against the names of their task.

        Args:
            annotations_path: The directory of the parquet files.
            rows: The task, the UUID and the file of each keypoint row to
                write again. A task without names keeps its rows. The UUID
                comes from the bytes of the file, so a copy of a file from
                an earlier `add` has the same UUID. The file tells the rows
                of this `add` apart, because `add` removes the old rows of
                a file that it adds again.
            declared_keypoint_metadata: The keypoint metadata that the
                records of `add` describe.

        """
        keypoint_metadata = self._alignment_keypoint_metadata(
            declared_keypoint_metadata
        )
        rows = {row for row in rows if row[0] in keypoint_metadata}
        if not rows:
            return
        uuids = [uuid for _, uuid, _ in rows]
        for path in annotations_path.glob("*.parquet"):
            df = pl.read_parquet(path)
            if not df["uuid"].is_in(uuids).any():
                continue
            annotations = [
                load_annotation(
                    "keypoints", json.loads(annotation)
                ).to_parquet_json(keypoint_metadata[task])
                if task_type == "keypoints" and (task, uuid, file) in rows
                else annotation
                for task, uuid, file, task_type, annotation in df.select(
                    "task_name", "uuid", "file", "task_type", "annotation"
                ).iter_rows()
            ]
            if annotations != df["annotation"].to_list():
                df.with_columns(
                    pl.Series("annotation", annotations)
                ).write_parquet(path)

    def _resolve_keypoint_metadata(
        self,
        num_kpts_per_task: dict[str, set[int]],
        declared: dict[str, KeypointMetadata],
        tasks_with_flip_pairs: set[str],
        row_widths: dict[str, int],
    ) -> dict[str, KeypointMetadata]:
        """Return the keypoint metadata that `add` stores for each task.

        A keypoint task with no keypoint metadata gets placeholder names
        and chain edges. The placeholder fields grow with the keypoint
        count, and an explicit value survives an `add`. A task that keeps
        its stored names is not in the result.

        Args:
            num_kpts_per_task: The numbers of keypoints in the records of
                each task.
            declared: The keypoint metadata that the records describe.
            tasks_with_flip_pairs: The tasks with a record that gives flip
                pairs. An empty list counts too, so these tasks get no
                inferred flip pairs.
            row_widths: The result of `_keypoint_row_widths`.

        Returns:
            The new keypoint metadata, keyed by task name.

        Raises:
            ValueError: If an entry does not fit the keypoints of its task.

        """
        resolved: dict[str, KeypointMetadata] = {}
        # `add` pads each row of a task with names to the number of names,
        # and it rejects a wider row. Only an entry without names can grow.
        aligned = self._alignment_keypoint_metadata(declared)
        for task, sizes in num_kpts_per_task.items():
            n_keypoints = self._keypoint_count(task, sizes, row_widths)
            stored = self._metadata.keypoint_metadata.get(task)
            described = declared.get(task)
            if described is not None and stored is not None:
                # A placeholder field describes nothing, so replacing it
                # needs no warning.
                generated = _placeholder_fields(stored)
                for field in described.conflicting_fields(stored):
                    if field not in generated:
                        logger.warning(
                            f"The annotations of task '{task}' describe a "
                            f"different `{field}` than the one already "
                            "stored. Using the described one. Stored: "
                            f"{getattr(stored, field)}, described: "
                            f"{getattr(described, field)}."
                        )
            if described is not None:
                entry = _merge_into_stored(described, stored, n_keypoints)
            elif stored is None or stored == KeypointMetadata():
                # An empty entry describes no keypoints, and
                # `_write_metadata` can leave it out of the file. It thus
                # gets the same result as no entry.
                entry = _placeholder_keypoint_metadata(n_keypoints)
            elif task in aligned:
                continue
            else:
                entry = _merge_into_stored(
                    KeypointMetadata(), stored, n_keypoints
                )
            if not entry.labels:
                # Only the labels give the keypoint count. Edges or sigmas
                # alone name nothing, so the records and the stored rows
                # give the count, and the stored entry has to carry it.
                entry = entry.model_copy(
                    update={
                        "labels": _placeholder_keypoint_metadata(
                            n_keypoints
                        ).labels
                    }
                )
            # A stored field must fit the keypoint count too, except the
            # edges: an older luxonis-ml stored them without a check, and
            # `Metadata` still accepts them. Each record checks the edges
            # that it gives, and so does `set_keypoint_metadata`.
            entry.validate_for(
                n_keypoints, f"task '{task}'", check_edges=False
            )
            resolved[task] = _fill_in_flip_pairs(
                entry,
                stored,
                infer=False if task in tasks_with_flip_pairs else None,
            )
            if len(sizes) > 1 and task not in aligned:
                # The labels give the keypoint count, and an earlier `add`
                # can make it larger than any record of this one.
                logger.warning(
                    f"Task '{task}' mixes annotations with different "
                    f"numbers of keypoints ({sorted(sizes)}). Storing "
                    f"keypoint metadata for {len(entry.labels)} keypoints."
                )
        return resolved

    def _check_declared_keypoint_metadata(
        self,
        num_kpts_per_task: dict[str, set[int]],
        declared: dict[str, KeypointMetadata],
        row_widths: dict[str, int],
    ) -> None:
        """Check the declared fields against what `add` knows so far.

        A later record cannot fix a failure of this check. It can only add
        keypoints, and a declared field does not change. The check also
        uses the stored labels and rows, because `add` does not make a task
        smaller. A later record can replace any other stored field, so only
        `_resolve_keypoint_metadata` checks those fields.

        Raises:
            ValueError: If a declaration does not fit the keypoints of its
                task.

        """
        for task, task_keypoint_metadata in declared.items():
            task_keypoint_metadata.validate_for(
                self._keypoint_count(
                    task, num_kpts_per_task[task], row_widths
                ),
                f"task '{task}'",
            )

    def _keypoint_count(
        self, task: str, sizes: set[int], row_widths: dict[str, int]
    ) -> int:
        """Return the number of keypoints of a task after `add`.

        The loader pads each stored row to the number of stored labels.
        `add` does not make the task smaller, because a stored row can be
        that wide. A task without labels has only its rows, so
        ``row_widths`` gives its widest row.
        """
        stored = self._metadata.keypoint_metadata.get(task, KeypointMetadata())
        return max(*sizes, len(stored.labels), row_widths.get(task, 0))

    def _keypoint_row_widths(self) -> dict[str, int]:
        """Return the keypoint count of the widest stored row of each task.

        The result holds only the tasks without stored labels. The labels
        of the other tasks cover their rows. A call with empty labels
        removes the labels of a task. An older luxonis-ml also stored tasks
        without labels.
        """
        df = self._load_df_offline(lazy=True)
        if df is None:
            return {}
        labelled = [
            task
            for task, entry in self._metadata.keypoint_metadata.items()
            if entry.labels
        ]
        return get_keypoint_row_widths(
            df.filter(~pl.col("task_name").is_in(labelled))
        )


def _in_stored_order(
    declared: KeypointMetadata, stored: KeypointMetadata | None, task: str
) -> KeypointMetadata:
    """Move the declaration of a record to the stored names.

    The stored names own the label set. A record can name a subset of
    them, in any order, and its indices point into its own names.

    Raises:
        ValueError: If the record names its keypoints, but the stored
            names of the task repeat.

    """
    if not declared.labels or stored is None:
        return declared
    # `add` writes the rows in the stored order. A repeated name has no
    # single position, so a record with names cannot be written.
    if stored.repeated_labels:
        raise ValueError(
            f"Task '{task}' repeats the keypoint names "
            f"{', '.join(stored.repeated_labels)}, so a record cannot "
            "refer to its keypoints by name. Give each keypoint a "
            "unique name with "
            "`LuxonisDataset.set_keypoint_metadata(labels=...)`. You "
            "can also give the keypoints of the record as a list."
        )
    if stored.has_names and declared.labels != stored.labels:
        return declared.reindexed_to(stored.labels)
    return declared


def _merge_into_stored(
    declared: KeypointMetadata,
    stored: KeypointMetadata | None,
    n_keypoints: int,
) -> KeypointMetadata:
    """Combine described keypoints with the ones already stored.

    A described value wins. A field that the description leaves empty
    comes from the stored keypoint metadata. `LuxonisDataset.add` thus
    discards nothing, except the placeholder fields that it generated
    itself. Real names drop those. Without names, it generates them again
    for ``n_keypoints``. It already moved the description to the stored
    names, so the stored labels keep their order.
    """
    if stored is None:
        return declared
    placeholder = (
        KeypointMetadata()
        if declared.has_names
        else _placeholder_keypoint_metadata(n_keypoints)
    )
    return declared.filled_from(
        stored.model_copy(
            update={
                field: getattr(placeholder, field)
                for field in _placeholder_fields(stored)
            }
        )
    )


def _fill_in_flip_pairs(
    keypoint_metadata: KeypointMetadata,
    stored: KeypointMetadata | None,
    *,
    infer: bool | None,
) -> KeypointMetadata:
    """Infer flip pairs from the keypoint names when none are known.

    Only a write path infers them, never a read path. A read path would
    give flip pairs to a dataset that never asked for them.

    Args:
        keypoint_metadata: The new keypoint metadata of the task.
        stored: The stored keypoint metadata of the task.
        infer: ``True`` infers flip pairs for all names, and ``False``
            infers none. ``None`` infers them only for names that are new
            to the task. The stored entry does not record that the
            inference is off, so its empty list can mean that.

    """
    labels = keypoint_metadata.labels
    if infer is None:
        infer = stored is None or stored.labels != labels
    if not infer or keypoint_metadata.flip_pairs:
        return keypoint_metadata
    return keypoint_metadata.model_copy(
        update={"flip_pairs": KeypointMetadata.infer_flip_pairs(labels)}
    )


def _renames_keypoints(
    current: KeypointMetadata, labels: list[str] | None
) -> bool:
    """Whether new labels give the stored indices a new meaning.

    ``edges``, ``flip_pairs``, and ``sigmas`` use positional keypoint
    indices. New names put a different keypoint at a position, so the
    stored values describe the wrong keypoints. Placeholder names carry
    no identity. A rename that keeps their count therefore only names the
    keypoints that are already there.
    """
    if labels is None or not current.labels or current.labels == labels:
        return False
    return current.has_names or len(current.labels) != len(labels)


def _placeholder_keypoint_metadata(n_keypoints: int) -> KeypointMetadata:
    """Build what `LuxonisDataset.add` writes for keypoints that name none."""
    return KeypointMetadata(
        labels=[str(i) for i in range(n_keypoints)],
        edges=[(i, i + 1) for i in range(n_keypoints - 1)],
    )


def _placeholder_fields(stored: KeypointMetadata) -> set[str]:
    """Return the fields that still hold what `LuxonisDataset.add` made.

    The chain edges join keypoints only by their position, so they do not
    describe keypoints with names. An entry without labels holds nothing
    that was generated.
    """
    placeholder = _placeholder_keypoint_metadata(len(stored.labels))
    if not stored.labels or stored.labels != placeholder.labels:
        return set()
    return {
        field
        for field in KeypointMetadata.model_fields
        if getattr(stored, field) == getattr(placeholder, field)
    }


def _resolve_splits(
    splits: Mapping[str, Sequence[PathType]]
    | Mapping[str, float]
    | tuple[float, float, float],
) -> tuple[dict[str, float] | None, Mapping[str, Sequence[PathType]] | None]:
    """Read the ``splits`` argument as ratios or as filepath lists.

    Args:
        splits: The argument given to ``make_splits``.

    Returns:
        The ratios and the filepath lists. Exactly one of the two is
        ``None``.

    Raises:
        ValueError: If ``splits`` is empty, or if the ratios fall outside
            the range from 0 to 1, or do not sum to 1.
        TypeError: If the mapping values are neither ratios nor filepath
            lists.

    """
    if isinstance(splits, tuple):
        ratios = dict(
            zip(["train", "val", "test"], map(float, splits), strict=True)
        )
    elif not splits:
        raise ValueError("Splits cannot be empty")
    else:
        ratios = {
            split: float(value)
            for split, value in splits.items()
            if isinstance(value, (int, float))
        }
        if len(ratios) < len(splits):
            definitions = {
                split: value
                for split, value in splits.items()
                # A `str` is a Sequence, but one path is not a list.
                if isinstance(value, Sequence) and not isinstance(value, str)
            }
            if len(definitions) < len(splits):
                raise TypeError(
                    "Splits must map names to either ratios or filepath "
                    "lists. A ratio is a number from 0 to 1."
                )
            return None, definitions

    # Check the range first; a bad ratio breaks the sum too.
    if any(not 0.0 <= ratio <= 1.0 for ratio in ratios.values()):
        raise ValueError("Ratios must be between 0.0 and 1.0 (inclusive)")
    sum_ = sum(ratios.values())
    if not math.isclose(sum_, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {sum_:0.4f}")

    return ratios, None


def _write_json(path: Path, data: Mapping[str, list[str]]) -> None:
    """Write JSON to a temporary file, then rename the file.

    ``Path.replace`` is atomic on one filesystem. A failure during the
    write thus leaves the previous file unchanged, instead of a
    truncated file that no longer parses.

    Args:
        path: The destination file.
        data: The data to write.

    """
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=4))
        tmp_path.replace(path)
    finally:
        # `push_to_cloud` uploads the whole metadata directory.
        tmp_path.unlink(missing_ok=True)


def _split_sizes(n_groups: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Divide the groups between the splits.

    The function uses the largest remainder method. Each split first
    gets the whole part of its exact share. The leftover groups then go
    to the splits with the largest fractional parts. A ``ceil`` of each
    share would give every split more than its part in turn, and the
    last split would absorb all the error.

    Args:
        n_groups: The number of groups to divide.
        ratios: A mapping of split names to ratios. The ratios sum to 1.

    Returns:
        A mapping of split names to group counts. The counts sum to
        ``n_groups``.

    """
    shares = {split: n_groups * ratio for split, ratio in ratios.items()}
    sizes = {split: int(share) for split, share in shares.items()}
    leftover = max(0, n_groups - sum(sizes.values()))
    by_remainder = sorted(
        shares, key=lambda split: shares[split] - sizes[split], reverse=True
    )
    for split in by_remainder[:leftover]:
        sizes[split] += 1
    return sizes
