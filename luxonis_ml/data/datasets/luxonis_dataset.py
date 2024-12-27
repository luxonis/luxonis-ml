import json
import logging
import math
import shutil
import tempfile
from collections import defaultdict
from contextlib import suppress
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
    overload,
)

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from ordered_set import OrderedSet
from semver.version import Version
from typing_extensions import Self, override

from luxonis_ml.data.utils import (
    BucketStorage,
    BucketType,
    ParquetFileManager,
    infer_task,
    warn_on_duplicates,
)
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.typing import PathType
from luxonis_ml.utils import (
    LuxonisFileSystem,
    deprecated,
    environ,
    make_progress_bar,
)

from .annotation import DatasetRecord
from .base_dataset import BaseDataset, DatasetIterator
from .source import LuxonisSource
from .utils import find_filepath_uuid, get_dir, get_file

logger = logging.getLogger(__name__)


class Skeletons(TypedDict):
    labels: List[str]
    edges: List[Tuple[int, int]]


class Metadata(TypedDict):
    source: LuxonisSource.LuxonisSourceDocument
    ldf_version: str
    classes: Dict[str, List[str]]
    tasks: List[str]
    skeletons: Dict[str, Skeletons]


class LuxonisDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        team_id: Optional[str] = None,
        bucket_type: BucketType = BucketType.INTERNAL,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
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

        self.base_path = environ.LUXONISML_BASE_PATH
        self.base_path.mkdir(exist_ok=True)

        self._credentials = self._init_credentials()
        self._is_synced = False

        self.bucket_type = bucket_type
        self.bucket_storage = bucket_storage

        if self.bucket_storage == BucketStorage.AZURE_BLOB:
            raise NotImplementedError("Azure Blob Storage not yet supported")

        self.bucket = self._get_credential("LUXONISML_BUCKET")

        if self.is_remote and self.bucket is None:
            raise ValueError("Must set LUXONISML_BUCKET environment variable!")

        self.dataset_name = dataset_name
        self.team_id = team_id or self._get_credential("LUXONISML_TEAM_ID")

        if delete_existing:
            if LuxonisDataset.exists(
                dataset_name, team_id, bucket_storage, self.bucket
            ):
                LuxonisDataset(
                    dataset_name, team_id, bucket_type, bucket_storage
                ).delete_dataset(delete_remote=delete_remote)

        self._init_paths()

        if not self.is_remote:
            self.fs = LuxonisFileSystem(f"file://{self.path}")
        else:
            self.fs = LuxonisFileSystem(self.path)

        self.metadata = cast(Metadata, defaultdict(dict, self._get_metadata()))

        if self.version != LDF_VERSION:
            logger.warning(
                f"LDF versions do not match. The current `luxonis-ml` "
                f"installation supports LDF v{LDF_VERSION}, but the "
                f"`{self.identifier}` dataset is in v{self.metadata['ldf_version']}. "
                "Internal migration will be performed. Note that some parts "
                "and new features might not work correctly unless you "
                "manually re-create the dataset using the latest version "
                "of `luxonis-ml`."
            )
        self.progress = make_progress_bar()

    @cached_property
    def version(self) -> Version:
        return Version.parse(
            self.metadata["ldf_version"], optional_minor_and_patch=True
        )

    @property
    def source(self) -> LuxonisSource:
        if "source" not in self.metadata:
            raise ValueError("Source not found in metadata")
        return LuxonisSource.from_document(self.metadata["source"])

    @property
    @override
    def identifier(self) -> str:
        return self.dataset_name

    def __len__(self) -> int:
        """Returns the number of instances in the dataset."""

        df = self._load_df_offline()
        return len(df.select("uuid").unique()) if df is not None else 0

    def _get_credential(self, key: str) -> str:
        """Gets secret credentials from credentials file or ENV
        variables."""

        if key in self._credentials.keys():
            return self._credentials[key]
        else:
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

    @overload
    def _load_df_offline(
        self, lazy: Literal[False] = ...
    ) -> Optional[pl.DataFrame]: ...

    @overload
    def _load_df_offline(
        self, lazy: Literal[True] = ...
    ) -> Optional[pl.LazyFrame]: ...

    def _load_df_offline(
        self, lazy: bool = False
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        path = get_dir(self.fs, "annotations", self.local_path)

        if path is None or not path.exists():
            return None

        if lazy:
            dfs = [pl.scan_parquet(file) for file in path.glob("*.parquet")]
            df = pl.concat(dfs) if dfs else None
        else:
            dfs = [pl.read_parquet(file) for file in path.glob("*.parquet")]
            df = pl.concat(dfs) if dfs else None

        if self.version == LDF_VERSION or df is None:
            return df

        return (
            df.rename({"class": "class_name"})
            .with_columns(
                [
                    pl.col("task").alias("task_type"),
                    pl.col("task").alias("task_name"),
                    pl.lit("image").alias("source_name"),
                ]
            )
            .select(
                [
                    "file",
                    "source_name",
                    "task_name",
                    "created_at",
                    "class_name",
                    "instance_id",
                    "task_type",
                    "annotation",
                    "uuid",
                ]
            )
        )

    @overload
    def _get_file_index(
        self, lazy: Literal[False] = ...
    ) -> Optional[pl.DataFrame]: ...

    @overload
    def _get_file_index(
        self, lazy: Literal[True] = ...
    ) -> Optional[pl.LazyFrame]: ...

    def _get_file_index(
        self, lazy: bool = False
    ) -> Optional[Union[pl.DataFrame, pl.LazyFrame]]:
        path = get_file(
            self.fs, "metadata/file_index.parquet", self.metadata_path
        )
        if path is not None and path.exists():
            if not lazy:
                df = pl.read_parquet(path)
            else:
                df = pl.scan_parquet(path)

            return df.select(pl.all().exclude("^__index_level_.*$"))

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
        path.write_text(json.dumps(self.metadata, indent=4))
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
        if self.fs.exists("metadata/metadata.json"):
            path = get_file(
                self.fs,
                "metadata/metadata.json",
                self.metadata_path,
                default=self.metadata_path / "metadata.json",
            )
            return json.loads(path.read_text())
        else:
            return {
                "source": LuxonisSource().to_document(),
                "ldf_version": str(LDF_VERSION),
                "classes": {},
                "tasks": [],
                "skeletons": {},
            }

    @property
    def is_remote(self) -> bool:
        return self.bucket_storage != BucketStorage.LOCAL

    @override
    def update_source(self, source: LuxonisSource) -> None:
        """Updates underlying source of the dataset with a new
        L{LuxonisSource}.

        @type source: LuxonisSource
        @param source: The new C{LuxonisSource} to replace the old one.
        """

        self.metadata["source"] = source.to_document()
        self._write_metadata()

    @override
    def set_classes(
        self, classes: List[str], task: Optional[str] = None
    ) -> None:
        if task is not None:
            self.metadata["classes"][task] = classes
        else:
            raise NotImplementedError(
                "Setting classes for all tasks not yet supported. "
                "Set classes individually for each task"
            )
        self._write_metadata()

    @override
    def get_classes(self) -> Tuple[List[str], Dict[str, List[str]]]:
        all_classes = list(
            {
                c
                for classes in self.metadata["classes"].values()
                for c in classes
            }
        )
        return sorted(all_classes), self.metadata["classes"]

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
        for task in tasks:
            self.metadata["skeletons"][task] = {
                "labels": labels or [],
                "edges": edges or [],
            }
        self._write_metadata()

    @override
    def get_skeletons(
        self,
    ) -> Dict[str, Tuple[List[str], List[Tuple[int, int]]]]:
        return {
            task: (skel["labels"], skel["edges"])
            for task, skel in self.metadata["skeletons"].items()
        }

    @override
    def get_tasks(self) -> List[str]:
        return self.metadata.get("tasks", [])

    def sync_from_cloud(self, force: bool = False) -> None:
        """Downloads data from a remote cloud bucket."""

        if not self.is_remote:
            logger.warning("This is a local dataset! Cannot sync")
        else:
            if not self._is_synced or force:
                logger.info("Syncing from cloud...")
                local_dir = self.base_path / "data" / self.team_id / "datasets"
                local_dir.mkdir(exist_ok=True, parents=True)

                self.fs.get_dir(remote_paths="", local_dir=local_dir)

                self._is_synced = True
            else:
                logger.warning("Already synced. Use force=True to resync")

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
            logger.info(f"Deleted dataset {self.dataset_name}")

        if self.is_remote and delete_remote:
            logger.info(f"Deleting dataset {self.dataset_name} from cloud")
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
        paths = set(data.file for data in data_batch)
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
        index = self._get_file_index()
        new_index = {"uuid": [], "file": [], "original_filepath": []}
        processed_uuids = set()

        data_batch: list[DatasetRecord] = []

        classes_per_task: Dict[str, OrderedSet[str]] = defaultdict(
            lambda: OrderedSet([])
        )
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
                explicit_task = False
                if not isinstance(record, DatasetRecord):
                    explicit_task = "task_name" in record or "task" in record
                    record = DatasetRecord(**record)
                ann = record.annotation
                if ann is not None:
                    if not explicit_task:
                        record.task = infer_task(
                            record.task, ann.class_name, self.get_classes()[1]
                        )
                    if ann.class_name is not None:
                        classes_per_task[record.task].add(ann.class_name)
                    else:
                        classes_per_task[record.task] = OrderedSet([])
                    if ann.keypoints is not None:
                        num_kpts_per_task[record.task] = len(
                            ann.keypoints.keypoints
                        )

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

        _, curr_classes = self.get_classes()
        for task, classes in classes_per_task.items():
            old_classes = set(curr_classes.get(task, []))
            new_classes = list(classes - old_classes)
            if new_classes or task not in curr_classes:
                logger.info(
                    f"Detected new classes for task {task}: {new_classes}"
                )
                self.set_classes(list(classes | old_classes), task)
        for task, num_kpts in num_kpts_per_task.items():
            self.set_skeletons(
                labels=[str(i) for i in range(num_kpts)],
                edges=[(i, i + 1) for i in range(num_kpts - 1)],
                task=task,
            )

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            self._write_index(index, new_index, path=tmp_file.name)

        self.fs.put_file(tmp_file.name, "metadata/file_index.parquet")
        self._write_metadata()
        self._warn_on_duplicates()
        self._save_tasks_to_metadata()
        return self

    def _save_tasks_to_metadata(self) -> None:
        df = self._load_df_offline()
        if df is None:
            return
        tasks = []
        for task_name, task_type in (
            df.select("task_name", "task_type")
            .unique()
            .drop_nulls()
            .iter_rows()
        ):
            tasks.append(f"{task_name}/{task_type}")
        self.metadata["tasks"] = tasks
        self._write_metadata()

    def _warn_on_duplicates(self) -> None:
        df = self._load_df_offline(lazy=True)
        index = self._get_file_index(lazy=True)
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

        with open(splits_path, "r") as file:
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
            with open(splits_path, "r") as file:
                old_splits = defaultdict(list, json.load(file))

        defined_uuids = set(
            uuid for uuids in old_splits.values() for uuid in uuids
        )

        if definitions is None:
            ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
            df = self._load_df_offline()
            if df is None:
                raise FileNotFoundError("No data found in dataset")
            ids = (
                df.filter(~pl.col("uuid").is_in(defined_uuids))
                .select("uuid")
                .unique()
                .get_column("uuid")
                .to_list()
            )
            if not ids:
                if not replace_old_splits:
                    raise ValueError(
                        "No new files to add to splits. "
                        "If you want to generate new splits, set `replace_old_splits=True`"
                    )
                else:
                    ids = (
                        df.select("uuid").unique().get_column("uuid").to_list()
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
            index = self._get_file_index()
            if index is None:
                raise FileNotFoundError("File index not found")
            for split, filepaths in definitions.items():
                splits_to_update.append(split)
                if not isinstance(filepaths, list):
                    raise ValueError(
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
        """Returns a dictionary of all datasets.

        @type team_id: Optional[str]
        @param team_id: Optional team identifier
        @type bucket_storage: BucketStorage
        @param bucket_storage: Underlying bucket storage from C{local},
            C{S3}, or C{GCS}. Default is C{local}.
        @type bucket: Optional[str]
        @param bucket: Name of the bucket. Default is C{None}.
        @rtype: List[str]
        @return: List of all dataset names.
        """
        base_path = environ.LUXONISML_BASE_PATH

        team_id = team_id or environ.LUXONISML_TEAM_ID
        names = []

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

        for path in fs.walk_dir("", recursive=False, typ="directory"):
            path = Path(path)
            metadata_path = path / "metadata" / "metadata.json"
            if not fs.exists(metadata_path):
                continue
            metadata_text = fs.read_text(metadata_path)
            if isinstance(metadata_text, bytes):
                metadata_text = metadata_text.decode()
            names.append(path.name)
        return names
