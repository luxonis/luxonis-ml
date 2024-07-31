import json
import logging
import shutil
import tempfile
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pycocotools import mask as mask_utils
from typing_extensions import Self

import luxonis_ml.data.utils.data_utils as data_utils
from luxonis_ml.utils import LuxonisFileSystem, environ, make_progress_bar
from luxonis_ml.utils.filesystem import ModuleType, PathType

from ..utils.constants import LDF_VERSION
from ..utils.enums import BucketStorage, BucketType
from ..utils.parquet import ParquetFileManager
from .annotation import Annotation, ArrayAnnotation, DatasetRecord
from .base_dataset import BaseDataset, DatasetIterator
from .source import LuxonisSource


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
        """Luxonis Dataset Format (LDF) is used to define datasets in the Luxonis MLOps
        ecosystem.

        @type dataset_name: str
        @param dataset_name: Name of the dataset
        @type team_id: Optional[str]
        @param team_id: Optional unique team identifier for the cloud
        @type bucket_type: BucketType
        @param bucket_type: Whether to use external cloud buckets
        @type bucket_storage: BucketStorage
        @param bucket_storage: Underlying bucket storage from local, S3, or GCS
        @type delete_existing: bool
        @param delete_existing: Whether to delete a dataset with the same name if it
            exists
        @type delete_remote: bool
        @param delete_remote: Whether to delete the dataset from the cloud as well
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

        self.metadata = defaultdict(dict, self._get_metadata())

        self.logger = logging.getLogger(__name__)

        self.progress = make_progress_bar()

    @property
    def source(self) -> LuxonisSource:
        if "source" not in self.metadata:
            raise ValueError("Source not found in metadata")
        return LuxonisSource.from_document(self.metadata["source"])

    @property
    def identifier(self) -> str:
        return self.dataset_name

    def __len__(self) -> int:
        """Returns the number of instances in the dataset."""

        df = self._load_df_offline()
        return len(df.select("uuid").unique()) if df is not None else 0

    def _get_credential(self, key: str) -> str:
        """Gets secret credentials from credentials file or ENV variables."""

        if key in self._credentials.keys():
            return self._credentials[key]
        else:
            if not hasattr(environ, key):
                raise RuntimeError(f"Must set {key} in ENV variables")
            return getattr(environ, key)

    def _init_paths(self) -> None:
        """Configures local path or bucket directory."""

        self.local_path = (
            self.base_path / "data" / self.team_id / "datasets" / self.dataset_name
        )
        self.media_path = self.local_path / "media"
        self.annotations_path = self.local_path / "annotations"
        self.metadata_path = self.local_path / "metadata"
        self.masks_path = self.local_path / "masks"

        for path in [self.media_path, self.annotations_path, self.metadata_path]:
            path.mkdir(exist_ok=True, parents=True)

        if not self.is_remote:
            self.path = str(self.local_path)
        else:
            self.path = self._construct_url(
                self.bucket_storage, self.bucket, self.team_id, self.dataset_name
            )

    def _load_df_offline(self) -> Optional[pl.DataFrame]:
        path = self._get_dir("annotations", self.local_path)

        if path is None or not path.exists():
            return None

        dfs = [pl.read_parquet(file) for file in path.glob("*.parquet")]

        return pl.concat(dfs) if dfs else None

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
        path = self._get_file("metadata/file_index.parquet", self.media_path)
        if path is not None and path.exists():
            return pl.read_parquet(path).select(pl.all().exclude("^__index_level_.*$"))
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
        bucket_storage: BucketStorage, bucket: str, team_id: str, dataset_name: str
    ) -> str:
        """Constructs a URL for a remote dataset."""
        return f"{bucket_storage.value}://{bucket}/{team_id}/datasets/{dataset_name}"

    def _init_credentials(self) -> Dict[str, Any]:
        credentials_cache_file = self.base_path / "credentials.json"
        if credentials_cache_file.exists():
            return json.loads(credentials_cache_file.read_text())
        return {}

    def _get_metadata(self) -> Dict[str, Any]:
        if self.fs.exists("metadata/metadata.json"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                self.fs.get_file("metadata/metadata.json", tmp_dir)
                with open(Path(tmp_dir, "metadata.json")) as file:
                    return json.load(file)
        else:
            return {
                "source": LuxonisSource().to_document(),
                "ldf_version": LDF_VERSION,
                "classes": {},
            }

    @property
    def is_remote(self) -> bool:
        return self.bucket_storage != BucketStorage.LOCAL

    def update_source(self, source: LuxonisSource) -> None:
        """Updates underlying source of the dataset with a new L{LuxonisSource}.

        @type source: L{LuxonisSource}
        @param source: The new L{LuxonisSource} to replace the old one.
        """

        self.metadata["source"] = source.to_document()
        self._write_metadata()

    def set_classes(
        self,
        classes: List[str],
        task: Optional[str] = None,
    ) -> None:
        if task is not None:
            self.metadata["classes"][task] = classes
        else:
            raise NotImplementedError(
                "Setting classes for all tasks not yet supported. "
                "Set classes individually for each task"
            )
        self._write_metadata()

    def get_classes(self) -> Tuple[List[str], Dict[str, List[str]]]:
        all_classes = list(
            {c for classes in self.metadata["classes"].values() for c in classes}
        )
        return sorted(all_classes), self.metadata["classes"]

    def set_skeletons(
        self, skeletons: Dict[str, Dict], task: Optional[str] = None
    ) -> None:
        if task is None:
            raise NotImplementedError("Skeletons must be set for a specific task")

        self.metadata["skeletons"][task] = skeletons
        self._write_metadata()

    def get_skeletons(self) -> Dict[str, Dict]:
        return self.metadata["skeletons"]

    def get_tasks(self) -> List[str]:
        return list(self.get_classes()[1].keys())

    def sync_from_cloud(self, force: bool = False) -> None:
        """Downloads data from a remote cloud bucket."""

        if not self.is_remote:
            self.logger.warning("This is a local dataset! Cannot sync")
        else:
            if not self._is_synced or force:
                self.logger.info("Syncing from cloud...")
                local_dir = self.base_path / "data" / self.team_id / "datasets"
                local_dir.mkdir(exist_ok=True, parents=True)

                self.fs.get_dir(remote_paths="", local_dir=local_dir)

                self._is_synced = True
            else:
                self.logger.warning("Already synced. Use force=True to resync")

    def delete_dataset(self, *, delete_remote: bool = False) -> None:
        """Deletes the dataset from local storage and optionally from the cloud.

        @type delete_remote: bool
        @param delete_remote: Whether to delete the dataset from the cloud.
        """
        if not self.is_remote:
            shutil.rmtree(self.path)
            self.logger.info(f"Deleted dataset {self.dataset_name}")

        if self.is_remote and delete_remote:
            self.logger.info(f"Deleting dataset {self.dataset_name} from cloud")
            assert self.path
            assert self.dataset_name
            self.fs.delete_dir(allow_delete_parent=True)

    def _infer_task(self, ann: Annotation) -> str:
        if not hasattr(LuxonisDataset._infer_task, "_logged_infered_classes"):
            LuxonisDataset._infer_task._logged_infered_classes = defaultdict(bool)

        def _log_once(cls_: str, message: str, level: str = "info"):
            if not LuxonisDataset._infer_task._logged_infered_classes[cls_]:
                LuxonisDataset._infer_task._logged_infered_classes[cls_] = True
                getattr(self.logger, level)(message, extra={"markup": True})

        cls_ = ann.class_
        _, current_classes = self.get_classes()
        infered_task = None

        for task, classes in current_classes.items():
            if cls_ in classes:
                if infered_task is not None:
                    _log_once(
                        cls_,
                        f"Class [red italic]{cls_}[reset] is ambiguous between tasks [magenta italic]{infered_task}[reset] and [magenta italic]{task}[reset]. Task inference failed.",
                        "warning",
                    )
                    infered_task = None
                    break
                infered_task = task
        if infered_task is None:
            _log_once(
                cls_,
                f"Task inference for class [red italic]{cls_}[reset] failed. "
                f"Autogenerated task [magenta italic]{ann.task}[reset] will be used.",
                "warning",
            )
        else:
            _log_once(
                cls_,
                f"Class [red italic]{cls_}[reset] infered to belong to task [magenta italic]{infered_task}[reset]",
            )
            return infered_task

        return ann.task

    def _process_arrays(self, batch_data: List[DatasetRecord]) -> None:
        array_paths = set(
            ann.path for ann in batch_data if isinstance(ann, ArrayAnnotation)
        )
        if array_paths:
            task = self.progress.add_task(
                "[magenta]Processing arrays...", total=len(batch_data)
            )
            self.logger.info("Checking arrays...")
            data_utils.check_arrays(array_paths)
            self.logger.info("Generating array UUIDs...")
            array_uuid_dict = self.fs.get_file_uuids(
                array_paths, local=True
            )  # TODO: support from bucket
            if self.is_remote:
                self.logger.info("Uploading arrays...")
                # TODO: support from bucket (likely with a self.fs.copy_dir)
                arrays_upload_dict = self.fs.put_dir(
                    local_paths=array_paths,
                    remote_dir="arrays",
                    uuid_dict=array_uuid_dict,
                )
            self.logger.info("Finalizing paths...")
            self.progress.start()
            for ann in batch_data:
                if isinstance(ann, ArrayAnnotation):
                    if self.is_remote:
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
        uuid_dict = self.fs.get_file_uuids(
            paths, local=True
        )  # TODO: support from bucket
        if self.is_remote:
            self.logger.info("Uploading media...")

            # TODO: support from bucket (likely with a self.fs.copy_dir)
            self.fs.put_dir(local_paths=paths, remote_dir="media", uuid_dict=uuid_dict)
            self.logger.info("Media uploaded")

        task = self.progress.add_task(
            "[magenta]Processing data...", total=len(batch_data)
        )

        self._process_arrays(batch_data)

        self.logger.info("Saving annotations...")
        with self.progress:
            for ann in batch_data:
                filepath = ann.file
                file = filepath.name
                uuid = uuid_dict[str(filepath)]
                matched_id = self._find_filepath_uuid(filepath, index)
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
                    new_index["original_filepath"].append(str(filepath.absolute()))
                    processed_uuids.add(uuid)

                pfm.write({"uuid": uuid, **ann.to_parquet_dict()})
                self.progress.update(task, advance=1)
        self.progress.remove_task(task)

    def add(self, generator: DatasetIterator, batch_size: int = 1_000_000) -> Self:
        generator = _add_generator_wrapper(generator)
        index = self._get_file_index()
        new_index = {"uuid": [], "file": [], "original_filepath": []}
        processed_uuids = set()

        batch_data: list[DatasetRecord] = []

        classes_per_task: Dict[str, Set[str]] = defaultdict(set)
        num_kpts_per_task: Dict[str, int] = {}

        annotations_path = self._get_dir(
            "annotations", self.local_path, default=self.annotations_path
        )
        assert annotations_path is not None

        with ParquetFileManager(annotations_path) as pfm:
            for i, data in enumerate(generator, start=1):
                record = (
                    data if isinstance(data, DatasetRecord) else DatasetRecord(**data)
                )
                ann = record.annotation
                if ann is not None:
                    if ann.task == ann._label_type.value and (
                        isinstance(data, DatasetRecord)
                        or "task" not in data.get("annotation", {})
                    ):
                        ann.task = self._infer_task(ann)

                    classes_per_task[ann.task].add(ann.class_)
                    if ann.type_ == "keypoints":
                        num_kpts_per_task[ann.task] = len(ann.keypoints)

                batch_data.append(record)
                if i % batch_size == 0:
                    self._add_process_batch(
                        batch_data, pfm, index, new_index, processed_uuids
                    )
                    batch_data = []

            self._add_process_batch(batch_data, pfm, index, new_index, processed_uuids)

        with suppress(shutil.SameFileError):
            self.fs.put_dir(annotations_path, "")

        _, curr_classes = self.get_classes()
        for task, classes in classes_per_task.items():
            old_classes = set(curr_classes.get(task, []))
            new_classes = list(classes - old_classes)
            if new_classes:
                self.logger.info(f"Detected new classes for task {task}: {new_classes}")
                self.set_classes(list(classes | old_classes), task)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            self._write_index(index, new_index, path=tmp_file.name)
        self.fs.put_file(tmp_file.name, "metadata/file_index.parquet")
        self._write_metadata()

        return self

    def make_splits(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        definitions: Optional[Dict[str, List[PathType]]] = None,
    ) -> None:
        new_splits = {"train": [], "val": [], "test": []}
        splits_to_update = []

        if definitions is None:
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

        else:
            index = self._get_file_index()
            if index is None:
                raise FileNotFoundError("File index not found")
            for split in ["train", "val", "test"]:
                if split not in definitions:
                    continue
                splits_to_update.append(split)
                filepaths = definitions[split]
                if not isinstance(filepaths, list):
                    raise ValueError("Must provide splits as a list of str")
                ids = [
                    self._find_filepath_uuid(
                        Path(filepath), index, raise_on_missing=True
                    )
                    for filepath in filepaths
                ]
                new_splits[split] = ids

        splits_path = self._get_file(
            "metadata/splits.json",
            self.local_path,
            default=self.metadata_path / "splits.json",
        )
        assert splits_path is not None

        if splits_path.exists():
            with open(splits_path, "r") as file:
                splits = json.load(file)
            for split in splits_to_update:
                splits[split] = new_splits[split]
        else:
            splits = new_splits

        splits_path.write_text(json.dumps(splits, indent=4))

        with suppress(shutil.SameFileError):
            self.fs.put_file(splits_path, "metadata/splits.json")

    @staticmethod
    def exists(
        dataset_name: str,
        team_id: Optional[str] = None,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
        bucket: Optional[str] = None,
    ) -> bool:
        """Checks if a dataset exists.

        @type dataset_name: str
        @param dataset_name: Name of the dataset to check
        @type remote: bool
        @param remote: Whether to check if the dataset exists in the cloud
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

        @rtype: Dict
        @return: Dictionary of all datasets
        """
        base_path = environ.LUXONISML_BASE_PATH

        team_id = team_id or environ.LUXONISML_TEAM_ID

        if bucket_storage == BucketStorage.LOCAL:
            local_path = base_path / "data" / team_id / "datasets"
            if not local_path.exists():
                return []
            return [d.name for d in local_path.iterdir() if d.is_dir()]

        bucket = bucket or environ.LUXONISML_BUCKET
        if bucket is None:
            raise ValueError("Must set LUXONISML_BUCKET environment variable!")

        fs = LuxonisFileSystem(
            LuxonisDataset._construct_url(bucket_storage, bucket, team_id, "")
        )
        return list(fs.walk_dir("", recursive=False, typ="directory"))

    def _get_dir(
        self,
        remote_path: PathType,
        local_dir: PathType,
        mlflow_instance: Optional[ModuleType] = None,
        default: Optional[PathType] = None,
    ) -> Optional[Path]:
        try:
            return self.fs.get_dir(remote_path, local_dir, mlflow_instance)
        except shutil.SameFileError:
            return Path(local_dir, Path(remote_path).name)
        except Exception:
            return Path(default) if default is not None else None

    def _get_file(
        self,
        remote_path: PathType,
        local_path: PathType,
        mlflow_instance: Optional[ModuleType] = None,
        default: Optional[PathType] = None,
    ) -> Optional[Path]:
        try:
            return self.fs.get_file(remote_path, local_path, mlflow_instance)
        except shutil.SameFileError:
            return Path(local_path, Path(remote_path).name)
        except Exception:
            return Path(default) if default is not None else None


def _rescale_rle(rle: dict, x: float, y: float, w: float, h: float) -> dict:
    height, width = rle["size"]

    if isinstance(rle["counts"], list):
        rle["counts"] = "".join(map(str, rle["counts"]))

    decoded_mask = mask_utils.decode(rle)  # type: ignore

    cropped_mask = decoded_mask[
        int(y * height) : int((y + h) * height),
        int(x * width) : int((x + w) * width),
    ]

    bbox_height = int(h * height)
    bbox_width = int(w * width)

    norm_mask = cropped_mask.astype(np.uint8)
    encoded_norm_mask = mask_utils.encode(np.asfortranarray(norm_mask))

    return {
        "height": bbox_height,
        "width": bbox_width,
        "counts": encoded_norm_mask["counts"].decode("utf-8")
        if isinstance(encoded_norm_mask["counts"], bytes)
        else encoded_norm_mask["counts"],
    }


def rescale_values(
    bbox: Dict[str, float],
    ann: Union[List, Dict],
    sub_ann_key: Literal["keypoints", "segmentation"],
) -> Optional[
    Union[
        List[Tuple[float, float, int]],
        List[Tuple[float, float]],
        Dict[str, Union[int, List[int]]],
    ]
]:
    """Rescale annotation values based on the bounding box coordinates."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    if sub_ann_key == "keypoints":
        return [
            (
                float(kp[0] * w + x),
                float(kp[1] * h + y),
                int(kp[2]),
            )
            for kp in ann
        ]

    if sub_ann_key == "segmentation":
        assert isinstance(ann, dict)
        if "polylines" in ann:
            return [(poly[0] * w + x, poly[1] * h + y) for poly in ann["polylines"]]

        if "rle" in ann:
            return _rescale_rle(ann["rle"], x, y, w, h)

        raise ValueError(
            "Invalid segmentation format. Must be either 'polylines' or 'rle'"
        )

    return None


def _add_generator_wrapper(generator: DatasetIterator) -> DatasetIterator:
    """Generator wrapper to rescale and reformat annotations for each record in the
    input generator."""

    def create_new_record(
        record: Dict[str, Union[str, Dict]],
        annotation: Dict[str, Union[str, int, float, List, Dict]],
    ) -> Dict[str, Union[str, Dict]]:
        """Create a new record with the updated annotation."""
        return {
            "file": record["file"],
            "annotation": annotation,
        }

    for record in generator:
        if isinstance(record, DatasetRecord):
            yield record
            continue

        ann = record["annotation"]
        if ann["type"] != "detection":
            yield record
            continue

        bbox = ann.get("boundingbox", None)
        for sub_ann_key in ["boundingbox", "segmentation", "keypoints"]:
            if sub_ann_key not in ann:
                continue

            sub_ann = ann[sub_ann_key]
            if sub_ann_key == "boundingbox":
                bbox = sub_ann

            if ann.get("scaled_to_boxes", False):
                sub_ann = rescale_values(bbox, sub_ann, sub_ann_key)  # type: ignore

            task = ann.get("task", "detection")

            new_ann = {
                "type": sub_ann_key,
                "class": ann["class"],
                "task": f"{task}-{sub_ann_key}",
            }

            if sub_ann_key == "boundingbox":
                new_ann.update({"instance_id": ann["instance_id"], **bbox})
            elif sub_ann_key == "segmentation":
                if isinstance(sub_ann, list):
                    new_ann.update({"points": sub_ann, "type": "polyline"})
                elif isinstance(sub_ann, dict) and "counts" in sub_ann:
                    new_ann.update(
                        {
                            "height": sub_ann["height"],
                            "width": sub_ann["width"],
                            "counts": sub_ann["counts"],
                            "type": "rle",
                        }
                    )
            elif sub_ann_key == "keypoints":
                new_ann.update(
                    {"instance_id": ann["instance_id"], "keypoints": sub_ann}
                )

            yield create_new_record(record, new_ann)
