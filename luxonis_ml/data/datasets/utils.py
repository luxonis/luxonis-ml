import shutil
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import polars as pl
from pycocotools import mask as mask_utils
from typing_extensions import overload

from luxonis_ml.utils.filesystem import LuxonisFileSystem, ModuleType, PathType

from .annotation import DatasetRecord
from .base_dataset import DatasetIterator


@overload
def get_file(
    fs: LuxonisFileSystem,
    remote_path: PathType,
    local_path: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    default: Literal[None] = ...,
) -> Optional[Path]:
    pass


@overload
def get_file(
    fs: LuxonisFileSystem,
    remote_path: PathType,
    local_path: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    default: PathType = ...,
) -> Path:
    pass


def get_file(
    fs: LuxonisFileSystem,
    remote_path: PathType,
    local_path: PathType,
    mlflow_instance: Optional[ModuleType] = None,
    default: Optional[PathType] = None,
) -> Optional[Path]:
    try:
        return fs.get_file(remote_path, local_path, mlflow_instance)
    except shutil.SameFileError:
        return Path(local_path, Path(remote_path).name)
    except Exception:
        return Path(default) if default is not None else None


@overload
def find_filepath_uuid(
    filepath: Path,
    index: Optional[pl.DataFrame],
    *,
    raise_on_missing: Literal[False] = ...,
) -> Optional[str]:
    pass


@overload
def find_filepath_uuid(
    filepath: Path,
    index: Optional[pl.DataFrame],
    *,
    raise_on_missing: Literal[True] = ...,
) -> str:
    pass


def find_filepath_uuid(
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


@overload
def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PathType,
    local_dir: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    *,
    default: Literal[None] = None,
) -> Optional[Path]:
    pass


@overload
def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PathType,
    local_dir: PathType,
    mlflow_instance: Optional[ModuleType] = ...,
    *,
    default: Path = ...,
) -> Path:
    pass


def get_dir(
    fs: LuxonisFileSystem,
    remote_path: PathType,
    local_dir: PathType,
    mlflow_instance: Optional[ModuleType] = None,
    *,
    default: Optional[PathType] = None,
) -> Optional[Path]:
    try:
        return fs.get_dir(remote_path, local_dir, mlflow_instance)
    except shutil.SameFileError:
        return Path(local_dir, Path(remote_path).name)
    except Exception:
        return Path(default) if default is not None else None


def _rescale_mask(
    mask: np.ndarray, mask_w: int, mask_h: int, x: float, y: float, w: float, h: float
) -> np.ndarray:
    return mask[
        int(y * mask_h) : int((y + h) * mask_h),
        int(x * mask_w) : int((x + w) * mask_w),
    ].astype(np.uint8)


def _rescale_rle(rle: dict, x: float, y: float, w: float, h: float) -> dict:
    height, width = rle["size"]

    if isinstance(rle["counts"], list):
        rle["counts"] = "".join(map(str, rle["counts"]))

    decoded_mask = mask_utils.decode(rle)  # type: ignore

    cropped_mask = _rescale_mask(decoded_mask, width, height, x, y, w, h)

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
        np.ndarray,
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

        if "mask" in ann:
            mask = ann["mask"]
            width, height = mask.shape
            return _rescale_mask(ann["mask"], width, height, x, y, w, h)

        raise ValueError(
            "Invalid segmentation format. Must be either 'polylines', 'rle', or 'mask'"
        )

    return None


def add_generator_wrapper(generator: DatasetIterator) -> DatasetIterator:
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
