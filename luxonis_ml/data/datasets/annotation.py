import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw
from pydantic import (
    Field,
    field_serializer,
    model_serializer,
    model_validator,
)
from pydantic.types import FilePath, NonNegativeInt, PositiveInt
from typeguard import check_type
from typing_extensions import Annotated, Self, TypeAlias, override

from luxonis_ml.data.utils.parquet import ParquetDetection, ParquetRecord
from luxonis_ml.utils import BaseModelExtraForbid

logger = logging.getLogger(__name__)

KeypointVisibility: TypeAlias = Literal[0, 1, 2]
NormalizedFloat: TypeAlias = Annotated[float, Field(ge=0, le=1)]
"""C{NormalizedFloat} is a float that is restricted to the range [0,
1]."""


class Detection(BaseModelExtraForbid):
    class_name: Optional[str] = Field(None, alias="class")
    instance_id: int = -1

    metadata: Dict[str, Union[int, float, str]] = {}

    boundingbox: Optional["BBoxAnnotation"] = None
    keypoints: Optional["KeypointAnnotation"] = None
    instance_segmentation: Optional["InstanceSegmentationAnnotation"] = None
    segmentation: Optional["SegmentationAnnotation"] = None
    array: Optional["ArrayAnnotation"] = None

    scale_to_boxes: bool = False

    sub_detections: Dict[str, "Detection"] = {}

    def to_parquet_rows(self, prefix: str = "") -> Iterable[ParquetDetection]:
        for task_type in [
            "boundingbox",
            "keypoints",
            "segmentation",
            "instance_segmentation",
            "array",
        ]:
            label: Optional[Annotation] = getattr(self, task_type)

            if label is not None:
                yield {
                    "class_name": self.class_name,
                    "instance_id": self.instance_id,
                    "task_type": f"{prefix}{task_type}",
                    "annotation": label.model_dump_json(),
                }
        for key, data in self.metadata.items():
            yield {
                "class_name": self.class_name,
                "instance_id": self.instance_id,
                "task_type": f"{prefix}metadata/{key}",
                "annotation": json.dumps(data),
            }
        if self.class_name is not None:
            yield {
                "class_name": self.class_name,
                "instance_id": self.instance_id,
                "task_type": f"{prefix}classification",
                "annotation": "{}",
            }
        for name, detection in self.sub_detections.items():
            yield from detection.to_parquet_rows(f"{prefix}{name}/")

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        for name in self.sub_detections:
            check_valid_identifier("Sub-detection name", name)
        for key in self.metadata:
            check_valid_identifier("Metadata key", key)
        return self

    @model_validator(mode="after")
    def rescale_values(self) -> Self:
        if not self.scale_to_boxes:
            return self
        elif self.boundingbox is None:
            raise ValueError(
                "`scaled_to_boxes` is set to True, "
                "but no bounding box is provided."
            )
        x, y, w, h = (
            self.boundingbox.x,
            self.boundingbox.y,
            self.boundingbox.w,
            self.boundingbox.h,
        )

        if self.keypoints is not None:
            self.keypoints = KeypointAnnotation(
                keypoints=[
                    (x + w * kp[0], y + h * kp[1], kp[2])
                    for kp in self.keypoints.keypoints
                ]
            )
        return self


class Annotation(ABC, BaseModelExtraForbid):
    """Base class for an annotation."""

    @staticmethod
    @abstractmethod
    def combine_to_numpy(
        annotations: List["Annotation"],
        classes: List[int],
        n_classes: int,
    ) -> np.ndarray: ...


class ClassificationAnnotation(Annotation):
    @staticmethod
    @override
    def combine_to_numpy(
        _: List["ClassificationAnnotation"],
        classes: List[int],
        n_classes: int,
    ) -> np.ndarray:
        classify_vector = np.zeros(n_classes)
        for class_id in classes:
            classify_vector[class_id] = 1
        return classify_vector


class BBoxAnnotation(Annotation):
    """Bounding box annotation.

    Values are normalized based on the image size.

    @type x: float
    @ivar x: The top-left x coordinate of the bounding box. Normalized
        to M{[0, 1]}.
    @type y: float
    @ivar y: The top-left y coordinate of the bounding box. Normalized
        to M{[0, 1]}.
    @type w: float
    @ivar w: The width of the bounding box. Normalized to M{[0, 1]}.
    @type h: float
    @ivar h: The height of the bounding box. Normalized to M{[0, 1]}.
    """

    x: NormalizedFloat
    y: NormalizedFloat
    w: NormalizedFloat
    h: NormalizedFloat

    def to_numpy(self, class_id: int) -> np.ndarray:
        return np.array([class_id, self.x, self.y, self.w, self.h])

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["BBoxAnnotation"], classes: List[int], _: int
    ) -> np.ndarray:
        boxes = np.zeros((len(annotations), 5))
        for i, ann in enumerate(annotations):
            boxes[i] = ann.to_numpy(classes[i])
        return boxes

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        warn = False
        for key in ["x", "y", "w", "h"]:
            if values[key] < -2 or values[key] > 2:
                raise ValueError(
                    "BBox annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            if not (0 <= values[key] <= 1):
                warn = True
                values[key] = max(0, min(1, values[key]))
        if warn:
            logger.warning(
                "BBox annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )

        values = cls._clip_sum(values)
        return values

    @staticmethod
    def _clip_sum(values: Dict[str, Any]) -> Dict[str, Any]:
        if values["x"] + values["w"] > 1:
            values["w"] = 1 - values["x"]
            logger.warning(
                "BBox annotation has x + width > 1. Clipping width so the sum is 1."
            )
        if values["y"] + values["h"] > 1:
            values["h"] = 1 - values["y"]
            logger.warning(
                "BBox annotation has y + height > 1. Clipping height so the sum is 1."
            )
        return values


class KeypointAnnotation(Annotation):
    """Keypoint annotation.

    Values are normalized to M{[0, 1]} based on the image size.

    @type keypoints: List[Tuple[float, float, L{KeypointVisibility}]]
    @ivar keypoints: List of keypoints. Each keypoint is a tuple of (x, y, visibility).
        x and y are normalized to M{[0, 1]}. visibility is one of M{0}, M{1}, or M{2} where:
            - 0: Not visible / not labeled
            - 1: Occluded
            - 2: Visible
    """

    keypoints: List[
        Tuple[NormalizedFloat, NormalizedFloat, KeypointVisibility]
    ]

    def to_numpy(self, class_id: int) -> np.ndarray:
        return np.array(self.keypoints).reshape((-1, 3)).flatten()

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["KeypointAnnotation"], classes: List[int], _: int
    ) -> np.ndarray:
        keypoints = np.zeros(
            (len(annotations), len(annotations[0].keypoints) * 3)
        )
        for i, ann in enumerate(annotations):
            keypoints[i] = ann.to_numpy(classes[i])
        return keypoints

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        warn = False
        for i, keypoint in enumerate(values["keypoints"]):
            if (keypoint[0] < -2 or keypoint[0] > 2) or (
                keypoint[1] < -2 or keypoint[1] > 2
            ):
                raise ValueError(
                    "Keypoint annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            new_keypoint = list(keypoint)
            if not (0 <= keypoint[0] <= 1):
                new_keypoint[0] = max(0, min(1, keypoint[0]))
                warn = True
            if not (0 <= keypoint[1] <= 1):
                new_keypoint[1] = max(0, min(1, keypoint[1]))
                warn = True
            values["keypoints"][i] = tuple(new_keypoint)

        if warn:
            logger.warning(
                "Keypoint annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )
        return values


class SegmentationAnnotation(Annotation):
    """Run-length encoded segmentation mask.

    @type height: int
    @ivar height: The height of the segmentation mask.

    @type width: int
    @ivar width: The width of the segmentation mask.

    @type counts: Union[List[int], bytes]
    @ivar counts: The run-length encoded mask.
        This can be a list of integers or a byte string.

    @see: U{Run-length encoding<https://en.wikipedia.org/wiki/Run-length_encoding>}
    """

    height: PositiveInt
    width: PositiveInt
    counts: Union[List[NonNegativeInt], bytes]

    def to_numpy(self) -> np.ndarray:
        assert isinstance(self.counts, bytes)
        return mask_util.decode(
            {"counts": self.counts, "size": [self.height, self.width]}
        ).astype(np.uint8)

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["SegmentationAnnotation"],
        classes: List[int],
        n_classes: int,
    ) -> np.ndarray:
        ref = annotations[0]
        width, height = ref.width, ref.height
        masks = np.stack([ann.to_numpy() for ann in annotations])

        segmentation = np.zeros((n_classes, height, width), dtype=np.uint8)

        for i, class_id in enumerate(classes):
            segmentation[class_id, ...] = np.maximum(
                segmentation[class_id, ...], masks[i]
            )

        return segmentation

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        if isinstance(self.counts, bytes):
            return {
                "height": self.height,
                "width": self.width,
                "counts": self.counts.decode("utf-8"),
            }

        rle: Any = mask_util.frPyObjects(
            {
                "counts": self.counts,
                "size": [self.height, self.width],
            },
            self.height,
            self.width,
        )

        return {
            "height": rle["size"][0],
            "width": rle["size"][1],
            "counts": rle["counts"].decode("utf-8"),
        }

    @model_validator(mode="before")
    @classmethod
    def validate_mask(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "mask" not in values:
            return values

        mask = values.pop("mask")
        if isinstance(mask, str):
            mask_path = Path(mask)
            try:
                mask = np.load(mask_path)
            except Exception as e:
                raise ValueError(
                    f"Failed to load mask from {mask_path}"
                ) from e
        if not isinstance(mask, np.ndarray):
            raise ValueError(
                "Mask must be either a numpy array, "
                "or a path to a saved numpy array"
            )

        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D binary array")

        mask = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask)

        return {
            "height": rle["size"][0],
            "width": rle["size"][1],
            "counts": rle["counts"].decode("utf-8"),  # type: ignore
            **values,
        }

    @model_validator(mode="before")
    @classmethod
    def validate_polyline(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if {"points", "width", "height"} - set(values.keys()):
            return values

        points = check_type(values.pop("points"), List[Tuple[float, float]])
        width = check_type(values.pop("width"), int)
        height = check_type(values.pop("height"), int)

        cls._clip_points(points)

        polyline = [(round(x * width), round(y * height)) for x, y in points]
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polyline, fill=1, outline=1)
        return {"mask": np.array(mask).astype(np.uint8), **values}

    @staticmethod
    def _clip_points(points: List[Tuple[float, float]]) -> None:
        warn = False
        for i in range(len(points)):
            x, y = points[i]
            if (x < -2 or x > 2) or (y < -2 or y > 2):
                raise ValueError(
                    "Polyline annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            new_x, new_y = x, y
            if not (0 <= x <= 1):
                new_x = max(0, min(1, x))
                warn = True
            if not (0 <= y <= 1):
                new_y = max(0, min(1, y))
                warn = True

            points[i] = (new_x, new_y)

        if warn:
            logger.warning(
                "Polyline annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )


class InstanceSegmentationAnnotation(SegmentationAnnotation):
    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["SegmentationAnnotation"], classes: List[int], _: int
    ) -> np.ndarray:
        return np.stack(
            [
                ann.to_numpy() * (class_id + 1)
                for ann, class_id in zip(annotations, classes)
            ]
        )


class ArrayAnnotation(Annotation):
    """A custom unspecified annotation that is an arbitrary numpy array.

    All instances of this annotation must have the same shape.

    @type path: FilePath
    @ivar path: The path to the numpy array saved as a C{.npy} file.
    """

    path: FilePath

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["ArrayAnnotation"], classes: List[int], _: int
    ) -> np.ndarray:
        out_arr = np.zeros(
            (
                len(annotations),
                len(classes),
                *np.load(annotations[0].path).shape,
            )
        )
        for i, ann in enumerate(annotations):
            out_arr[i, classes[i]] = np.load(ann.path)
        return out_arr

    @field_serializer("path")
    def serialize_path(self, value: FilePath) -> str:
        return str(value)


class DatasetRecord(BaseModelExtraForbid):
    files: Dict[str, FilePath]
    annotation: Optional[Detection] = None
    task: str

    @property
    def file(self) -> FilePath:
        if len(self.files) != 1:
            raise ValueError("DatasetRecord must have exactly one file")
        return next(iter(self.files.values()))

    @model_validator(mode="after")
    def validate_task_name(self) -> Self:
        check_valid_identifier("Task name", self.task)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_files(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "file" in values:
            values["files"] = {"image": values.pop("file")}
        return values

    @model_validator(mode="before")
    @classmethod
    def auto_populate_task(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "task" not in values:
            annotations = values.get("annotation", {})
            if (
                "segmentation" in annotations
                and "boundingbox" not in annotations
            ):
                values["task"] = "segmentation"
            else:
                values["task"] = "detection"
        return values

    def to_parquet_rows(self) -> Iterable[ParquetRecord]:
        """Converts an annotation to a dictionary for writing to a
        parquet file.

        @rtype: L{ParquetDict}
        @return: A dictionary of annotation data.
        """
        timestamp = datetime.now(timezone.utc)
        for source, file_path in self.files.items():
            if self.annotation is not None:
                for detection in self.annotation.to_parquet_rows():
                    yield {
                        "file": str(file_path),
                        "source_name": source,
                        "task_name": self.task,
                        "created_at": timestamp,
                        **detection,
                    }
            else:
                yield {
                    "file": str(file_path),
                    "source_name": source,
                    "task_name": self.task,
                    "created_at": timestamp,
                    "class_name": None,
                    "instance_id": None,
                    "task_type": None,
                    "annotation": None,
                }


def check_valid_identifier(label: str, name: str) -> None:
    """Check if a name is a valid Python identifier after converting
    dashes to underscores.

    Albumentations requires that the names of the targets
    passed as `additional_targets` are valid Python identifiers.
    """
    if not name.replace("-", "_").isidentifier():
        raise ValueError(
            f"{label} can only contain alphanumeric characters, "
            "underscores, and dashes. Additionaly, the first character "
            f"must be a letter or underscore. Got {name}"
        )


def load_annotation(task_type: str, data: Dict[str, Any]) -> "Annotation":
    classes = {
        "classification": ClassificationAnnotation,
        "boundingbox": BBoxAnnotation,
        "keypoints": KeypointAnnotation,
        "segmentation": SegmentationAnnotation,
        "instance_segmentation": InstanceSegmentationAnnotation,
        "array": ArrayAnnotation,
    }
    if task_type not in classes:
        raise ValueError(f"Unknown label type: {task_type}")
    return classes[task_type](**data)
