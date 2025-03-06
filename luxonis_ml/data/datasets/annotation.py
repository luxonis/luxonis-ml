import json
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import cv2
import numpy as np
import pycocotools.mask
from loguru import logger
from PIL import Image, ImageDraw
from pydantic import (
    Field,
    GetCoreSchemaHandler,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.types import FilePath, PositiveInt
from pydantic_core import core_schema
from typing_extensions import Annotated, Self, TypeAlias, override

from luxonis_ml.data.utils.parquet import ParquetRecord
from luxonis_ml.typing import check_type
from luxonis_ml.utils import BaseModelExtraForbid

KeypointVisibility: TypeAlias = Literal[0, 1, 2]
NormalizedFloat: TypeAlias = Annotated[float, Field(ge=0, le=1)]


class Category(str):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(cls)


class Detection(BaseModelExtraForbid):
    class_name: Optional[str] = Field(None, alias="class")
    instance_id: int = -1

    metadata: Dict[str, Union[int, float, str, Category]] = {}

    boundingbox: Optional["BBoxAnnotation"] = None
    keypoints: Optional["KeypointAnnotation"] = None
    instance_segmentation: Optional["InstanceSegmentationAnnotation"] = None
    segmentation: Optional["SegmentationAnnotation"] = None
    array: Optional["ArrayAnnotation"] = None

    scale_to_boxes: bool = False

    sub_detections: Dict[str, "Detection"] = {}

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        for name in self.sub_detections:
            check_valid_identifier(name, label="Sub-detection name")
        for key in self.metadata:
            check_valid_identifier(key, label="Metadata key")
        return self

    @model_validator(mode="after")
    def rescale_values(self) -> Self:
        if not self.scale_to_boxes:
            return self
        if self.boundingbox is None:
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

    def get_task_types(self) -> Set[str]:
        task_types = {
            task_type
            for task_type in [
                "boundingbox",
                "keypoints",
                "segmentation",
                "instance_segmentation",
                "array",
            ]
            if getattr(self, task_type) is not None
        }
        if self.class_name is not None:
            task_types.add("classification")
        for metadata_key in self.metadata:
            task_types.add(f"metadata/{metadata_key}")

        return task_types


class Annotation(ABC, BaseModelExtraForbid):
    """Base class for an annotation."""

    @staticmethod
    @abstractmethod
    def combine_to_numpy(
        annotations: List["Annotation"], classes: List[int], n_classes: int
    ) -> np.ndarray: ...


class ClassificationAnnotation(Annotation):
    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["ClassificationAnnotation"],
        classes: List[int],
        n_classes: int,
    ) -> np.ndarray:
        classify_vector = np.zeros(n_classes)
        for i in range(len(annotations)):
            classify_vector[classes[i]] = 1
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
        annotations: List["BBoxAnnotation"],
        classes: List[int],
        n_classes: int = ...,
    ) -> np.ndarray:
        boxes = np.empty((len(annotations), 5))
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

        return cls._clip_sum(values)

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

    def to_numpy(self) -> np.ndarray:
        return np.array(self.keypoints).reshape((-1, 3)).flatten()

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["KeypointAnnotation"],
        classes: List[int] = ...,
        n_classes: int = ...,
    ) -> np.ndarray:
        keypoints = np.empty(
            (len(annotations), len(annotations[0].keypoints) * 3)
        )
        for i, ann in enumerate(annotations):
            keypoints[i] = ann.to_numpy()
        return keypoints

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "keypoints" not in values:
            return values

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
    counts: bytes

    def to_numpy(self) -> np.ndarray:
        with warnings.catch_warnings(record=True):
            return pycocotools.mask.decode(
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

        assigned_pixels = np.zeros((height, width), dtype=bool)
        for i, class_id in enumerate(classes):
            mask = masks[i] & (assigned_pixels == 0)
            segmentation[class_id, ...] = np.maximum(
                segmentation[class_id, ...], mask
            )
            assigned_pixels |= mask.astype(bool)

        return segmentation

    @field_serializer("counts", when_used="json")
    def serialize_counts(self, counts: bytes) -> str:
        return counts.decode("utf-8")

    @model_validator(mode="before")
    @classmethod
    def validate_rle(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if {"counts", "width", "height"} - set(values.keys()):
            return values

        height = values["height"]
        width = values["width"]

        if not check_type(height, int) or not check_type(width, int):
            raise ValueError("Height and width must be integers")

        counts = values["counts"]
        if isinstance(counts, str):
            values["counts"] = counts.encode("utf-8")

        elif isinstance(counts, list):
            for c in counts:
                if not isinstance(c, int) or c < 0:
                    raise ValueError(
                        "RLE counts must be a list of positive integers"
                    )

            with warnings.catch_warnings(record=True):
                rle: Any = pycocotools.mask.frPyObjects(
                    {"counts": counts, "size": [height, width]},  # type: ignore
                    height,
                    width,
                )
            values["counts"] = rle["counts"]
            values["height"] = rle["size"][0]
            values["width"] = rle["size"][1]

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_mask(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "mask" not in values:
            return values
        values = deepcopy(values)

        mask = values.pop("mask")
        if isinstance(mask, (str, Path)):
            mask_path = Path(mask)
            if mask_path.suffix == ".npy":
                try:
                    mask = np.load(mask_path)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load mask from array at '{mask_path}'"
                    ) from e
            elif mask_path.suffix == ".png":
                try:
                    mask = (
                        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        .astype(bool)
                        .astype(np.uint8)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to load mask from image at '{mask_path}'"
                    ) from e
            else:
                raise ValueError(
                    f"Unsupported mask format: {mask_path.suffix}. "
                    "Supported formats are .npy and .png"
                )
        if not isinstance(mask, np.ndarray):
            raise TypeError(
                "Mask must be either a numpy array, "
                "or a path to a saved numpy array"
            )

        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D binary array")

        mask = np.asfortranarray(mask.astype(np.uint8))
        with warnings.catch_warnings(record=True):
            rle = pycocotools.mask.encode(mask)

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

        values = deepcopy(values)

        width = values.pop("width")
        height = values.pop("height")
        if not check_type(height, int) or not check_type(width, int):
            raise ValueError("Height and width must be integers")

        points = values.pop("points")
        if not check_type(points, List[Tuple[float, float]]):
            raise ValueError("Polyline must be a list of float 2D points")

        if len(points) < 3:
            raise ValueError("Polyline must contain at least 3 points")

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
        annotations: List["InstanceSegmentationAnnotation"],
        classes: List[int] = ...,
        n_classes: int = ...,
    ) -> np.ndarray:
        return np.stack([ann.to_numpy() for ann in annotations])


class ArrayAnnotation(Annotation):
    """A custom unspecified annotation that is an arbitrary numpy array.

    All instances of this annotation must have the same shape.

    @type path: FilePath
    @ivar path: The path to the numpy array saved as a C{.npy} file.
    """

    path: FilePath

    def to_numpy(self) -> np.ndarray:
        return np.load(self.path)

    @staticmethod
    @override
    def combine_to_numpy(
        annotations: List["ArrayAnnotation"],
        classes: List[int],
        n_classes: int,
    ) -> np.ndarray:
        out_arr = np.zeros(
            (len(annotations), n_classes, *np.load(annotations[0].path).shape)
        )
        for i, ann in enumerate(annotations):
            out_arr[i, classes[i]] = ann.to_numpy()
        return out_arr

    @field_serializer("path", when_used="json")
    def serialize_path(self, value: FilePath) -> str:
        return str(value)

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: FilePath) -> FilePath:
        if path.suffix != ".npy":
            raise ValueError(
                f"Array annotation file must be a .npy file. Got {path}"
            )
        try:
            np.load(path)
        except Exception as e:
            raise ValueError(
                f"Failed to load array annotation from {path}."
            ) from e
        return path


class DatasetRecord(BaseModelExtraForbid):
    files: Dict[str, FilePath]
    annotation: Optional[Detection] = None
    task_name: str = ""

    @property
    def file(self) -> FilePath:
        if len(self.files) != 1:
            raise ValueError("DatasetRecord must have exactly one file")
        return next(iter(self.files.values()))

    @model_validator(mode="after")
    def validate_task_name_valid_identifier(self) -> Self:
        check_valid_identifier(self.task_name, label="Task name")
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_task_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "task" in values:
            warnings.warn(
                "The 'task' field is deprecated. Use 'task_name' instead.",
                stacklevel=2,
            )
            values["task_name"] = values.pop("task")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_files(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values = deepcopy(values)
        if "file" in values:
            values["files"] = {"image": values.pop("file")}
        return values

    def to_parquet_rows(self) -> Iterable[ParquetRecord]:
        """Converts an annotation to a dictionary for writing to a
        parquet file.

        @rtype: L{ParquetDict}
        @return: A dictionary of annotation data.
        """
        yield from self._to_parquet_rows(self.annotation, self.task_name)

    def _to_parquet_rows(
        self, annotation: Optional[Detection], task_name: str
    ) -> Iterable[ParquetRecord]:
        """Converts an annotation to a dictionary for writing to a
        parquet file.

        @rtype: L{ParquetDict}
        @return: A dictionary of annotation data.
        """
        for source, file_path in self.files.items():
            if annotation is None:
                yield {
                    "file": str(file_path),
                    "source_name": source,
                    "task_name": task_name,
                    "class_name": None,
                    "instance_id": None,
                    "task_type": None,
                    "annotation": None,
                }
            else:
                for task_type in [
                    "boundingbox",
                    "keypoints",
                    "segmentation",
                    "instance_segmentation",
                    "array",
                ]:
                    label: Optional[Annotation] = getattr(
                        annotation, task_type
                    )

                    if label is not None:
                        yield {
                            "file": str(file_path),
                            "source_name": source,
                            "task_name": task_name,
                            "class_name": annotation.class_name,
                            "instance_id": annotation.instance_id,
                            "task_type": task_type,
                            "annotation": label.model_dump_json(),
                        }
                for key, data in annotation.metadata.items():
                    yield {
                        "file": str(file_path),
                        "source_name": source,
                        "task_name": task_name,
                        "class_name": annotation.class_name,
                        "instance_id": annotation.instance_id,
                        "task_type": f"metadata/{key}",
                        "annotation": json.dumps(data),
                    }
                if annotation.class_name is not None:
                    yield {
                        "file": str(file_path),
                        "source_name": source,
                        "task_name": task_name,
                        "class_name": annotation.class_name,
                        "instance_id": annotation.instance_id,
                        "task_type": "classification",
                        "annotation": "{}",
                    }
                for name, detection in annotation.sub_detections.items():
                    yield from self._to_parquet_rows(
                        detection, f"{task_name}/{name}"
                    )


def check_valid_identifier(name: str, *, label: str) -> None:
    """Check if a name is a valid Python identifier after converting
    dashes to underscores.

    Albumentations requires that the names of the targets
    passed as `additional_targets` are valid Python identifiers.
    """
    name = name.replace("-", "_")
    if name and not name.isidentifier():
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
