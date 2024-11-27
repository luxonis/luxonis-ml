import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw
from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.types import FilePath, NonNegativeInt, PositiveInt
from typing_extensions import Annotated, TypeAlias

from luxonis_ml.utils import BaseModelExtraForbid

from ..utils.enums import LabelType

logger = logging.getLogger(__name__)

KeypointVisibility: TypeAlias = Literal[0, 1, 2]
NormalizedFloat: TypeAlias = Annotated[float, Field(ge=0, le=1)]
"""C{NormalizedFloat} is a float that is restricted to the range [0,
1]."""

AnnotationName: TypeAlias = Literal[
    "boundingbox", "keypoints", "segmentation", "array"
]


def load_annotation(name: str, data: Dict[str, Any]) -> "Annotation":
    return {
        "BBoxAnnotation": BBoxAnnotation,
        "KeypointAnnotation": KeypointAnnotation,
        "RLESegmentationAnnotation": RLESegmentationAnnotation,
        "PolylineSegmentationAnnotation": PolylineSegmentationAnnotation,
        "MaskSegmentationAnnotation": MaskSegmentationAnnotation,
        "ArrayAnnotation": ArrayAnnotation,
    }[name](**data)


class InstanceLabel(BaseModelExtraForbid):
    class_name: Optional[str] = None
    instance_id: Optional[int] = None
    metadata: Dict[str, Union[int, float, str]] = {}
    task_name: str

    boundingbox: Optional["BBoxAnnotation"] = None
    keypoints: Optional["KeypointAnnotation"] = None
    segmentation: Optional["SegmentationAnnotation"] = None
    array: Optional["ArrayAnnotation"] = None

    @model_validator(mode="before")
    def deserialize_annotations(
        self, values: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "annotation" not in values:
            return values
        for name, value in json.loads(values.pop("annotation")).items():
            values[name] = load_annotation(name, value)
        return values

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        model = {
            "class": self.class_name,
            "instance_id": self.instance_id,
            "metadata": self.metadata,
            "task": self.task_name,
        }
        for key in ["boundingbox", "keypoints", "segmentation", "array"]:
            model[key] = None
            if getattr(self, key, None) is not None:
                model[key] = getattr(self, key).model_dump()

        return model

    def to_numpy(self) -> Dict[AnnotationName, np.ndarray]:
        arrays = {}
        for key in ["boundingbox", "keypoints", "segmentation", "array"]:
            if getattr(self, key, None) is not None:
                arrays[key] = getattr(self, key).to_numpy(self.class_name)
        return arrays


class Annotation(ABC, BaseModelExtraForbid):
    """Base class for an annotation.

    @type task: str
    @ivar task: The task name. By default it is the string
        representation of the L{LabelType}.
    @type class_: str
    @ivar class_: The class name for the annotation.
    @type instance_id: int
    @ivar instance_id: The instance id of the annotation. This
        determines the order in which individual instances are loaded in
        L{luxonis_ml.data.LuxonisLoader}.
    @type _label_type: ClassVar[LabelType]
    @ivar _label_type: The label type of the annotation.
    """

    task: str

    @abstractmethod
    def to_numpy(self, class_id: int) -> np.ndarray: ...


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

    type_: Literal["boundingbox"] = Field("boundingbox", alias="type")

    x: NormalizedFloat
    y: NormalizedFloat
    w: NormalizedFloat
    h: NormalizedFloat

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

        # cliping done in function instead of separate model validator so
        # order of execution is explicitly defined
        values = cls.clip_sum(values)
        return values

    @classmethod
    def clip_sum(cls, values: Dict[str, Any]) -> Dict[str, Any]:
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

    def to_numpy(self, class_id: int) -> np.ndarray:
        return np.array([class_id, self.x, self.y, self.w, self.h])

    @staticmethod
    def combine_to_numpy(
        annotations: List["BBoxAnnotation"], classes: List[int]
    ) -> np.ndarray:
        boxes = np.zeros((len(annotations), 5))
        for i, ann in enumerate(annotations):
            boxes[i] = ann.to_numpy(classes[i])
        return boxes


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

    type_: Literal["keypoints"] = Field("keypoints", alias="type")

    keypoints: List[
        Tuple[NormalizedFloat, NormalizedFloat, KeypointVisibility]
    ]

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

    def to_numpy(self, class_id: int) -> np.ndarray:
        kps = np.array(self.keypoints).reshape((-1, 3)).astype(np.float32)
        return np.concatenate([[class_id], kps.flatten()])

    @staticmethod
    def combine_to_numpy(
        annotations: List["KeypointAnnotation"],
        classes: List[int],
    ) -> np.ndarray:
        keypoints = np.zeros(
            (len(annotations), len(annotations[0].keypoints) * 3 + 1)
        )
        for i, ann in enumerate(annotations):
            keypoints[i] = ann.to_numpy(classes[i])
        return keypoints


class SegmentationAnnotation(Annotation):
    """Base class for segmentation annotations."""

    _label_type = LabelType.SEGMENTATION

    @staticmethod
    def combine_to_numpy(
        annotations: List["SegmentationAnnotation"],
        classes: List[int],
    ) -> np.ndarray:
        width, height = annotations[0].get_size()
        seg = np.zeros((len(classes), height, width), dtype=np.uint8)

        masks = np.stack(
            [
                ann.to_numpy(class_id)
                for class_id, ann in zip(classes, annotations)
            ]
        )

        for i, class_id in enumerate(classes):
            seg[class_id, ...] = np.maximum(
                seg[class_id, ...], masks[i].astype(np.uint8)
            )

        return seg

    @abstractmethod
    def get_size(self) -> Tuple[int, int]: ...


class RLESegmentationAnnotation(SegmentationAnnotation):
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

    type_: Literal["rle"] = Field("rle", alias="type")

    height: PositiveInt
    width: PositiveInt
    counts: Union[List[NonNegativeInt], bytes]

    def get_value(self) -> Dict[str, Any]:
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

    def to_numpy(self, _: int) -> np.ndarray:
        assert isinstance(self.counts, bytes)
        return mask_util.decode(
            {"counts": self.counts, "size": [self.height, self.width]}
        ).astype(np.bool_)

    def get_size(self) -> Tuple[int, int]:
        return self.width, self.height


class MaskSegmentationAnnotation(SegmentationAnnotation):
    """Pixel-wise binary segmentation mask.

    @type mask: npt.NDArray[np.bool_]
    @ivar mask: The segmentation mask as a numpy array. The mask must be
        2D and must be castable to a boolean array.
    """

    type_: Literal["mask"] = Field("mask", alias="type")
    mask: np.ndarray

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _convert_rle(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "mask" in values:
            return values

        if (
            "width" not in values
            or "height" not in values
            or "counts" not in values
        ):
            raise ValueError(
                "MaskSegmentationAnnotation must have either "
                "'mask' or 'width', 'height', and 'counts'"
            )

        width: int = values.pop("width")
        height: int = values.pop("height")
        counts: str = values.pop("counts")

        values["mask"] = mask_util.decode(
            {
                "counts": counts.encode("utf-8"),
                "size": [height, width],
            }
        ).astype(np.bool_)
        return values

    @field_validator("mask", mode="after")
    @staticmethod
    def _validate_shape(mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        return mask

    @field_validator("mask", mode="before")
    @staticmethod
    def _validate_mask(mask: Any) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise ValueError("Mask must be a numpy array")
        return mask.astype(np.bool_)

    def get_value(self) -> Dict[str, Any]:
        mask = np.asfortranarray(self.mask.astype(np.uint8))
        rle = mask_util.encode(mask)

        return {
            "height": rle["size"][0],
            "width": rle["size"][1],
            "counts": rle["counts"].decode("utf-8"),  # type: ignore
        }

    def to_numpy(self, _: int) -> np.ndarray:
        return self.mask

    def get_size(self) -> Tuple[int, int]:
        w, h = self.mask.shape
        return w, h


class PolylineSegmentationAnnotation(SegmentationAnnotation):
    """Polyline segmentation mask.

    @type points: List[Tuple[float, float]]
    @ivar points: List of points that define the polyline. Each point is
        a tuple of (x, y). x and y are normalized to M{[0, 1]} based on
        the image size.
    """

    type_: Literal["polyline"] = Field("polyline", alias="type")

    points: List[Tuple[NormalizedFloat, NormalizedFloat]] = Field(min_length=3)
    width: PositiveInt
    height: PositiveInt

    @model_validator(mode="before")
    @classmethod
    def validate_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        warn = False
        for i, point in enumerate(values["points"]):
            if (point[0] < -2 or point[0] > 2) or (
                point[1] < -2 or point[1] > 2
            ):
                raise ValueError(
                    "Polyline annotation has value outside of automatic clipping range ([-2, 2]). "
                    "Values should be normalized based on image size to range [0, 1]."
                )
            new_point = list(point)
            if not (0 <= point[0] <= 1):
                new_point[0] = max(0, min(1, point[0]))
                warn = True
            if not (0 <= point[1] <= 1):
                new_point[1] = max(0, min(1, point[1]))
                warn = True
            values["points"][i] = tuple(new_point)

        if warn:
            logger.warning(
                "Polyline annotation has values outside of [0, 1] range. Clipping them to [0, 1]."
            )
        return values

    def to_numpy(self, _: int) -> np.ndarray:
        polyline = [
            (round(x * self.width), round(y * self.height))
            for x, y in self.points
        ]
        mask = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polyline, fill=1, outline=1)
        return np.array(mask).astype(np.bool_)

    def get_size(self) -> Tuple[int, int]:
        return self.width, self.height


class ArrayAnnotation(Annotation):
    """A custom unspecified annotation that is an arbitrary numpy array.

    All instances of this annotation must have the same shape.

    @type path: FilePath
    @ivar path: The path to the numpy array saved as a C{.npy} file.
    """

    type_: Literal["array"] = Field("array", alias="type")

    path: FilePath

    _label_type = LabelType.ARRAY

    @staticmethod
    def combine_to_numpy(
        annotations: List["ArrayAnnotation"], classes: List[int]
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
    label: InstanceLabel

    @property
    def file(self) -> FilePath:
        if len(self.files) != 1:
            raise ValueError("DatasetRecord must have exactly one file")
        return next(iter(self.files.values()))

    @model_validator(mode="before")
    @classmethod
    def validate_files(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "file" in values:
            values["files"] = {"image": values.pop("file")}
        return values

    def to_parquet_dict(self) -> Dict[str, Any]:
        """Converts an annotation to a dictionary for writing to a
        parquet file.

        @rtype: L{ParquetDict}
        @return: A dictionary of annotation data.
        """

        label_json = self.label.model_dump_json()
        return {
            "file": self.file.name,
            "created_at": datetime.now(timezone.utc),
            "class": self.label.class_name,
            "instance_id": self.label.instance_id,
            "task": self.label.task_name,
            "annotation": label_json,
        }
