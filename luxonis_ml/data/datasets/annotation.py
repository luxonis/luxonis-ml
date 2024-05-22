import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw
from pydantic import Field, model_validator
from pydantic.types import FilePath, PositiveInt
from typing_extensions import TypeAlias

from luxonis_ml.utils import BaseModelExtraForbid, Registry

from ..utils.enums import LabelType

DATASETS_REGISTRY = Registry(name="datasets")


KeypointVisibility: TypeAlias = Literal[0, 1, 2]
ParquetDict: TypeAlias = Dict[str, Any]


def load_annotation(name: str, js: str, data: Dict[str, Any]) -> "Annotation":
    return {
        "ClassificationAnnotation": ClassificationAnnotation,
        "BBoxAnnotation": BBoxAnnotation,
        "KeypointAnnotation": KeypointAnnotation,
        "RLESegmentationAnnotation": RLESegmentationAnnotation,
        "PolylineSegmentationAnnotation": PolylineSegmentationAnnotation,
        "ArrayAnnotation": ArrayAnnotation,
        "LabelAnnotation": LabelAnnotation,
    }[name](**json.loads(js), **data)


class Annotation(ABC, BaseModelExtraForbid):
    """Base class for an annotation."""

    _label_type: ClassVar[LabelType]

    task: str
    class_: str = Field("", alias="class")
    # instance_id: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_task(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "task" not in values:
            values = {**values, "task": cls._label_type.value}
        return values

    def get_value(self) -> Dict[str, Any]:
        return self.dict(exclude={"class_", "class_id", "instance_id", "task", "type_"})

    @staticmethod
    @abstractmethod
    def combine_to_numpy(
        annotations: List["Annotation"],
        class_mapping: Dict[str, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        pass


class ClassificationAnnotation(Annotation):
    _label_type = LabelType.CLASSIFICATION
    type_: Literal["classification"] = Field("classification", alias="type")

    @staticmethod
    def combine_to_numpy(
        annotations: List["BBoxAnnotation"], class_mapping: Dict[str, int], **_
    ) -> np.ndarray:
        classify_vector = np.zeros(len(class_mapping))
        for ann in annotations:
            class_ = class_mapping.get(ann.class_, 0)
            classify_vector[class_] = 1
        return classify_vector


class BBoxAnnotation(Annotation):
    type_: Literal["boundingbox"] = Field("boundingbox", alias="type")

    x: float
    y: float
    w: float
    h: float

    _label_type = LabelType.BOUNDINGBOX

    @staticmethod
    def combine_to_numpy(
        annotations: List["BBoxAnnotation"], class_mapping: Dict[str, int], **_
    ) -> np.ndarray:
        boxes = np.zeros((len(annotations), 5))
        for i, ann in enumerate(annotations):
            class_ = class_mapping.get(ann.class_, 0)
            boxes[i] = np.array([class_, ann.x, ann.y, ann.w, ann.h])
        return boxes


class KeypointAnnotation(Annotation):
    type_: Literal["keypoints"] = Field("keypoints", alias="type")

    keypoints: List[Tuple[float, float, KeypointVisibility]]

    _label_type = LabelType.KEYPOINTS

    @staticmethod
    def combine_to_numpy(
        annotations: List["KeypointAnnotation"], class_mapping: Dict[str, int], **_
    ) -> np.ndarray:
        keypoints = np.zeros((len(annotations), len(annotations[0].keypoints) * 3 + 1))
        for i, ann in enumerate(annotations):
            class_ = class_mapping.get(ann.class_, 0)
            kps = np.array(ann.keypoints).reshape((-1, 3)).astype(np.float32)
            keypoints[i] = np.concatenate([[class_], kps.flatten()])
        return keypoints


class SegmentationAnnotation(Annotation):
    _label_type = LabelType.SEGMENTATION


class RLESegmentationAnnotation(SegmentationAnnotation):
    type_: Literal["rle"] = Field("rle", alias="type")

    height: PositiveInt
    width: PositiveInt
    counts: Union[List[PositiveInt], bytes]

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

    @staticmethod
    def combine_to_numpy(
        annotations: List["RLESegmentationAnnotation"],
        class_mapping: Dict[str, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        seg = np.zeros((len(class_mapping), height, width))
        for ann in annotations:
            assert isinstance(ann.counts, bytes)
            class_ = class_mapping.get(ann.class_, 0)
            mask = mask_util.decode({"counts": ann.counts, "size": [height, width]})
            seg[class_, ...] += mask

        seg = np.clip(seg, 0, 1)
        return seg


class PolylineSegmentationAnnotation(SegmentationAnnotation):
    type_: Literal["polyline"] = Field("polyline", alias="type")

    points: List[Tuple[float, float]] = Field(min_length=3)

    @staticmethod
    def combine_to_numpy(
        annotations: List["PolylineSegmentationAnnotation"],
        class_mapping: Dict[str, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        seg = np.zeros((len(class_mapping), height, width))
        for ann in annotations:
            class_ = class_mapping.get(ann.class_, 0)
            polyline = [(round(x * width), round(y * height)) for x, y in ann.points]
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polyline, fill=1, outline=1)
            mask = np.array(mask)
            seg[class_, ...] += mask

        seg = np.clip(seg, 0, 1)
        return seg


class ArrayAnnotation(Annotation):
    type_: Literal["array"] = Field("array", alias="type")

    path: FilePath

    _label_type = LabelType.ARRAY


class LabelAnnotation(Annotation):
    type_: Literal["label"] = Field("label", alias="type")

    value: Union[bool, int, float, str]

    _label_type = LabelType.LABEL


class DatasetRecord(BaseModelExtraForbid):
    file: FilePath
    annotation: Optional[
        Union[
            ClassificationAnnotation,
            BBoxAnnotation,
            KeypointAnnotation,
            RLESegmentationAnnotation,
            PolylineSegmentationAnnotation,
            ArrayAnnotation,
            LabelAnnotation,
        ]
    ] = Field(None, discriminator="type_")

    def to_parquet(self) -> ParquetDict:
        """Converts an annotation to a dictionary for writing to a parquet file.

        @rtype: L{ParquetDict}
        @return: A dictionary of annotation data.
        """

        value = self.annotation.get_value() if self.annotation is not None else {}
        json_value = json.dumps(value)
        return {
            "file": self.file.name,
            "type": self.annotation.__class__.__name__,
            "created_at": datetime.utcnow(),
            "class": self.annotation.class_ or ""
            if self.annotation is not None
            else "",
            # "instance_id": self.annotation.instance_id or -1
            # if self.annotation is not None
            # else -1,
            "task": self.annotation.task if self.annotation is not None else "",
            "annotation": json_value,
        }
