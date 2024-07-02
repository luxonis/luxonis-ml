import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw
from pydantic import Field, model_validator
from pydantic.types import FilePath, PositiveInt
from typing_extensions import Annotated, TypeAlias

from luxonis_ml.utils import BaseModelExtraForbid, Registry

from ..utils.enums import LabelType

DATASETS_REGISTRY = Registry(name="datasets")


KeypointVisibility: TypeAlias = Literal[0, 1, 2]
NormalizedFloat: TypeAlias = Annotated[float, Field(ge=0, le=1)]
"""C{NormalizedFloat} is a float that is restricted to the range [0, 1]."""

ParquetDict = TypedDict(
    "ParquetDict",
    {
        "file": str,
        "type": str,
        "created_at": datetime,
        "class": str,
        "instance_id": int,
        "task": str,
        "annotation": str,
    },
)


def load_annotation(name: str, js: str, data: Dict[str, Any]) -> "Annotation":
    return {
        "ClassificationAnnotation": ClassificationAnnotation,
        "BBoxAnnotation": BBoxAnnotation,
        "KeypointAnnotation": KeypointAnnotation,
        "RLESegmentationAnnotation": RLESegmentationAnnotation,
        "PolylineSegmentationAnnotation": PolylineSegmentationAnnotation,
        "DetectionAnnotation": DetectionAnnotation,
        "ArrayAnnotation": ArrayAnnotation,
        "LabelAnnotation": LabelAnnotation,
    }[name](**json.loads(js), **data)


class Annotation(ABC, BaseModelExtraForbid):
    """Base class for an annotation.

    @type task: str
    @ivar task: The task name. By default it is the string representation of the
        L{LabelType}.
    @type class_: str
    @ivar class_: The class name for the annotation.
    @type instance_id: int
    @ivar instance_id: The instance id of the annotation. This determines the order in
        which individual instances are loaded in L{LuxonisLoader}.
    @type _label_type: ClassVar[L{LabelType}]
    @ivar _label_type: The label type of the annotation.
    """

    _label_type: ClassVar[LabelType]

    task: str
    class_: str = Field("", alias="class")
    instance_id: int = -1

    @model_validator(mode="before")
    @classmethod
    def validate_task(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "task" not in values:
            values = {**values, "task": cls._label_type.value}
        return values

    def get_value(self) -> Dict[str, Any]:
        """Converts the annotation to a dictionary that can be saved to a parquet
        file."""
        return self.dict(exclude={"class_", "class_id", "instance_id", "task", "type_"})

    @staticmethod
    @abstractmethod
    def combine_to_numpy(
        annotations: List["Annotation"],
        class_mapping: Dict[str, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Combines multiple instance annotations into a single numpy array."""
        pass


class ClassificationAnnotation(Annotation):
    _label_type = LabelType.CLASSIFICATION
    type_: Literal["classification"] = Field("classification", alias="type")

    @staticmethod
    def combine_to_numpy(
        annotations: List["ClassificationAnnotation"],
        class_mapping: Dict[str, int],
        **_,
    ) -> np.ndarray:
        classify_vector = np.zeros(len(class_mapping))
        for ann in annotations:
            class_ = class_mapping.get(ann.class_, 0)
            classify_vector[class_] = 1
        return classify_vector


class BBoxAnnotation(Annotation):
    """Bounding box annotation.

    Values are normalized based on the image size.

    @type x: float
    @ivar x: The center x-coordinate of the bounding box. Normalized to [0, 1].
    @type y: float
    @ivar y: The center y-coordinate of the bounding box. Normalized to [0, 1].
    @type w: float
    @ivar w: The width of the bounding box. Normalized to [0, 1].
    @type h: float
    @ivar h: The height of the bounding box. Normalized to [0, 1].
    """

    type_: Literal["boundingbox"] = Field("boundingbox", alias="type")

    x: NormalizedFloat
    y: NormalizedFloat
    w: NormalizedFloat
    h: NormalizedFloat

    _label_type = LabelType.BOUNDINGBOX

    def to_numpy(self, class_mapping: Dict[str, int]) -> np.ndarray:
        class_ = class_mapping.get(self.class_, 0)
        return np.array([class_, self.x, self.y, self.w, self.h])

    @staticmethod
    def combine_to_numpy(
        annotations: List["BBoxAnnotation"], class_mapping: Dict[str, int], **_
    ) -> np.ndarray:
        boxes = np.zeros((len(annotations), 5))
        for i, ann in enumerate(annotations):
            boxes[i] = ann.to_numpy(class_mapping)
        return boxes


class KeypointAnnotation(Annotation):
    """Keypoint annotation.

    Values are normalized to [0, 1] based on the image size.

    @type keypoints: List[Tuple[float, float, L{KeypointVisibility}]]
    @ivar keypoints: List of keypoints. Each keypoint is a tuple of (x, y, visibility).
        x and y are normalized to [0, 1]. visibility is one of {0, 1, 2} where:
            - 0: Not visible / not labeled
            - 1: Occluded
            - 2: Visible
    """

    type_: Literal["keypoints"] = Field("keypoints", alias="type")

    keypoints: List[Tuple[NormalizedFloat, NormalizedFloat, KeypointVisibility]]

    _label_type = LabelType.KEYPOINTS

    def to_numpy(self, class_mapping: Dict[str, int]) -> np.ndarray:
        class_ = class_mapping.get(self.class_, 0)
        kps = np.array(self.keypoints).reshape((-1, 3)).astype(np.float32)
        return np.concatenate([[class_], kps.flatten()])

    @staticmethod
    def combine_to_numpy(
        annotations: List["KeypointAnnotation"], class_mapping: Dict[str, int], **_
    ) -> np.ndarray:
        keypoints = np.zeros((len(annotations), len(annotations[0].keypoints) * 3 + 1))
        for i, ann in enumerate(annotations):
            keypoints[i] = ann.to_numpy(class_mapping)
        return keypoints


class SegmentationAnnotation(Annotation):
    """Base class for segmentation annotations."""

    _label_type = LabelType.SEGMENTATION


class RLESegmentationAnnotation(SegmentationAnnotation):
    """U{Run-length encoded<https://en.wikipedia.org/wiki/Run-length_encoding>}
        segmentation mask.

    @type height: int
    @ivar height: The height of the segmentation mask.

    @type width: int
    @ivar width: The width of the segmentation mask.

    @type counts: Union[List[int], bytes]
    @ivar counts: The run-length encoded mask.
        This can be a list of integers or a byte string.
    """

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

    def to_numpy(self, _: Dict[str, int], width: int, height: int) -> np.ndarray:
        assert isinstance(self.counts, bytes)
        return mask_util.decode({"counts": self.counts, "size": [height, width]})

    @staticmethod
    def combine_to_numpy(
        annotations: List["RLESegmentationAnnotation"],
        class_mapping: Dict[str, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        seg = np.zeros((len(class_mapping), height, width))
        for ann in annotations:
            class_ = class_mapping.get(ann.class_, 0)
            mask = ann.to_numpy(class_mapping, width, height)
            seg[class_, ...] += mask

        seg = np.clip(seg, 0, 1)
        return seg


class PolylineSegmentationAnnotation(SegmentationAnnotation):
    """Polyline segmentation mask.

    @type points: List[Tuple[float, float]]
    @ivar points: List of points that define the polyline. Each point is a tuple of (x,
        y). x and y are normalized to [0, 1] based on the image size.
    """

    type_: Literal["polyline"] = Field("polyline", alias="type")

    points: List[Tuple[NormalizedFloat, NormalizedFloat]] = Field(min_length=3)

    def to_numpy(self, _: Dict[str, int], width: int, height: int) -> np.ndarray:
        polyline = [(round(x * width), round(y * height)) for x, y in self.points]
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polyline, fill=1, outline=1)
        return np.array(mask)

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
            seg[class_, ...] += ann.to_numpy(class_mapping, width, height)

        seg = np.clip(seg, 0, 1)
        return seg


class DetectionAnnotation(Annotation):
    """Detection annotation containing bounding box, keypoints, and segmentation.

    @type bounding_box: Optional[BBoxAnnotation]
    @ivar bounding_box: The bounding box annotation.
    @type keypoints: Optional[KeypointAnnotation]
    @ivar keypoints: The keypoints annotation.
    @type segmentation: Optional[Union[RLESegmentationAnnotation,
        PolylineSegmentationAnnotation]]
    @ivar segmentation: The segmentation annotation.
    """

    type_: Literal["detection"] = Field("detection", alias="type")

    bounding_box: Optional[BBoxAnnotation] = None
    keypoints: Optional[KeypointAnnotation] = None
    segmentation: Optional[
        Union[RLESegmentationAnnotation, PolylineSegmentationAnnotation]
    ] = None

    _label_type = LabelType.DETECTION

    def get_value(self) -> Dict[str, Any]:
        value = {}
        if self.bounding_box:
            value["bounding_box"] = self.bounding_box.get_value()
        if self.keypoints:
            value["keypoints"] = self.keypoints.get_value()
        if self.segmentation:
            value["segmentation"] = self.segmentation.get_value()
        return value

    @staticmethod
    def combine_to_numpy(
        annotations: List["DetectionAnnotation"],
        class_mapping: Dict[str, int],
        height: int,
        width: int,
    ) -> Dict[str, np.ndarray]:
        bbox_arr = np.zeros((len(annotations), 5))
        keypoints_arr = np.zeros(
            (len(annotations), len(annotations[0].keypoints.keypoints) * 3 + 1)
        )
        seg_arr = np.zeros((len(class_mapping), height, width))

        for i, ann in enumerate(annotations):
            if ann.bounding_box:
                bbox_arr[i] = ann.bounding_box.to_numpy(class_mapping)
            if ann.keypoints:
                keypoints_arr[i] = ann.keypoints.to_numpy(class_mapping)
            if ann.segmentation:
                seg_mask = ann.segmentation.to_numpy(class_mapping, width, height)
                class_ = class_mapping.get(ann.class_, 0)
                seg_arr[class_, ...] += seg_mask

        seg_arr = np.clip(seg_arr, 0, 1)

        return {
            "bounding_boxes": bbox_arr,
            "keypoints": keypoints_arr,
            "segmentation": seg_arr,
        }

    @property
    def sub_annotation_count(self) -> int:
        """Returns the number of sub-annotations."""
        count = 0
        if self.bounding_box is not None:
            count += 1
        if self.keypoints is not None:
            count += 1
        if self.segmentation is not None:
            count += 1
        return count


class ArrayAnnotation(Annotation):
    """A custom unspecified annotation that is an arbitrary numpy array.

    All instances of this annotation must have the same shape.

    @type path: L{FilePath}
    @ivar path: The path to the numpy array saved as a C{.npy} file.
    """

    type_: Literal["array"] = Field("array", alias="type")

    path: FilePath

    _label_type = LabelType.ARRAY

    @staticmethod
    def combine_to_numpy(
        annotations: List["ArrayAnnotation"], class_mapping: Dict[str, int], **_
    ) -> np.ndarray:
        out_arr = np.zeros(
            (len(annotations), len(class_mapping), *np.load(annotations[0].path).shape)
        )
        for i, ann in enumerate(annotations):
            class_ = class_mapping.get(ann.class_, 0)
            out_arr[i, class_] = np.load(ann.path)
        return out_arr


class LabelAnnotation(Annotation):
    """A custom unspecified annotation with a single primitive value.

    @type value: Union[bool, int, float, str]
    @ivar value: The value of the annotation.
    """

    type_: Literal["label"] = Field("label", alias="type")

    value: Union[bool, int, float, str]

    _label_type = LabelType.LABEL

    @staticmethod
    def combine_to_numpy(
        annotations: List["LabelAnnotation"], class_mapping: Dict[str, int], **_
    ) -> np.ndarray:
        out_arr = np.zeros((len(annotations), len(class_mapping))).astype(
            type(annotations[0].value)
        )
        for i, ann in enumerate(annotations):
            class_ = class_mapping.get(ann.class_, 0)
            out_arr[i, class_] = ann.value
        return out_arr


class DatasetRecord(BaseModelExtraForbid):
    """A record of an image and its annotation.

    @type file: L{FilePath}
    @ivar file: A path to the image.
    @type annotation: Optional[Annotation]
    @ivar annotation: The annotation for the image.
    """

    file: FilePath
    annotation: Optional[
        Union[
            ClassificationAnnotation,
            BBoxAnnotation,
            KeypointAnnotation,
            RLESegmentationAnnotation,
            PolylineSegmentationAnnotation,
            DetectionAnnotation,
            ArrayAnnotation,
            LabelAnnotation,
        ]
    ] = Field(None, discriminator="type_")

    def to_parquet_dict(self) -> ParquetDict:
        """Converts an annotation to a dictionary for writing to a parquet file.

        @rtype: L{ParquetDict}
        @return: A dictionary of annotation data.
        """

        files, types, created_ats, classes, instance_ids, tasks, annotations = (
            [] for _ in range(7)
        )

        def append_annotation_data(annotation, annotation_type):
            """Helper function to append annotation data."""
            value = annotation.get_value()
            json_value = json.dumps(value)
            files.append(self.file.name)
            types.append(annotation_type.__name__)
            created_ats.append(datetime.utcnow())
            classes.append(self.annotation.class_ or "")
            instance_ids.append(annotation.instance_id or -1)
            tasks.append(annotation.task or "")
            annotations.append(json_value)

        if isinstance(self.annotation, DetectionAnnotation):
            if self.annotation.bounding_box is not None:
                append_annotation_data(
                    self.annotation.bounding_box, type(self.annotation.bounding_box)
                )
            if self.annotation.keypoints is not None:
                append_annotation_data(
                    self.annotation.keypoints, type(self.annotation.keypoints)
                )
            if self.annotation.segmentation is not None:
                append_annotation_data(
                    self.annotation.segmentation, type(self.annotation.segmentation)
                )
        else:
            append_annotation_data(self.annotation, type(self.annotation))

        return {
            "file": files,
            "type": types,
            "created_at": created_ats,
            "class": classes,
            "instance_id": instance_ids,
            "task": tasks,
            "annotation": annotations,
        }
