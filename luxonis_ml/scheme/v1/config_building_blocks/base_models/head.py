from abc import ABC
from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

from ..enums import ObjectDetectionSubtypeYOLO
from .custom_base_model import CustomBaseModel


class HeadMetadata(BaseModel, ABC):
    """Head metadata parent class.

    @type family: str
    @ivar family: Decoding family.
    @type classes: list
    @ivar classes: Names of object classes recognized by the model.
    @type n_classes: int
    @ivar n_classes: Number of object classes recognized by the model.
    """

    family: str
    classes: List[str]
    n_classes: int


class HeadMetadataClassification(HeadMetadata):
    """Metadata for classification head.

    @type family: str
    @ivar family: Decoding family.
    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed.
    """

    family: str = Field("Classification", Literal=True)
    is_softmax: bool

    @validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "Classification":
            raise ValueError("Invalid family")
        return value


class HeadMetadataObjectDetection(HeadMetadata, ABC):
    """Metadata for object detection head.

    @type stride: int
    @ivar stride: Step size at which the filter (or kernel) moves across the input data
        during convolution.
    @type iou_threshold: float
    @ivar iou_threshold: Non-max supression threshold limiting boxes intersection.
    @type conf_threshold: float
    @ivar conf_threshold: Confidence score threshold above which a detected object is
        considered valid.
    @type max_det: int
    @ivar max_det: Maximum detections per image.
    """

    stride: int
    iou_threshold: float
    conf_threshold: float
    max_det: int


class HeadMetadataObjectDetectionYOLO(HeadMetadataObjectDetection):
    """Metadata for YOLO object detection head.

    @type family: str
    @ivar family: Decoding family.
    @type subtype: ObjectDetectionSubtypeYOLO
    @ivar subtype: YOLO family decoding subtype (e.g. v5, v6, v7 etc.).
    @type n_keypoints: int
    @ivar n_keypoints: Number of keypoints per bbox if provided.
    @type n_prototypes: int
    @ivar n_prototypes: Number of prototypes per bbox if provided.
    @type prototype_output_name: str
    @ivar prototype_output_name: Output node containing prototype information.
    """

    family: str = Field("ObjectDetectionYOLO", Literal=True)
    subtype: ObjectDetectionSubtypeYOLO
    n_keypoints: Optional[int] = None
    n_prototypes: Optional[int] = None
    prototype_output_name: Optional[str] = None

    @validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "ObjectDetectionYOLO":
            raise ValueError("Invalid family")
        return value


class HeadMetadataObjectDetectionSSD(HeadMetadataObjectDetection):
    """Metadata for SSD object detection head.

    @type family: str
    @ivar family: Decoding family.
    @type anchors: list
    @ivar anchors: Predefined bounding boxes of different sizes and aspect ratios.
    """

    family: str = Field("ObjectDetectionSSD", Literal=True)
    anchors: Optional[List[List[int]]] = None

    @validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "ObjectDetectionSSD":
            raise ValueError("Invalid family")
        return value


class HeadMetadataSegmentation(HeadMetadata):
    """Metadata for segmentation head.

    @type family: str
    @ivar family: Decoding family.
    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed.
    """

    family: str = Field("Segmentation", Literal=True)
    is_softmax: bool

    @validator("family")
    def validate_label_type(
        cls,
        value,
    ):
        if value != "Segmentation":
            raise ValueError("Invalid family")
        return value


class HeadMetadataKeypointDetection(HeadMetadata):
    """Metadata for keypoint detection head."""

    def __init__(self):
        raise NotImplementedError


class Head(CustomBaseModel):
    """Represents head of a model.

    @type head_id: str
    @ivar head_id: Unique head identifier.
    @type metadata: HeadMetadata
    @ivar metadata: Parameters required by head to run postprocessing.
    """

    head_id: str
    metadata: Union[
        HeadMetadataObjectDetectionYOLO,
        HeadMetadataObjectDetectionSSD,
        HeadMetadataSegmentation,
        HeadMetadataClassification,
        # HeadMetadataKeypointDetection, # TODO
    ]
