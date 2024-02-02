from abc import ABC
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

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

    family: str = Field(description="Decoding family.")
    classes: List[str] = Field(
        description="Names of object classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of object classes recognized by the model."
    )


class HeadMetadataClassification(HeadMetadata):
    """Metadata for classification head.

    @type family: str
    @ivar family: Decoding family.
    @type is_softmax: bool
    @ivar is_softmax: True, if output is already softmaxed.
    """

    family: Literal["Classification"] = Field(..., description="Decoding family.")
    is_softmax: bool = Field(description="True, if output is already softmaxed.")

    @field_validator("family")
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

    stride: int = Field(
        description="Step size at which the filter (or kernel) moves across the input data during convolution."
    )
    iou_threshold: float = Field(
        description="Non-max supression threshold limiting boxes intersection."
    )
    conf_threshold: float = Field(
        description="Confidence score threshold above which a detected object is considered valid."
    )
    max_det: int = Field(description="Maximum detections per image.")


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

    family: Literal["ObjectDetectionYOLO"] = Field(..., description="Decoding family.")
    subtype: ObjectDetectionSubtypeYOLO = Field(
        description="YOLO family decoding subtype (e.g. v5, v6, v7 etc.)."
    )
    n_keypoints: Optional[int] = Field(
        None, description="Number of keypoints per bbox if provided."
    )
    n_prototypes: Optional[int] = Field(
        None, description="Number of prototypes per bbox if provided."
    )
    prototype_output_name: Optional[str] = Field(
        None, description="Output node containing prototype information."
    )

    @field_validator("family")
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

    family: Literal["ObjectDetectionSSD"] = Field(..., description="Decoding family.")
    anchors: Optional[List[List[int]]] = Field(
        None,
        description="Predefined bounding boxes of different sizes and aspect ratios.",
    )

    @field_validator("family")
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

    family: Literal["Segmentation"] = Field(..., description="Decoding family.")
    is_softmax: bool = Field(description="True, if output is already softmaxed.")

    @field_validator("family")
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

    head_id: str = Field(description="Unique head identifier.")
    metadata: Union[
        HeadMetadataObjectDetectionYOLO,
        HeadMetadataObjectDetectionSSD,
        HeadMetadataSegmentation,
        HeadMetadataClassification,
        # HeadMetadataKeypointDetection, # TODO
    ] = Field(description="Parameters required by head to run postprocessing.")
