from abc import ABC
from typing import Dict, List, Literal, Optional, Union

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
    @type outputs: C{Union[List[str], Dict[str, str]]}
    @ivar outputs: A list of output names from the `outputs` block of the archive or a dictionary mapping DepthAI parser names needed for the head to output names. The referenced outputs will be used by the DepthAI parser.
    """

    family: str = Field(description="Decoding family.")
    classes: List[str] = Field(
        description="Names of object classes recognized by the model."
    )
    n_classes: int = Field(
        description="Number of object classes recognized by the model."
    )
    outputs: Union[List[str], Dict[str, str]] = Field(
        description="A list of output names from the `outputs` block of the archive or a dictionary mapping DepthAI parser names needed for the head to output names. The referenced outputs will be used by the DepthAI parser."
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

    @type iou_threshold: float
    @ivar iou_threshold: Non-max supression threshold limiting boxes intersection.
    @type conf_threshold: float
    @ivar conf_threshold: Confidence score threshold above which a detected object is
        considered valid.
    @type max_det: int
    @ivar max_det: Maximum detections per image.
    @type anchors: C{Optional[List[List[List[int]]]]}
    @ivar anchors: Predefined bounding boxes of different sizes and aspect ratios. The
        innermost lists are length 2 tuples of box sizes. The middle lists are anchors
        for each output. The outmost lists go from smallest to largest output.
    """

    iou_threshold: float = Field(
        description="Non-max supression threshold limiting boxes intersection."
    )
    conf_threshold: float = Field(
        description="Confidence score threshold above which a detected object is considered valid."
    )
    max_det: int = Field(description="Maximum detections per image.")
    anchors: Optional[List[List[List[int]]]] = Field(
        None,
        description="Predefined bounding boxes of different sizes and aspect ratios. The innermost lists are length 2 tuples of box sizes. The middle lists are anchors for each output. The outmost lists go from smallest to largest output.",
    )


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

    @field_validator("anchors")
    def validate_anchors(
        cls,
        value,
    ):
        if cls.subtype == ObjectDetectionSubtypeYOLO.YOLOv6 and value is not None:
            raise ValueError("YOLOv6 does not support anchors.")
        return value


class HeadMetadataObjectDetectionSSD(HeadMetadataObjectDetection):
    """Metadata for SSD object detection head.

    @type family: str
    @ivar family: Decoding family.
    """

    family: Literal["ObjectDetectionSSD"] = Field(..., description="Decoding family.")

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


class HeadMetadataSegmentationYOLO(
    HeadMetadataObjectDetectionYOLO, HeadMetadataSegmentation
):
    """Metadata for YOLO instance segmentation head.

    @type family: str
    @ivar family: Decoding family.
    @type postprocessor_path: str
    @ivar postprocessor_path: Path to the secondary executable used in YOLO instance
        segmentation.
    """

    family: Literal["InstanceSegmentationYOLO"] = Field(
        ..., description="Decoding family."
    )
    postprocessor_path: str = Field(
        ...,
        description="Path to the secondary executable used in YOLO instance segmentation.",
    )

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
