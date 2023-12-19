from pydantic import BaseModel, validator, Field
from .custom_base_model import CustomBaseModel
from typing import Optional, List, Union
from ..enums import ObjectDetectionSubtypeYOLO
from abc import ABC


class HeadMetadata(BaseModel, ABC):
    """Head metadata parent class.

    Attributes:
        family (str): Decoding family.
        classes (list): Names of object classes recognized by the model.
        n_classes (int): Number of object classes recognized by the model.
    """

    family: str
    classes: List[str]
    n_classes: int


class HeadMetadataClassification(HeadMetadata):
    """Metadata for classification head.

    Attributes:
        family (str): Decoding family.
        is_softmax (bool): True, if output is already softmaxed.
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

    Attributes:
        stride (int): Step size at which the filter (or kernel) moves across the input data during convolution.
        iou_threshold (float): Non-max supression threshold limiting boxes intersection.
        conf_threshold (float): Confidence score threshold above which a detected object is considered valid.
        max_det (int): Maximum detections per image.
    """

    stride: int
    iou_threshold: float
    conf_threshold: float
    max_det: int


class HeadMetadataObjectDetectionYOLO(HeadMetadataObjectDetection):
    """Metadata for YOLO object detection head.

    Attributes:
        family (str): Decoding family.
        subtype (ObjectDetectionSubtypeYOLO): YOLO family decoding subtype (e.g. v5, v6, v7 etc.).
        n_keypoints (int): Number of keypoints per bbox if provided.
        n_prototypes (int): Number of prototypes per bbox if provided.
        prototype_output_name (str): Output node containing prototype information.
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

    Attributes:
        family (str): Decoding family.
        anchors (list): Predefined bounding boxes of different sizes and aspect ratios.
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

    Attributes:
        family (str): Decoding family.
        is_softmax (bool): True, if output is already softmaxed.
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

    Attributes:
        head_id (str): Unique head identifier.
        metadata: (HeadMetadata object): Parameters required by head to run postprocessing.
    """

    head_id: str
    metadata: Union[
        HeadMetadataObjectDetectionYOLO,
        HeadMetadataObjectDetectionSSD,
        HeadMetadataSegmentation,
        HeadMetadataClassification,
        # HeadMetadataKeypointDetection, # TODO
    ]
