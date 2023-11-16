from pydantic import BaseModel, validator
from .custom_base_model import CustomBaseModel
from typing import Optional, List, Union
from ..enums import *
from abc import ABC

class HeadMetadata(BaseModel, ABC):
    """
    Head metadata parent class.
    
    Attributes:
        label_type (LabelType): Head label type (used to promote differentiation between classes).
        classes (list): Names of object classes recognized by the model.
        n_classes (int): Number of object classes recognized by the model.
    """
    label_type: LabelType
    classes: List[str]
    n_classes: int


class HeadMetadataClassification(HeadMetadata):
    """
    Metadata for classification head.
    
    Attributes:
        is_softmax (bool): True, if output is already softmaxed.
    """
    is_softmax: bool

    @validator("label_type")
    def validate_label_type(
        cls,
        value,
        ):
        if value != LabelType.CLASSIFICATION:
            raise ValueError("wrong HeadMetadata child class")
        return value

class HeadMetadataObjectDetection(HeadMetadata):
    """
    Metadata for object detection head.

    Attributes:
        stride (int): Step size at which the filter (or kernel) moves across the input data during convolution.
        iou_threshold (float): Non-max supression threshold limiting boxes intersection.
        conf_threshold (float): Confidence score threshold above which a detected object is considered valid.
        max_det (int): Maximum detections per image.
        n_keypoints (int): Number of keypoints per bbox if provided.
        n_prototypes (int): Number of prototypes per bbox if provided.
        prototype_output_name (str): Output node containing prototype information.
        anchors (list): Predefined bounding boxes of different sizes and aspect ratios.
    """
    stride: int
    iou_threshold: float
    conf_threshold: float
    max_det: int

    @validator("label_type")
    def validate_label_type(
        cls,
        value,
        ):
        if value != LabelType.OBJECT_DETECTION:
            raise ValueError("wrong HeadMetadata child class")
        return value

class HeadMetadataObjectDetectionYOLO(HeadMetadataObjectDetection):
    """
    Metadata for YOLO object detection head.

    Attributes:
        subtype (ObjectDetectionSubtypeYOLO): Determines YOLO family decoding subtype.
        n_keypoints (int): Number of keypoints per bbox if provided.
        n_prototypes (int): Number of prototypes per bbox if provided.
        prototype_output_name (str): Output node containing prototype information.
    """
    subtype: ObjectDetectionSubtypeYOLO
    n_keypoints: Optional[int] = None
    n_prototypes: Optional[int] = None
    prototype_output_name: Optional[str] = None
    
class HeadMetadataObjectDetectionSSD(HeadMetadataObjectDetection):
    """
    Metadata for SSD object detection head.

    Attributes:
        subtype (ObjectDetectionSubtypeYOLO): Determines SSD family decoding subtype.
        anchors (list): Predefined bounding boxes of different sizes and aspect ratios.
    """
    subtype: ObjectDetectionSubtypeSSD
    anchors: Optional[List[List[int]]] = None

class HeadMetadataSegmentation(HeadMetadata):
    """
    Metadata for segmentation head. 
    
    Attributes:
        is_softmax (bool): True, if output is already softmaxed.
    """
    is_softmax: bool

    @validator("label_type")
    def validate_label_type(
        cls,
        value,
        ):
        if value != LabelType.SEGMENTATION:
            raise ValueError("wrong HeadMetadata child class")
        return value

class HeadMetadataKeypointDetection(HeadMetadata):
    """
    Metadata for keypoint detection head.
    """
    def __init__(self):
        raise NotImplementedError

class Head(CustomBaseModel):
    """
    Represents head of a model.

    Attributes:
        head_id (str): Unique head identifier.
        metadata: (HeadMetadata object): Parameters required by head to run postprocessing.
    """
    head_id: str
    metadata: Union[
        HeadMetadataObjectDetection,
        HeadMetadataSegmentation,
        HeadMetadataClassification,
        #HeadMetadataKeypointDetection, # TODO
        ]