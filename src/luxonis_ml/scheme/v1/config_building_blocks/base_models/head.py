from pydantic import BaseModel
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *
from abc import ABC

class HeadMetadataClassification(BaseModel):
    """
    Metadata for classification head.
    
    Attributes:
        classes (list): Names of object classes recognized by the model.
        n_classes (int): Number of object classes recognized by the model.
    """
    classes: List[str]
    n_classes: int

class HeadMetadataObjectDetection(BaseModel):
    """
    Metadata for object detection head.

    Attributes:
        classes (list): Names of object classes recognized by the model.
        n_classes (int): Number of object classes recognized by the model.
        stride (int): Step size at which the filter (or kernel) moves across the input data during convolution.
        iou_threshold (float): Non-max supression threshold limiting boxes intersection.
        conf_threshold (float): Confidence score threshold above which a detected object is considered valid.
        max_det (int): Maximum detections per image.
        n_keypoints (int): Number of keypoints per bbox if provided.
        n_prototypes (int): Number of prototypes per bbox if provided.
        prototype_output_name (str): Output node containing prototype information.
        anchors (list): Predefined bounding boxes of different sizes and aspect ratios.
    """
    classes: List[str]
    n_classes: int
    stride: int
    iou_threshold: float
    conf_threshold: float
    max_det: int
    subtype: ObjectDetectionSubtype
    n_keypoints: Optional[int] = 0
    n_prototypes: Optional[int] = 0 #
    prototype_output_name: Optional[str] = None
    anchors: Optional[List[List[int]]] = None # optional as some models (e.g. late versions of YOLO) use anchors as an integral part of their architecture.
    
class HeadMetadataSegmentation(BaseModel):
    """
    Metadata for segmentation head. 
    
    Attributes:
        classes (list): Names of object classes recognized by the model.
        n_classes (int): Number of object classes recognized by the model.
        is_softmax (bool): True, if output is already softmaxed.
    """
    classes: List[str]
    n_classes: int
    is_softmax: bool

class HeadMetadataKeypointDetection(BaseModel):
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
        decoding_family (DecodingFamily): Decoding family of the head.
        metadata: (HeadMetadata object): Parameters required by head to run postprocessing.
    """
    head_id: str
    decoding_family: Optional[DecodingFamily] = None # optional because this is mostly relevant for object detection
    metadata: Union[
        HeadMetadataObjectDetection,
        HeadMetadataSegmentation,
        HeadMetadataClassification,
        #HeadMetadataKeypointDetection, # TODO
        ]