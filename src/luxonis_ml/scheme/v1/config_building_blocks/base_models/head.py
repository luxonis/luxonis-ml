from pydantic import BaseModel
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *
from abc import ABC

class HeadMetadataClassification(BaseModel):
    """
    Metadata for classification head. The following arguments are accepted:
        - classes: Array of object class names recognized by the model;
        - n_classes: Number of object classes recognized by the model;
    """
    classes: List[str]
    n_classes: int

class HeadMetadataObjectDetection(BaseModel):
    """
    Metadata for object detection head. The following arguments are accepted:
        - classes: Array of object class names recognized by the model;
        - n_classes: Number of object classes recognized by the model;
        - stride: Step size at which the filter (or kernel) moves across the input data during convolution;
        - anchors (aka anchor boxes): Predefined bounding boxes of different sizes and aspect ratios;
        - iou_threshold (aka. NMS threshold): Limits intersection of boxes (boxes with intersection-over-union (IoU) greater than this threshold are suppressed, and only the one with the highest confidence score is kept);
        - conf_threshold: Confidence score threshold above which a detected object is considered valid;
        - max_det: maximum detections per image
        - n_keypoints: number of keypoints per bbox if provided;
        - n_prototypes: number of prototypes per bbox if provided;
        - prototype_output_name: output node containing prototype information
        - subtype: decoding subtype used to differentiate object decoding of members of the same decoding_family

    """
    classes: List[str]
    n_classes: int
    stride: int
    iou_threshold: float
    conf_threshold: float
    max_det: int
    subtype: ObjectDetectionSubtype
    n_keypoints: int = 0
    n_prototypes: int = 0 #
    prototype_output_name: str = None
    anchors: List[List[int]] = None # optional as some models (e.g. late versions of YOLO) use anchors as an integral part of their architecture.
    
class HeadMetadataSegmentation(BaseModel):
    """
    Metadata for segmentation head. The following arguments are accepted:
        - classes: Array of object class names recognized by the model;
        - n_classes: Number of object classes recognized by the model;
        - is_softmax: True, if output is already softmaxed
    """
    classes: List[str]
    n_classes: int
    is_softmax: bool

class HeadMetadataKeypointDetection(BaseModel):
    """
    Metadata for keypoint detection head
    """
    def __init__(self):
        raise NotImplementedError

class Head(CustomBaseModel):
    head_id: str
    decoding_family: DecodingFamily = None # optional because this is mostly relevant for object detection
    metadata: Union[
        HeadMetadataObjectDetection,
        HeadMetadataSegmentation,
        HeadMetadataClassification,
        #HeadMetadataKeypointDetection, # TODO
        ]