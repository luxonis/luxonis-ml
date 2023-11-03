from pydantic import BaseModel
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *
from abc import ABC

class HeadMetadata(BaseModel, ABC):

    """
    Parent class for decoding head metadata. Considered together with its children classes, the following arguments are accepted:
        - classes: Array of object class names recognized by the model;
        - n_classes: Number of object classes recognized by the model;
        - stride: Step size at which the filter (or kernel) moves across the input data during convolution;
        - anchors (aka anchor boxes): Predefined bounding boxes of different sizes and aspect ratios;
        - iou_threshold (aka. NMS threshold): Limits intersection of boxes (boxes with intersection-over-union (IoU) greater than this threshold are suppressed, and only the one with the highest confidence score is kept);
        - conf_threshold: Confidence score threshold above which a detected object is considered valid;
        - max_det: maximum detections per image
    """

    decoding_subfamily: DecodingSubFamily = None

class HeadMetadataClassification(HeadMetadata):
    labels: List[str]
    n_labels: int

class HeadMetadataObjectDetection(HeadMetadata):
    labels: List[str]
    n_labels: int
    stride: int
    anchors: List[List[int]] = None # optional as some models (e.g. late versions of YOLO) use anchors as an integral part of their architecture.
    iou_threshold: float
    conf_threshold: float
    max_det: int

class HeadMetadataSegmentation(HeadMetadata):
    pass # TODO

class HeadMetadataKeypointDetection(HeadMetadata):
    n_keypoints = int = None

class Head(CustomBaseModel):
    head_id: str
    task_type: TaskType
    decoding_family: DecodingFamily = None # optional because this is mostly relevant for object detection
    metadata: Union[
        HeadMetadataObjectDetection,
        HeadMetadataKeypointDetection,
        HeadMetadataClassification,
        #HeadMetadataSegmentation, # TODO
        ]