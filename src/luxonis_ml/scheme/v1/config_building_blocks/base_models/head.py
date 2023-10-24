from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *
    
class Head(CustomBaseModel):
    head_id: str
    task_type: TaskType
    decoding_family: DecodingFamily = None # optional because this is mostly relevant for object detection
    decoding_subfamily: DecodingSubFamily = None
    metadata: Dict[str, Any] = None # optional; additional info required for decoding, e.g. "labels", "anchors", "anchor_masks", "iou_threshold", "conf_threshold"