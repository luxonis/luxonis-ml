from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *

class PostprocessingBlock(CustomBaseModel):
    type: PostprocessingBlockType
    param: Dict[str, Any] = {}
    
class Head(CustomBaseModel):
    head_id: str
    family: Optional[Any] = None 
    subfamily: Optional[Any] = None
    task_type: TaskType
    metadata: Dict[str, Any] = {} # e.g. "labels", "labels_n", "anchors", "anchor_masks", "iou_threshold", "confidence_threshold"
    postprocessing: List[PostprocessingBlock] # list of postprocessing blocks