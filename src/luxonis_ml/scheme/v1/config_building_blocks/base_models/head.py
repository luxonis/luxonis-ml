from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *

class PostprocessingBlock(BaseModel):
    type: PostprocessingBlockType
    param: Dict[str, Any] = {}
    
class Head(BaseModel):
    head_id: str
    family: Optional[Any] = None 
    subfamily: Optional[Any] = None
    task_type: TaskType
    metadata: Dict[str, Any] = {} # e.g. "labels", "labels_n", "anchors", "anchor_masks", "iou_threshold", "confidence_threshold"
    postprocessing: List[PostprocessingBlock] # list of postprocessing blocks