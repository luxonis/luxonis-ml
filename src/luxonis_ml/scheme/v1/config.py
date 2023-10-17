from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from enums import *

class PreprocessingBlock(BaseModel):
    type: PreprocessingBlockType
    param: Dict[str, Any] = {} # e.g. ["mean_b","mean_g","mean_r"], ["scale_b","scale_g","scale_r"], ...

class PostprocessingBlock(BaseModel):
    type: PostprocessingBlockType
    param: Dict[str, Any] = {}

class Metadata(BaseModel):
    name: str
    backbone: str
    platform: Platform
    version: int

class Input(BaseModel):
    name: str
    dtype: DataType
    input_type: InputType
    shape: List[int] # ["bs", "h", "w", "c"]
    preprocessing: List[PreprocessingBlock] # list of preprocessing blocks; order is important!

class Output(BaseModel):
    name: str
    dtype: DataType
    subtask: Optional[Any] = None # TODO
    subtype: Optional[Any] = None # TODO
    head_ids: List[str] # list because a single output can go into multiple heads
    
class Head(BaseModel):
    head_id: str
    task_type: TaskType
    task_subtype: Optional[Any] = None # TODO
    metadata: Dict[str, Any] = {} # e.g. "labels", "labels_n", "anchors", "anchor_masks", "iou_threshold", "confidence_threshold"
    postprocessing: List[PostprocessingBlock] # list of postprocessing blocks

class Config(BaseModel):
    metadata: Metadata
    inputs: List[Input]
    outputs: List[Output]
    heads: List[Head]