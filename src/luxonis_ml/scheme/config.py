from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from enum import Enum

# ---- ENUM CLASSES ----

class Platform(Enum):
    HAILO = "hailo"
    RVC2 = "rvc2"
    RVC3 = "rvc3"
    RVC4 = "rvc4"

class DataType(Enum):
    INT8 = "int8"
    UINT8 = "uint8"
    FLOAT32 = "float32"

class InputType(Enum):
    RAW = "raw"
    IMAGE = "image"

class PreprocessingBlockType(Enum): # required only for RVC4
    MEAN = "mean"
    SCALE = "scale"
    REVERSE_CHANNELS = "reverse_channels" #bgr<->rgb

class PostprocessingBlockType(Enum):
    MISC = "misc" # TODO
    
class TaskType(Enum):
    CLASSIFICATION = "classification" # ?
    MULTICLASS_CLASSIFICATION = "multiclass_classification" # e.g. male-blue, female-blue, male-orange, female-orange - 4 labels
    MULTILABEL_CLASSIFICARION = "multilabel_classification" # e.g. male/female and the other blue/orange - 2 labels
    OBJECT_DETECTION = "object_detection" # ?
    SEMANTIC_SEGMENTATION = "semantic_segmentation" # ?
    INSTANCE_SEGMENTATION = "instance segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    MISC = "misc"
    RAW = "raw"


# ---- PYDANTIC CLASSES ----

class PreprocessingBlock(BaseModel):
    TYPE: PreprocessingBlockType
    PARAMETERS: Dict[str, Any] = {} # e.g. ["mean_b","mean_g","mean_r"], ["scale_b","scale_g","scale_r"], ...

class PostprocessingBlock(BaseModel):
    TYPE: PostprocessingBlockType
    PARAMETERS: Dict[str, Any] = {}

class Metadata(BaseModel):
    model_name: str
    backbone: str
    platform: Platform
    version: int

class Input(BaseModel):
    name: str
    data_type: DataType
    input_type: InputType
    input_shape: List(int) # ["bs", "h", "w", "c"]
    preprocessing: List(PreprocessingBlock) # list of preprocessing blocks; order is important!

class Output(BaseModel):
    name: str
    data_type: DataType
    subtask: Optional[Any] = None # TODO
    subtype: Optional[Any] = None # TODO
    head_id: List(str) # list because a single output can go into multiple heads
    
class Head(BaseModel):
    head_id: str
    task_type: TaskType
    task_subtype: Optional[Any] = None # TODO
    metadata: Dict[str, Any] = {} # e.g. "labels", "labels_n", "anchors", "anchor_masks", "iou_threshold", "confidence_threshold"
    postprocessing: List(PostprocessingBlock) # list of postprocessing blocks

class Config(BaseModel):
    metadata: Metadata
    inputs: List(Input)
    outputs: List(Output)
    heads: List(Head)


# ---- TESTING ----
"""
cfg = Config(
    metadata={"field1":4},
    inputs={"field1":4},
    outputs={"field1":4},
    heads={"field1":4}
)
print(cfg.model_dump_json())
"""