from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *

class PreprocessingBlock(BaseModel):
    type: PreprocessingBlockType
    param: Dict[str, Any] = {} # e.g. ["mean_b","mean_g","mean_r"], ["scale_b","scale_g","scale_r"], ...

class Input(BaseModel):
    name: str
    dtype: DataType
    input_type: InputType
    shape: List[int] # ["bs", "h", "w", "c"]
    preprocessing: List[PreprocessingBlock] # list of preprocessing blocks; order is important!