from pydantic import Field
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *
from abc import ABC

class PreprocessingBlock(CustomBaseModel):
    mean: Optional[List[float]] = None # e.g. [mean_B, mean_G, mean_R]
    scale: Optional[List[float]] = None # e.g. [scale_B, scale_G, scale_R]
    reverse_channels: Optional[bool] = False
    interleaved_to_planar: Optional[bool] = False

class Input(CustomBaseModel):
    name: str
    dtype: DataType
    input_type: InputType
    shape: Annotated[List[int], Field(min_length=1, max_length=5)] # ..., [H,W] or [H,W,C] or [BS,H,W,C], ...
    preprocessing: PreprocessingBlock