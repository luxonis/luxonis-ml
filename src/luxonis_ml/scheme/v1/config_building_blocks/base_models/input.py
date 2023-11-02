from pydantic import Field
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *
from abc import ABC

class PreprocessingBlock(CustomBaseModel):
    mean: Annotated[List[float], Field(min_length=3, max_length=3)] = None # [mean_B, mean_G, mean_R]
    scale: Annotated[List[float], Field(min_length=3, max_length=3)] = None # [scale_B, scale_G, scale_R]
    reverse_channels: bool = False
    interleaved_to_planar: bool = False

class Input(CustomBaseModel):
    name: str
    dtype: DataType
    input_type: InputType
    shape: Annotated[List[int], Field(min_length=1, max_length=5)] # ..., [H,W] or [H,W,C] or [BS,H,W,C], ...
    preprocessing: PreprocessingBlock