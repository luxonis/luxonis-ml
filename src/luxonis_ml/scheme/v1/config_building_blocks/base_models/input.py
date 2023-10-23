from pydantic import Field
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from typing_extensions import TypedDict
from ..enums import *
from abc import ABC

class MeanParamsDict(TypedDict):
    mean_b: float
    mean_g: float
    mean_r: float

class ScaleParamsDict(TypedDict):
    scale_b: float
    scale_g: float
    scale_r: float

class PreprocessingBlock(CustomBaseModel):
    mean: MeanParamsDict = None
    scale: ScaleParamsDict = None
    reverse_channels: bool = False
    interleaved_to_planar: bool = False

class Input(CustomBaseModel):
    name: str
    dtype: DataType
    input_type: InputType
    shape: Annotated[List[int], Field(min_length=2, max_length=4)] # [H,W] or [H,W,C] or [BS,H,W,C]
    preprocessing: PreprocessingBlock