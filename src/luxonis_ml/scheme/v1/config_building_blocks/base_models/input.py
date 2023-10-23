from pydantic import Field
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from typing_extensions import TypedDict
from ..enums import *
from abc import ABC

class PreprocessingBlock(CustomBaseModel, ABC):
    type: PreprocessingBlockType

class MeanParamsDict(TypedDict):
    mean_b: float
    mean_g: float
    mean_r: float

class MeanBlock(PreprocessingBlock):
    type: str = Field(pattern="mean", min_length=4, max_length=4)
    param: MeanParamsDict

class ScaleParamsDict(TypedDict):
    scale_b: float
    scale_g: float
    scale_r: float

class ScaleBlock(PreprocessingBlock):
    type: str = Field(pattern="scale", min_length=5, max_length=5)
    param: ScaleParamsDict

class ReverseChannelsBlock(PreprocessingBlock):
    type: str = Field(pattern="reverse_channels", min_length=16, max_length=16)

class InterleavedToPlanarBlock(PreprocessingBlock):
    type: str = Field(pattern="interleaved_to_planar", min_length=21, max_length=21)

class Input(CustomBaseModel):
    name: str
    dtype: DataType
    input_type: InputType
    shape: Annotated[List[int], Field(min_length=2, max_length=4)] # [H,W] or [H,W,C] or [BS,H,W,C]
    preprocessing: List[Union[MeanBlock, ScaleBlock, ReverseChannelsBlock, InterleavedToPlanarBlock, PreprocessingBlock]] 
    # adding parent PreprocessingBlock at the end to catch all missed blocks and compare them to PreprocessingBlockType enum