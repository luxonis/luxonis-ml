from pydantic import Field
from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *
from abc import ABC

class PreprocessingBlock(CustomBaseModel):
    """
    Preprocessing operations required by the model. The following arguments are accepted:
        - mean: mean normalization values in BGR order,
        - scale: standardization values in BGR order,
        - reverse_channels: If True, image color channels are reversed (e.g. RGB to BGR and vice versa),
        - interleaved_to_planar: If True, change input from interleaved to planar format,
    """
    mean: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    reverse_channels: Optional[bool] = False
    interleaved_to_planar: Optional[bool] = False

class Input(CustomBaseModel):
    """
    Model input class.
    """
    name: str
    dtype: DataType
    input_type: InputType
    shape: Annotated[List[int], Field(min_length=1, max_length=5)] # ..., [H,W] or [H,W,C] or [BS,H,W,C], ...
    preprocessing: PreprocessingBlock