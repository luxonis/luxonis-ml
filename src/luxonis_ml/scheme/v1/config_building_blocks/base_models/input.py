from pydantic import Field
from .custom_base_model import CustomBaseModel
from typing import Optional, List, Annotated
from ..enums import *


class PreprocessingBlock(CustomBaseModel):
    """Represents preprocessing operations applied to the input data.

    Attributes:
        mean (list): Mean values in BGR order.
        scale (list): Standardization values in BGR order.
        reverse_channels (bool): If True, color channels are reversed (e.g. BGR to RGB or vice versa).
        interleaved_to_planar (bool): If True, format is changed from interleaved to planar.
    """

    mean: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    reverse_channels: Optional[bool] = False
    interleaved_to_planar: Optional[bool] = False


class Input(CustomBaseModel):
    """Represents input stream of a model.

    Attributes:
        name (str): Name of the input layer.
        dtype (DataType): Data type of the input data (e.g., 'float32').
        input_type (InputType): Type of input data (e.g., 'image').
        shape (list): Shape of the input data as a list of integers (e.g. [H,W], [H,W,C], [BS,H,W,C], ...).
        preprocessing (PreprocessingBlock): Preprocessing steps applied to the input data.
    """

    name: str
    dtype: DataType
    input_type: InputType
    shape: Annotated[List[int], Field(min_length=1, max_length=5)]
    preprocessing: PreprocessingBlock
