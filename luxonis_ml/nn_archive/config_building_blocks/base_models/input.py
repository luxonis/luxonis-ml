from typing import List, Optional

from pydantic import Field

from ..enums import DataType, InputType
from .custom_base_model import CustomBaseModel


class PreprocessingBlock(CustomBaseModel):
    """Represents preprocessing operations applied to the input data.

    @type mean: list
    @ivar mean: Mean values in channel order. Typically, this is BGR order.
    @type scale: list
    @ivar scale: Standardization values in channel order. Typically, this is BGR order.
    @type reverse_channels: bool
    @ivar reverse_channels: If True, color channels are reversed (e.g. BGR to RGB or
        vice versa).
    @type interleaved_to_planar: bool
    @ivar interleaved_to_planar: If True, format is changed from interleaved to planar.
    """

    mean: Optional[List[float]] = Field(
        None, description="Mean values in channel order. Typically, this is BGR order."
    )
    scale: Optional[List[float]] = Field(
        None,
        description="Standardization values in channel order. Typically, this is BGR order.",
    )
    reverse_channels: Optional[bool] = Field(
        False,
        description="If True, color channels are reversed (e.g. BGR to RGB or vice versa).",
    )
    interleaved_to_planar: Optional[bool] = Field(
        False, description="If True, format is changed from interleaved to planar."
    )


class Input(CustomBaseModel):
    """Represents input stream of a model.

    @type name: str
    @ivar name: Name of the input layer.

    @type dtype: DataType
    @ivar dtype: Data type of the input data (e.g., 'float32').

    @type input_type: InputType
    @ivar input_type: Type of input data (e.g., 'image').

    @type shape: list
    @ivar shape: Shape of the input data as a list of integers (e.g. [H,W], [H,W,C], [BS,H,W,C], ...).

    @type preprocessing: PreprocessingBlock
    @ivar preprocessing: Preprocessing steps applied to the input data.
    """

    name: str = Field(description="Name of the input layer.")
    dtype: DataType = Field(
        description="Data type of the input data (e.g., 'float32')."
    )
    input_type: InputType = Field(description="Type of input data (e.g., 'image').")
    shape: List[int] = Field(
        min_length=1,
        max_length=5,
        description="Shape of the input data as a list of integers (e.g. [H,W], [H,W,C], [BS,H,W,C], ...).",
    )
    preprocessing: PreprocessingBlock = Field(
        description="Preprocessing steps applied to the input data."
    )
