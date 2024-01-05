from typing import List, Optional

from pydantic import Field
from typing_extensions import Annotated

from ..enums import DataType, InputType
from .custom_base_model import CustomBaseModel


class PreprocessingBlock(CustomBaseModel):
    """Represents preprocessing operations applied to the input data.

    @type mean: list
    @ivar mean: Mean values in BGR order.
    @type scale: list
    @ivar scale: Standardization values in BGR order.
    @type reverse_channels: bool
    @ivar reverse_channels: If True, color channels are reversed (e.g. BGR to RGB or
        vice versa).
    @type interleaved_to_planar: bool
    @ivar interleaved_to_planar: If True, format is changed from interleaved to planar.
    """

    mean: Optional[List[float]] = None
    scale: Optional[List[float]] = None
    reverse_channels: Optional[bool] = False
    interleaved_to_planar: Optional[bool] = False


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

    name: str
    dtype: DataType
    input_type: InputType
    shape: Annotated[List[int], Field(min_length=1, max_length=5)]
    preprocessing: PreprocessingBlock
