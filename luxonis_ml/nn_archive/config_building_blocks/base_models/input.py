from typing import Any, List, Optional

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from luxonis_ml.utils import BaseModelExtraForbid

from ...utils import parse_layout
from ..enums import DataType, InputType


class PreprocessingBlock(BaseModelExtraForbid):
    """Represents preprocessing operations applied to the input data.

    @type mean: list
    @ivar mean: Mean values in channel order. Typically, this is BGR order.
    @type scale: list
    @ivar scale: Standardization values in channel order. Typically, this is BGR order.
    @type reverse_channels: bool | None
    @ivar reverse_channels: If True, color channels are reversed (e.g. BGR to RGB or
        vice versa).
    @type interleaved_to_planar: bool | None
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
        None,
        description="If True, color channels are reversed (e.g. BGR to RGB or vice versa).",
    )
    interleaved_to_planar: Optional[bool] = Field(
        None, description="If True, format is changed from interleaved to planar."
    )


class Input(BaseModelExtraForbid):
    """Represents input stream of a model.

    @type name: str
    @ivar name: Name of the input layer.

    @type dtype: DataType
    @ivar dtype: Data type of the input data (e.g., 'float32').

    @type input_type: InputType
    @ivar input_type: Type of input data (e.g., 'image').

    @type shape: list
    @ivar shape: Shape of the input data as a list of integers (e.g. [H,W], [H,W,C], [N,H,W,C], ...).

    @type layout: str
    @ivar layout: Lettercode interpretation of the input data dimensions (e.g., 'NCHW').

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
        description="Shape of the input data as a list of integers (e.g. [H,W], [H,W,C], [N,H,W,C], ...).",
    )
    layout: List[str] = Field(
        description="List of letters describing the input layout (e.g., ['N', 'C', 'H', 'W']).",
        min_length=1,
        default=list("NCHW"),
    )
    preprocessing: PreprocessingBlock = Field(
        description="Preprocessing steps applied to the input data."
    )

    @field_validator("layout", mode="before")
    @classmethod
    def parse_layout(cls, layout: Any) -> List[str]:
        if not isinstance(layout, (list, str)):
            raise ValueError("Layout must be a list of strings or a string.")
        return parse_layout(layout)

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        if len(self.layout) != len(self.shape):
            raise ValueError("Layout and shape must have the same length.")

        if self.input_type == InputType.IMAGE:
            if "C" not in self.layout:
                raise ValueError(
                    "C letter must be present in layout for image input type."
                )

        return self
