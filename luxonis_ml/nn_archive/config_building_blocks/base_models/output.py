from typing import Any, List, Optional

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from luxonis_ml.utils import BaseModelExtraForbid

from ...utils import parse_layout
from ..enums import DataType


class Output(BaseModelExtraForbid):
    """Represents output stream of a model.

    @type name: str
    @ivar name: Name of the output layer.
    @type dtype: DataType
    @ivar dtype: Data type of the output data (e.g., 'float32').
    """

    name: str = Field(description="Name of the output layer.")
    dtype: DataType = Field(
        description="Data type of the output data (e.g., 'float32')."
    )
    shape: Optional[List[int]] = Field(
        None,
        description="Shape of the output as a list of integers (e.g. [H,W], [H,W,C], [N,H,W,C], ...).",
    )
    layout: Optional[List[str]] = Field(
        None,
        description="List of letters describing the output layout (e.g., ['N', 'C', 'H', 'W']).",
    )

    @field_validator("layout", mode="before")
    @classmethod
    def parse_layout(cls, layout: Any) -> Optional[List[str]]:
        if layout is None:
            return None
        if not isinstance(layout, (list, str)):
            raise ValueError("Layout must be a list of strings or a string.")
        return parse_layout(layout)

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        if self.layout is None:
            return self

        if self.shape is None:
            raise ValueError("Shape must be defined if layout is defined.")

        if len(self.layout) != len(self.shape):
            raise ValueError("Layout and shape must have the same length.")

        return self
