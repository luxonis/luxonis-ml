from contextlib import suppress
from typing import List, Optional

from pydantic import Field, model_validator
from typing_extensions import Self

from luxonis_ml.utils import BaseModelExtraForbid

from ...utils import infer_layout
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
        description="Shape of the output as a list of integers (e.g. [1, 1000]).",
    )
    layout: Optional[str] = Field(
        None,
        description="List of letters describing the output layout (e.g. 'NC').",
    )

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        if self.layout is None:
            return self

        self.layout = self.layout.upper()

        if "N" in self.layout and self.layout[0] != "N":
            raise ValueError(
                "If N (batch size) is included in the layout, it must be first"
            )

        if self.shape is None:
            raise ValueError("Shape must be defined if layout is defined.")

        if len(self.layout) != len(self.shape):
            raise ValueError("Layout and shape must have the same length.")

        return self

    @model_validator(mode="after")
    def infer_layout(self) -> Self:
        if self.layout is None and self.shape is not None:
            with suppress(Exception):
                self.layout = infer_layout(self.shape)

        return self
