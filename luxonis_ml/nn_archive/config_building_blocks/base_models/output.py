from contextlib import suppress

from pydantic import Field, model_validator
from typing_extensions import Self

from luxonis_ml.nn_archive.config_building_blocks.enums import DataType
from luxonis_ml.nn_archive.utils import infer_layout
from luxonis_ml.typing import BaseModelExtraForbid


class Output(BaseModelExtraForbid):
    """Model output stream definition.

    Attributes:
        name: Name of the output layer.
        dtype: Data type of the output data, such as ``"float32"``.
        shape: Optional output tensor shape.
        layout: Optional letter code for interpreting tensor dimensions,
            such as ``"NC"``.
    """

    name: str = Field(description="Name of the output layer.")
    dtype: DataType = Field(
        description="Data type of the output data, such as 'float32'."
    )
    shape: list[int] | None = Field(
        None,
        description="Output tensor shape.",
    )
    layout: str | None = Field(
        None,
        description="Letter code for interpreting tensor dimensions, such as 'NC'.",
    )

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        """Validate that the layout is compatible with the output
        shape."""
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
        """Infer the layout when a shape is provided without one."""
        if self.layout is None and self.shape is not None:
            with suppress(Exception):
                self.layout = infer_layout(self.shape)

        return self
