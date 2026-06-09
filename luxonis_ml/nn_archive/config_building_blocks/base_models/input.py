from contextlib import suppress
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from luxonis_ml.nn_archive.config_building_blocks.enums import (
    DataType,
    InputType,
)
from luxonis_ml.nn_archive.utils import infer_layout
from luxonis_ml.typing import BaseModelExtraForbid


class PreprocessingBlock(BaseModelExtraForbid):
    """Preprocessing operations applied to model input data.

    Attributes:
        mean: Optional mean values in channel order. The order should
            match the preprocessing used during training.
        scale: Optional standardization values in channel order. The
            order should match the preprocessing used during training.
        reverse_channels: Optional legacy channel-order flag. Deprecated;
            use ``dai_type`` instead.
        interleaved_to_planar: Optional legacy layout conversion flag.
            Deprecated; use ``dai_type`` instead.
        dai_type: Optional DepthAI input type used to configure pipeline
            input handling.
    """

    mean: list[float] | None = Field(
        None,
        description=(
            "Mean values in channel order, matching the preprocessing used "
            "during training."
        ),
    )
    scale: list[float] | None = Field(
        None,
        description=(
            "Standardization values in channel order, matching the "
            "preprocessing used during training."
        ),
    )
    reverse_channels: bool | None = Field(
        None,
        deprecated="Deprecated, use `dai_type` instead.",
        description="Legacy channel-order flag. Deprecated; use `dai_type` instead.",
    )
    interleaved_to_planar: bool | None = Field(
        None,
        deprecated="Deprecated, use `dai_type` instead.",
        description="Legacy layout conversion flag. Deprecated; use `dai_type` instead.",
    )
    dai_type: str | None = Field(
        None,
        description="DepthAI input type used to configure pipeline input handling.",
    )


class Input(BaseModelExtraForbid):
    """Model input stream definition.

    Attributes:
        name: Name of the input layer.
        dtype: Data type of the input data, such as ``"float32"``.
        input_type: Expected kind of input data, such as ``"image"``.
        shape: Input tensor shape.
        layout: Letter code for interpreting tensor dimensions, such as
            ``"NCHW"``.
        preprocessing: Preprocessing applied to the input data.
    """

    name: str = Field(description="Name of the input layer.")
    dtype: DataType = Field(
        description="Data type of the input data, such as 'float32'."
    )
    input_type: InputType = Field(
        description="Expected kind of input data, such as 'image'."
    )
    shape: list[int] = Field(
        min_length=1,
        description="Input tensor shape.",
    )
    layout: str = Field(
        "NCHW",
        description="Letter code for interpreting tensor dimensions, such as 'NCHW'.",
        min_length=1,
    )
    preprocessing: PreprocessingBlock = Field(
        description="Preprocessing steps applied to the input data."
    )

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        """Validate that the layout is compatible with the input
        shape."""
        self.layout = self.layout.upper()

        if len(self.layout) != len(set(self.layout)):
            raise ValueError("Layout must not contain duplicate letters.")

        if len(self.layout) != len(self.shape):
            raise ValueError("Layout and shape must have the same length.")

        if "N" in self.layout and self.layout[0] != "N":
            raise ValueError(
                "If N (batch size) is included in the layout, it must be first"
            )

        if self.input_type is InputType.IMAGE and "C" not in self.layout:
            raise ValueError(
                "C letter must be present in layout for image input type."
            )

        return self

    @model_validator(mode="before")
    @staticmethod
    def infer_layout(data: dict[str, Any]) -> dict[str, Any]:
        """Infer the layout when a shape is provided without one."""
        if "shape" in data and "layout" not in data:
            with suppress(Exception):
                data["layout"] = infer_layout(data["shape"])
        return data
