from contextlib import suppress
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator
from typing_extensions import Self

from luxonis_ml.utils import BaseModelExtraForbid

from ...utils import infer_layout
from ..enums import DataType, InputType


class PreprocessingBlock(BaseModelExtraForbid):
    """Represents preprocessing operations applied to the input data.

    @type mean: list
    @ivar mean: Mean values in channel order. Typically, this is BGR
        order.
    @type scale: list
    @ivar scale: Standardization values in channel order. Typically,
        this is BGR order.
    @type reverse_channels: bool | None
    @ivar reverse_channels: If True input to the model is RGB else BGR.
    @type interleaved_to_planar: bool | None
    @ivar interleaved_to_planar: If True input to the model is
        interleaved (NHWC) else planar (NCHW).
    @type layout: str | None
    @ivar layout: DepthAI input type which is read by DepthAI to
        automatically setup the pipeline.
    """

    mean: Optional[List[float]] = Field(
        None,
        description="Mean values in channel order. Typically, this is BGR order.",
    )
    scale: Optional[List[float]] = Field(
        None,
        description="Standardization values in channel order. Typically, this is BGR order.",
    )
    reverse_channels: Optional[bool] = Field(
        True,
        deprecated="Deprecated, use `dai_type` instead.",
        description="If True input to the model is RGB else BGR.",
    )
    interleaved_to_planar: Optional[bool] = Field(
        False,
        deprecated="Deprecated, use `dai_type` instead.",
        description="If True input to the model is interleaved (NHWC) else planar (NCHW).",
    )
    dai_type: str = Field(
        "RGB888p",
        description="DepthAI input type which is read by DepthAI to automatically setup the pipeline.",
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
    input_type: InputType = Field(
        description="Type of input data (e.g., 'image')."
    )
    shape: List[int] = Field(
        min_length=1,
        description="Shape of the input data as a list of integers (e.g. [H,W], [H,W,C], [N,H,W,C], ...).",
    )
    layout: str = Field(
        "NCHW",
        description="Lettercode interpretation of the input data dimensions (e.g., 'NCHW')",
        min_length=1,
    )
    preprocessing: PreprocessingBlock = Field(
        description="Preprocessing steps applied to the input data."
    )

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        self.layout = self.layout.upper()

        if len(self.layout) != len(set(self.layout)):
            raise ValueError("Layout must not contain duplicate letters.")

        if len(self.layout) != len(self.shape):
            raise ValueError("Layout and shape must have the same length.")

        if "N" in self.layout and self.layout[0] != "N":
            raise ValueError(
                "If N (batch size) is included in the layout, it must be first"
            )

        if self.input_type == InputType.IMAGE:
            if "C" not in self.layout:
                raise ValueError(
                    "C letter must be present in layout for image input type."
                )

        return self

    @model_validator(mode="before")
    @staticmethod
    def infer_layout(data: Dict[str, Any]) -> Dict[str, Any]:
        if "shape" in data and "layout" not in data:
            with suppress(Exception):
                data["layout"] = infer_layout(data["shape"])
        return data
