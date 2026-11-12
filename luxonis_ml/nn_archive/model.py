"""Schema of one model in an NN Archive config."""

from pydantic import Field

from luxonis_ml.typing import BaseModelExtraForbid

from .config_building_blocks import (
    HeadType,
    Input,
    Metadata,
    Output,
    RuntimeDependencies,
)


class Model(BaseModelExtraForbid):
    """Configuration for one model stage in an NN Archive.

    Attributes:
        metadata: Model-level metadata, including the executable path.
        inputs: Input stream definitions.
        outputs: Output stream definitions.
        heads: Optional parser head definitions. If omitted, the archive
            exposes raw model outputs.
        runtime_dependencies: Optional runtime libraries required by the
            executable, such as SNPE UDO packages.

    """

    metadata: Metadata = Field(
        description="Model-level metadata, including the executable path."
    )
    inputs: list[Input] = Field(description="Input stream definitions.")
    outputs: list[Output] = Field(description="Output stream definitions.")
    heads: list[HeadType] | None = Field(
        description="Parser head definitions. If omitted, the archive exposes raw model outputs."
    )
    runtime_dependencies: RuntimeDependencies | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Runtime libraries required by the model executable.",
    )
