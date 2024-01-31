from typing import List

from pydantic import Field

from .config_building_blocks import CustomBaseModel, Head, Input, Metadata, Output


class Model(CustomBaseModel):
    """Class defining a single-stage model config scheme.

    @type metadata: Metadata
    @ivar metadata: Metadata object defining the model metadata.
    @type inputs: list
    @ivar inputs: List of Input objects defining the model inputs.
    @type outputs: list
    @ivar outputs: List of Output objects defining the model outputs.
    @type heads: list
    @ivar heads: List of Head objects defining the model heads.
    """

    metadata: Metadata = Field(
        description="Metadata object defining the model metadata."
    )
    inputs: List[Input] = Field(
        description="List of Input objects defining the model inputs."
    )
    outputs: List[Output] = Field(
        description="List of Output objects defining the model outputs."
    )
    heads: List[Head] = Field(
        description="List of Head objects defining the model heads."
    )
