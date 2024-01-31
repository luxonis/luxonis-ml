from typing import List

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

    metadata: Metadata
    inputs: List[Input]
    outputs: List[Output]
    heads: List[Head]
