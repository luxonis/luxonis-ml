from typing import List, Optional

from pydantic import Field

from luxonis_ml.utils import BaseModelExtraForbid

from .config_building_blocks import HeadType, Input, Metadata, Output


class Model(BaseModelExtraForbid):
    """Class defining a single-stage model config scheme.

    @type metadata: Metadata
    @ivar metadata: Metadata object defining the model metadata.
    @type inputs: list
    @ivar inputs: List of Input objects defining the model inputs.
    @type outputs: list
    @ivar outputs: List of Output objects defining the model outputs.
    @type heads: list
    @ivar heads: List of Head objects defining the model heads. If not
        defined, we assume a raw output.
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
    heads: Optional[List[HeadType]] = Field(
        description="List of Head objects defining the model heads. If not defined, we assume a raw output."
    )
