from typing import List
from config_building_blocks import Metadata, Input, Output, Head, CustomBaseModel


class Model(CustomBaseModel):
    """Class defining a single-stage model config scheme.

    Attributes:
        metadata (Metadata): Metadata object defining the model metadata.
        inputs (list): List of Input objects defining the model inputs.
        outputs (list): List of Output objects defining the model outputs.
        heads: (list): List of Head objects defining the model heads.
    """

    metadata: Metadata
    inputs: List[Input]
    outputs: List[Output]
    heads: List[Head]
