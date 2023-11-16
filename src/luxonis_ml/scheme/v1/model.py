from typing import List
from config_building_blocks import *

class Model(CustomBaseModel):
    """
    Class that defines a single-stage model.

    Attribures
        metadata (Metadata): Metadata object defining the model metadata.
        inputs (list): List of Input objects defining the model inputs.
        outputs (list): List of Output objects defining the model outputs.
        heads: (list): List of Head objects defining the model heads.
    """
    metadata: Metadata
    inputs: List[Input]
    outputs: List[Output]
    heads: List[Head]