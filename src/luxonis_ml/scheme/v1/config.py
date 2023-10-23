from typing import List
from config_building_blocks import *

class Config(CustomBaseModel):
    metadata: Metadata
    inputs: List[Input]
    outputs: List[Output]
    heads: List[Head]