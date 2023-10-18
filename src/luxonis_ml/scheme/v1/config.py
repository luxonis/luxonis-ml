from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from config_building_blocks import *

class Config(BaseModel):
    metadata: Metadata
    inputs: List[Input]
    outputs: List[Output]
    heads: List[Head]