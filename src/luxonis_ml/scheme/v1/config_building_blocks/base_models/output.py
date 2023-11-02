from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *

class Output(CustomBaseModel):
    name: str
    dtype: DataType
    head_ids: List[str] # list because a single output can go into multiple heads
    role: str = None # optional parameter that indicaties the role of this output for the head(s)