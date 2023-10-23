from .custom_base_model import CustomBaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *

class Output(CustomBaseModel):
    name: str
    dtype: DataType
    subtask: Optional[Any] = None # TODO
    subtype: Optional[Any] = None # TODO
    head_ids: List[str] # list because a single output can go into multiple heads