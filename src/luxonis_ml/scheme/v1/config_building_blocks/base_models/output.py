from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple, Literal, Union, Annotated
from ..enums import *

class Output(BaseModel):
    name: str
    dtype: DataType
    subtask: Optional[Any] = None # TODO
    subtype: Optional[Any] = None # TODO
    head_ids: List[str] # list because a single output can go into multiple heads