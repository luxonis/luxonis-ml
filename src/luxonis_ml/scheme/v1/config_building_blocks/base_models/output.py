from .custom_base_model import CustomBaseModel
from typing import List
from ..enums import *

class Output(CustomBaseModel):
    """
    Represents output stream of a model.

    Attributes:
        name (str): Name of the output layer.
        dtype (DataType): Data type of the output data (e.g., 'float32').
        head_ids (list): IDs of heads which accept this output stream (beware that a single output can go into multiple heads).
    """
    name: str
    dtype: DataType
    head_ids: List[str]