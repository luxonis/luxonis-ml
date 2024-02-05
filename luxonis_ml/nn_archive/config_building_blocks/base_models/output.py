from typing import List

from pydantic import Field

from ..enums import DataType
from .custom_base_model import CustomBaseModel


class Output(CustomBaseModel):
    """Represents output stream of a model.

    @type name: str
    @ivar name: Name of the output layer.
    @type dtype: DataType
    @ivar dtype: Data type of the output data (e.g., 'float32').
    @type head_ids: list
    @ivar head_ids: IDs of heads which accept this output stream (beware that a single
        output can go into multiple heads).
    """

    name: str = Field(description="Name of the output layer.")
    dtype: DataType = Field(
        description="Data type of the output data (e.g., 'float32')."
    )
    head_ids: List[str] = Field(
        description="IDs of heads which accept this output stream (beware that a single output can go into multiple heads)."
    )
