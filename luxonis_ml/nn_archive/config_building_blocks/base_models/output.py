from pydantic import Field

from luxonis_ml.utils import BaseModelExtraForbid

from ..enums import DataType


class Output(BaseModelExtraForbid):
    """Represents output stream of a model.

    @type name: str
    @ivar name: Name of the output layer.
    @type dtype: DataType
    @ivar dtype: Data type of the output data (e.g., 'float32').
    """

    name: str = Field(description="Name of the output layer.")
    dtype: DataType = Field(
        description="Data type of the output data (e.g., 'float32')."
    )
