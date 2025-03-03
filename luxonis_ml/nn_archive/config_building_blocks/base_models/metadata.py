from pydantic import BaseModel, Field

from luxonis_ml.nn_archive.config_building_blocks.enums import DataType


class Metadata(BaseModel):
    """Represents metadata of a model.

    @type name: str
    @ivar name: Name of the model.
    @type path: str
    @ivar path: Relative path to the model executable.
    """

    name: str = Field(description="Name of the model.")
    path: str = Field(description="Relative path to the model executable.")
    precision: DataType = Field(
        DataType.FLOAT32, description="Precision of the model weights."
    )
