from pydantic import BaseModel, Field

from luxonis_ml.nn_archive.config_building_blocks.enums import DataType


class Metadata(BaseModel):
    """Model metadata stored in an NN Archive.

    Attributes:
        name: Name of the model.
        path: Relative path to the model executable inside the archive.
        precision: Precision of the model weights.

    """

    name: str = Field(description="Name of the model.")
    path: str = Field(
        description="Relative path to the model executable inside the archive."
    )
    precision: DataType = Field(
        DataType.FLOAT32, description="Precision of the model weights."
    )
