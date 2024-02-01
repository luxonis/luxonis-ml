from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Represents metadata of a model.

    @type name: str
    @ivar name: Name of the model.
    @type path: str
    @ivar path: Relative path to the model executable.
    """

    name: str = Field(description="Name of the model.")
    path: str = Field(description="Relative path to the model executable.")
