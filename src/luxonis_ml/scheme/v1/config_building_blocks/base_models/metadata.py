from pydantic import BaseModel
from ..enums import *

class Metadata(BaseModel):
    """
    Represents metadata of a model.

    Attributes:
        name (str): Name of the model.
        platform (Platform): Luxonis hardware platform for which the model was exported (e.g. 'rvc4').

    """
    name: str
    platform: Platform