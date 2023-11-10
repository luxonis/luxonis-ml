from pydantic import BaseModel
from ..enums import *

class Metadata(BaseModel):
    """
    Represents metadata for a model.

    Attributes:
        name (str): Name of the model.
        platform (Platform): Luxonis hardware platform for which the model was exported (e.g. 'rvc4').
        config_version (int): Version of the config schema.

    """
    name: str
    platform: Platform
    config_version: int