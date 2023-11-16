from typing import List
from pydantic import Field
from config_building_blocks import *
from model import Model
from __init__ import CONFIG_VERSION

class Config(CustomBaseModel):
    """
    Main class of the scheme defining a multi/single-stage models (multi-stage models consists of 2 or more interconnected single-stage models).

    Attributes:
        config_version (str): Static variable representing the version of the config schema.
        stages (list): List of Model objects each representing a stage in the model (list of one element for single-stage models).
        connections (list): List of connections instructing how to connect multi stage models (empty for single-stage models).
    """
    config_version: Field(CONFIG_VERSION, const=True)
    stages: List[Model]
    connections: List = [] # TODO: To be implemented