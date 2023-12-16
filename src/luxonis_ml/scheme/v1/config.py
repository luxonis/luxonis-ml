from typing import List
from pydantic import Field
from config_building_blocks import *
from model import Model
from __init__ import CONFIG_VERSION


class Config(CustomBaseModel):
    """The main class of the multi/single-stage model config scheme (multi-
    stage models consists of interconnected single-stage models).

    Attributes:
        config_version (str): Static variable representing the version of the config scheme.
        stages (list): List of Model objects each representing a stage in the model (list of one element for single-stage models).
        connections (list): List of connections instructing how to connect multi stage models (empty for single-stage models).
    """

    config_version: str = Field(CONFIG_VERSION, Literal=True)
    stages: List[Model]
    connections: List = []  # TODO: To be implemented
