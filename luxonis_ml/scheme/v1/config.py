from typing import List
from pydantic import Field
from .config_building_blocks import CustomBaseModel
from model import Model
from __init__ import CONFIG_VERSION


class Config(CustomBaseModel):
    """The main class of the multi/single-stage model config scheme (multi- stage models
    consists of interconnected single-stage models).

    @type config_version: str
    @ivar config_version: Static variable representing the version of the config scheme.
    @type stages: list
    @ivar stages: List of Model objects each representing a stage in the model (list of
        one element for single-stage models).
    @type connections: list
    @ivar connections: List of connections instructing how to connect multi stage models
        (empty for single-stage models).
    """

    config_version: str = Field(CONFIG_VERSION, Literal=True)
    stages: List[Model]
    connections: List = []  # TODO: To be implemented
