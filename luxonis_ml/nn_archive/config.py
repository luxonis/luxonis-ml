from typing import Literal

from pydantic import Field

from .config_building_blocks import CustomBaseModel
from .model import Model

CONFIG_VERSION = Literal["1.0"]


class Config(CustomBaseModel):
    """The main class of the multi/single-stage model config scheme (multi- stage models
    consists of interconnected single-stage models).

    @type config_version: str
    @ivar config_version: Static variable representing the version of the config scheme.
    @type model: Model
    @ivar model: A Model object representing the neural network used in the archive.
    """

    config_version: CONFIG_VERSION = Field(
        ...,
        description="Static variable representing the version of the config scheme.",
    )
    model: Model = Field(
        description="A Model object representing the neural network used in the archive."
    )
