import re

from pydantic import Field, field_validator

from luxonis_ml.utils import BaseModelExtraForbid

from .model import Model

CONFIG_VERSION = "1.0"


class Config(BaseModelExtraForbid):
    """The main class of the multi/single-stage model config scheme
    (multi- stage models consists of interconnected single-stage
    models).

    @type config_version: str
    @ivar config_version: String representing config schema version in
        format 'x.y' where x is major version and y is minor version
    @type model: Model
    @ivar model: A Model object representing the neural network used in
        the archive.
    """

    config_version: str = Field(
        CONFIG_VERSION,
        description="String representing config schema version in format 'x.y' where x is major version and y is minor version.",
    )
    model: Model = Field(
        description="A Model object representing the neural network used in the archive."
    )

    @field_validator("config_version")
    @classmethod
    def validate_config_version_format(cls, v: str) -> str:
        # Regular expression to match 'x.y' where x and y are integers
        if not re.match(r"^\d+\.\d+$", v):
            raise ValueError(
                "'config_version' must be in format 'x.y' where x and y are integers"
            )
        return v
