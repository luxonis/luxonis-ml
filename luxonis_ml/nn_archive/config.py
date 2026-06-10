import re

from pydantic import Field, field_validator

from luxonis_ml.typing import BaseModelExtraForbid

from .model import Model

CONFIG_VERSION = "1.0"


class Config(BaseModelExtraForbid):
    """Configuration schema for an NN Archive.

    Attributes:
        config_version: Schema version in ``major.minor`` format.
        model: Neural network configuration stored in the archive.

    """

    config_version: str = Field(
        CONFIG_VERSION,
        description="Schema version in 'major.minor' format.",
    )
    model: Model = Field(
        description="Neural network configuration stored in the archive."
    )

    @field_validator("config_version")
    @classmethod
    def validate_config_version_format(cls, v: str) -> str:
        """Validate that the schema version uses ``major.minor``
        format.
        """
        # Regular expression to match 'x.y' where x and y are integers.
        if not re.match(r"^\d+\.\d+$", v):
            raise ValueError(
                "'config_version' must be in format 'x.y' where x and y are integers"
            )
        return v
