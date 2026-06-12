from enum import Enum


class InputType(Enum):
    """Input categories supported by NN Archive model inputs.

    Attributes:
        RAW: Raw tensor input.
        IMAGE: Image input with a channel dimension.

    """

    RAW = "raw"
    IMAGE = "image"
