from enum import Enum


class InputType(Enum):
    """Represents a type of input the model is expecting."""

    RAW = "raw"
    IMAGE = "image"
