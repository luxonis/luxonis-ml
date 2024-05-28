from enum import Enum


class ImageLayout(Enum):
    """Represents the input image layout."""

    HWC = "hwc"
    CHW = "chw"
