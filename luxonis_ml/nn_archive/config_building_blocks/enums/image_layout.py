from enum import Enum


class ImageLayout(Enum):
    """Represents the input image layout of the model."""

    HWC = "hwc"
    NHWC = "nhwc"
    CHW = "chw"
    NCHW = "nchw"
