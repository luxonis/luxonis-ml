from enum import Enum


class InputImageLayout(Enum):
    """Represents the input image layout of the model."""

    HWC = "hwc"
    NHWC = "nhwc"
    CHW = "chw"
    NCHW = "nchw"

