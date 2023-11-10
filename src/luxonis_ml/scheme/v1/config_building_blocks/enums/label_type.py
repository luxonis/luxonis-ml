from enum import Enum

class LabelType(Enum):

    """ 
    Represents all existing label types.
    """

    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    RAW = "raw"