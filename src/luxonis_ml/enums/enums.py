from enum import Enum


class LabelType(str, Enum):
    CLASSIFICATION = "class"
    SEGMENTATION = "segmentation"
    BOUNDINGBOX = "boxes"
    KEYPOINT = "keypoints"
