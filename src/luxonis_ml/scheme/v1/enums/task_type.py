from enum import Enum

class TaskType(Enum):

    """ all existing model task types """

    CLASSIFICATION = "classification" # ?
    MULTICLASS_CLASSIFICATION = "multiclass_classification" # e.g. male-blue, female-blue, male-orange, female-orange - 4 labels
    MULTILABEL_CLASSIFICARION = "multilabel_classification" # e.g. male/female and the other blue/orange - 2 labels
    OBJECT_DETECTION = "object_detection" # ?
    SEMANTIC_SEGMENTATION = "semantic_segmentation" # ?
    INSTANCE_SEGMENTATION = "instance segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    MISC = "misc"
    RAW = "raw"