from enum import Enum

class TaskType(Enum):

    """ all existing model task types """

    CLASSIFICATION = "classification"
    #MULTICLASS_CLASSIFICATION = "multiclass_classification"
    #MULTILABEL_CLASSIFICARION = "multilabel_classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    #SEMANTIC_SEGMENTATION = "semantic_segmentation"
    #INSTANCE_SEGMENTATION = "instance segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"
    RAW = "raw"