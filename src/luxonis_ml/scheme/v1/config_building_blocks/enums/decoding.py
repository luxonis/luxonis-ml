from enum import Enum

class DecodingFamily(Enum):

    """ members of a family should have similar decoding """

    # classification
    MULTICLASS_CLASSIFICATION= "multiclass_classification"

    # object detection
    YOLO = "yolo"
    SSD_PARSED = "ssd-mobilenet-parsed" # seems like outputs of all ssd-mobilenet networks are of shape (1,3000,classes_n) and (1,3000,4)
    
    # keypoints
    KEYPOINTS_HEATMAP = "heatmap"
    KEYPOINTS_REGRESSION = "regression"
    KEYPOINTS_ANCHORS = "anchors"
    
    # segmentation
    # TODO

    # misc
    RAW = "raw"

class ObjectDetectionSubtype(Enum):

    """ subtype members have exactly the same decoding """

    YOLOv5 = "yolov5"
    YOLOv6 = "yolov6"
    YOLOv7 = "yolov6"
    YOLOv8 = "yolov6"