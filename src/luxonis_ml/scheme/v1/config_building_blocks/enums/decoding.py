from enum import Enum

class DecodingFamily(Enum):

    """ members of a family should have similar decoding """

    # classification
    MULTICLASS_CLASSIFICATION= "multiclass_classification"

    # object detection
    YOLO = "yolo"
    SSD_PARSED = "ssd-mobilenet" # seems like outputs of all ssd-mobilenet networks are of shape (1,3000,classes_n) and (1,3000,4)
    
    # keypoints
    KEYPOINTS_HEATMAP = ""
    KEYPOINTS_REGRESSION = ""
    KEYPOINTS_ANCHORS = ""
    
    # segmentation
    # TODO

    # misc
    RAW = "raw"

class DecodingSubFamily(Enum):

    """ members of a subfamily should have exactly the same decoding """

    # classification
    SOFTMAX = "softmax_activation"
    SIGMOID = "sigmoid_activation"

    # object detection
    # postprocessing for YOLOs follows the same principle - running NMS on model output - but the outputs are structured a bit differently:
    YOLOv5_7 = "yolov5/yolov7"
    YOLOv6 = "yolov6"
    
    # keypoints
    # TODO

    # segmentation
    # TODO

    # misc
    RAW = "raw"