from enum import Enum

class DecodingFamily(Enum):

    """ members of a family should have similar decoding """

    YOLO = "yolo"
    # TODO: add additional families
    RAW = "raw"

class DecodingSubFamily(Enum):

    """ members of a subfamily should have exactly the same decoding """

    # postprocessing for YOLOs follows the same principle - running NMS on model output
    # however, outputs are structured a bit differently:
    YOLOv5_7 = "yolov5/yolov7" # both (1, 3, 80, 80, 85), (1, 3, 40, 40, 85), (1, 3, 20, 20, 85) shape outputs, as the (1, 25200, 85) concatenate
    YOLOv6 = "yolov6" # (1, 80, 80, 85), (1, 40, 40, 85), (1, 20, 20, 85) shape outputs
    #YOLOv3?
    
    # TODO: add additional sub-families
    RAW = "raw"