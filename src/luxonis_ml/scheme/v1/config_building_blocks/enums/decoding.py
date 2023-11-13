from enum import Enum

class ObjectDetectionSubtype(Enum):

    """ 
    Represents object detection subtypes. Subtype members have exactly the same decoding. 
    """

    YOLOv5 = "yolov5"
    YOLOv6 = "yolov6"
    YOLOv7 = "yolov6"
    YOLOv8 = "yolov6"

    SSD_PARSED = "ssd-mobilenet-parsed"