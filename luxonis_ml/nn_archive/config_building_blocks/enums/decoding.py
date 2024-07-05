from enum import Enum


class ObjectDetectionSubtypeYOLO(Enum):
    """Object detection decoding subtypes for YOLO networks.

    Subtype members have exactly the same decoding.
    """

    YOLOv5 = "yolov5"
    YOLOv6 = "yolov6"
    YOLOv6r2 = "yolov6r2"
    YOLOv7 = "yolov7"
    YOLOv8 = "yolov8"
    YOLOv10 = "yolov10"


class ObjectDetectionSubtypeSSD(Enum):
    """Object detection decoding subtypes for SSD networks.

    Subtype members have exactly the same decoding.
    """

    SSD_PARSED = "ssd-parsed"
