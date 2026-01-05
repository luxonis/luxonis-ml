from enum import Enum


class DatasetType(str, Enum):
    COCO = "coco"
    VOC = "voc"
    DARKNET = "darknet"
    YOLOV6 = "yolov6"
    YOLOV4 = "yolov4"
    CREATEML = "createml"
    TFCSV = "tfcsv"
    CLSDIR = "clsdir"
    SEGMASK = "segmask"
    SOLO = "solo"
    NATIVE = "native"
    YOLOV8BOUNDINGBOX = "yolov8"
    YOLOV8INSTANCESEGMENTATION = "yolov8instancesegmentation"
    YOLOV8KEYPOINTS = "yolov8keypoints"
    FIFTYONECLASSIFICATION = "fiftyoneclassification"
