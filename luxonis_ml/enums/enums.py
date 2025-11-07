from enum import Enum


class DatasetType(str, Enum):
    COCO = "coco"
    VOC = "voc"  # check precision errors and why they occur
    DARKNET = "darknet"
    YOLOV6 = "yolov6"
    YOLOV4 = "yolov4"
    CREATEML = "createml"
    TFCSV = "tfcsv"
    CLSDIR = "clsdir"
    SEGMASK = "segmask"
    SOLO = "solo"
    NATIVE = "native"
    YOLOV8 = "yolov8"  # add keypoints and instance segmentation
