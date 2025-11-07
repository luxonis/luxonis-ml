from enum import Enum


class DatasetType(str, Enum):
    COCO = "coco"
    VOC = "voc"
    DARKNET = "darknet"
    YOLOV6 = "yolov6"
    YOLOV4 = "yolov4"
    CREATEML = "createml"  # todo
    TFCSV = "tfcsv"  # todo
    CLSDIR = "clsdir"
    SEGMASK = "segmask"
    SOLO = "solo"
    NATIVE = "native"
    YOLOV8 = "yolov8"  # to add segmentation and keypoints
