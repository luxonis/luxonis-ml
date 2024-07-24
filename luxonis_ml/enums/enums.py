from enum import Enum


class DatasetType(str, Enum):
    LDF = "ldf"
    COCO = "coco"
    COCOIMGWITHANN = "cocoimgwithann"
    VOC = "voc"
    DARKNET = "darknet"
    YOLOV6 = "yolov6"
    YOLOV4 = "yolov4"
    CREATEML = "createml"
    TFCSV = "tfcsv"
    CLSDIR = "clsdir"
    SEGMASK = "segmask"
    SOLO = "solo"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
