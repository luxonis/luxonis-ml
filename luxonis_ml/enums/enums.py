from enum import Enum


class LabelType(str, Enum):
    CLASSIFICATION = "class"
    SEGMENTATION = "segmentation"
    BOUNDINGBOX = "boxes"
    KEYPOINT = "keypoints"


class DatasetType(str, Enum):
    LDF = "ldf"
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


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
