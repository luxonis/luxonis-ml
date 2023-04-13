from enum import Enum

class DatasetType(Enum):
    LDF = "LDF"
    COCO = "COCO"
    CDT = "ClassificationDirectoryTree"
    CTA = "ClassificationWithTextAnnotations"
    FOD = "FiftyOneDetection"
    CML = "CreateML"
    VOC = "VOC"
    YOLO4 = "YOLO4"
    YOLO5 = "YOLO5"
    TFODD = "TFObjectDetectionDataset"
    TFODC = "TFObjectDetectionCSV"
    YAML = "YAML"
    UNKNOWN = "unknown"