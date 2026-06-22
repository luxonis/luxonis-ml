from enum import Enum


class DatasetType(str, Enum):
    """Dataset import/export format identifiers.

    Attributes:
        COCO: COCO dataset format.
        VOC: Pascal VOC dataset format.
        DARKNET: Darknet dataset format.
        YOLOV6: YOLOv6 dataset format.
        YOLOV4: YOLOv4 dataset format.
        CREATEML: Apple Create ML dataset format.
        TFCSV: TensorFlow CSV dataset format.
        CLSDIR: Classification directory dataset format.
        SEGMASK: Segmentation mask directory format.
        SOLO: Unity SOLO dataset format.
        NATIVE: Native Luxonis Data Format.
        YOLOV8BOUNDINGBOX: YOLOv8 bounding-box format.
        YOLOV8INSTANCESEGMENTATION: YOLOv8 instance-segmentation format.
        YOLOV8KEYPOINTS: YOLOv8 keypoints format.
        ULTRALYTICSNDJSON: Ultralytics NDJSON detection format.
        ULTRALYTICSNDJSONINSTANCESEGMENTATION: Ultralytics NDJSON
            instance-segmentation format.
        ULTRALYTICSNDJSONKEYPOINTS: Ultralytics NDJSON keypoints format.
        FIFTYONECLS: FiftyOne classification format.

    """

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
    ULTRALYTICSNDJSON = "ultralytics-ndjson"
    ULTRALYTICSNDJSONINSTANCESEGMENTATION = (
        "ultralytics-ndjson-instancesegmentation"
    )
    ULTRALYTICSNDJSONKEYPOINTS = "ultralytics-ndjson-keypoints"
    FIFTYONECLS = "fiftyone-classification"

    @property
    def supported_annotation_formats(self) -> list[str]:
        return _SUPPORTED_ANNOTATION_FORMATS[self]


_SUPPORTED_ANNOTATION_FORMATS: dict[DatasetType, list[str]] = {
    DatasetType.COCO: [
        "boundingbox",
        "instance_segmentation",
        "keypoints",
    ],
    DatasetType.VOC: ["boundingbox"],
    DatasetType.DARKNET: ["boundingbox"],
    DatasetType.YOLOV6: ["boundingbox"],
    DatasetType.YOLOV4: ["boundingbox"],
    DatasetType.CREATEML: ["boundingbox"],
    DatasetType.TFCSV: ["boundingbox"],
    DatasetType.CLSDIR: ["classification"],
    DatasetType.SEGMASK: ["segmentation"],
    DatasetType.SOLO: [
        "boundingbox",
        "segmentation",
        "instance_segmentation",
        "keypoints",
    ],
    DatasetType.NATIVE: [
        "boundingbox",
        "segmentation",
        "instance_segmentation",
        "keypoints",
        "classification",
        "labels/text",
    ],
    DatasetType.YOLOV8BOUNDINGBOX: ["boundingbox"],
    DatasetType.YOLOV8INSTANCESEGMENTATION: ["instance_segmentation"],
    DatasetType.YOLOV8KEYPOINTS: ["keypoints"],
    DatasetType.ULTRALYTICSNDJSON: ["boundingbox"],
    DatasetType.ULTRALYTICSNDJSONINSTANCESEGMENTATION: [
        "instance_segmentation"
    ],
    DatasetType.ULTRALYTICSNDJSONKEYPOINTS: ["keypoints"],
    DatasetType.FIFTYONECLS: ["classification"],
}
