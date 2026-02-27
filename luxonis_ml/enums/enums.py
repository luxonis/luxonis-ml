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
    FIFTYONECLS = "fiftyone-classification"

    @property
    def supported_annotation_formats(self) -> tuple[str, ...]:
        return _SUPPORTED_ANNOTATION_FORMATS[self]


_SUPPORTED_ANNOTATION_FORMATS: dict[DatasetType, tuple[str, ...]] = {
    DatasetType.COCO: (
        "boundingbox",
        "instance_segmentation",
        "keypoints",
    ),
    DatasetType.VOC: ("boundingbox",),
    DatasetType.DARKNET: ("boundingbox",),
    DatasetType.YOLOV6: ("boundingbox",),
    DatasetType.YOLOV4: ("boundingbox",),
    DatasetType.CREATEML: ("boundingbox",),
    DatasetType.TFCSV: ("boundingbox",),
    DatasetType.CLSDIR: ("classification",),
    DatasetType.SEGMASK: ("segmentation",),
    DatasetType.SOLO: (
        "boundingbox",
        "segmentation",
        "instance_segmentation",
        "keypoints",
    ),
    DatasetType.NATIVE: (
        "boundingbox",
        "segmentation",
        "instance_segmentation",
        "keypoints",
        "classification",
        "metadata/text"
    ),
    DatasetType.YOLOV8BOUNDINGBOX: ("boundingbox",),
    DatasetType.YOLOV8INSTANCESEGMENTATION: ("instance_segmentation",),
    DatasetType.YOLOV8KEYPOINTS: ("keypoints",),
    DatasetType.FIFTYONECLS: ("classification",),
}
