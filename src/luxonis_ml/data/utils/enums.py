from enum import Enum


class ModelTasks(Enum):
    """Supported computer vision label types.
    Annotation types can be nested
    (e.g. a BOX has 2 LABELS,
        a BOX has a POLYLINE instance segmentation,
        etc.)"""

    CLASSIFICATION = "classification"  # label for image classification
    BOX = "box"  # bounding box
    LABEL = "label"  # key value pair of an arbitrary label name and it's value
    POLYLINE = "polyline"  # polyline to represent segmentation mask instances
    KEYPOINTS = "keypoints"  # keypoint skeleton instances


class MediaType(Enum):
    """Individual file type"""

    IMAGE = "image"
    VIDEO = "video"
    POINTCLOUD = "point cloud"


class ImageType(Enum):
    """Image type for IMAGE HType"""

    COLOR = "color"
    MONO = "mono"
    DISPARITY = "disparity"
    DEPTH = "depth"


class LDFTransactionType(Enum):
    """The type of transaction"""

    END = "END"
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class BucketType(Enum):
    """Whether storage is internal or external"""

    INTERNAL = "internal"
    EXTERNAL = "external"


class BucketStorage(Enum):
    """Underlying object storage for a bucket"""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure"
