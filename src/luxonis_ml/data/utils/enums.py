from enum import Enum


class DataLabelType(Enum):
    """Supported computer vision label types.

    Annotation types can be nested (e.g. a BOX has 2 LABELS,     a BOX
    has a POLYLINE instance segmentation,     etc.)
    """

    CLASSIFICATION = (
        "classification"  # used for single, multi-class, or multi-label classification
    )
    BOX = "box"  # bounding box
    POLYLINE = "polyline"  # polyline to represent segmentation mask instances
    SEGMENTATION = "segmentation"  # RLE encoding of a binary segmentation mask
    KEYPOINTS = "keypoints"  # keypoint skeleton instances
    LABEL = "label"  # an arbitrary label of string, bool, or number
    ARRAY = "array"  # a path to a numpy (.npy) array


class MediaType(Enum):
    """Individual file type."""

    IMAGE = "image"
    VIDEO = "video"
    POINTCLOUD = "pointcloud"


class ImageType(Enum):
    """Image type for IMAGE HType."""

    COLOR = "color"
    MONO = "mono"
    DISPARITY = "disparity"


class LDFTransactionType(Enum):
    """The type of transaction."""

    END = "END"
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class BucketType(Enum):
    """Whether storage is internal or external."""

    INTERNAL = "internal"
    EXTERNAL = "external"


class BucketStorage(Enum):
    """Underlying object storage for a bucket."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure"
