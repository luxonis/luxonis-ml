from enum import Enum


class DataLabelType(Enum):
    """Supported computer vision label types.

    Annotation types can be nested (I{e.g.} a BOX has 2 LABELS, a BOX has a POLYLINE
    instance segmentation, I{etc.})
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

    COLOR = "color"  # 3 channel BGR or RGB (uint8)
    MONO = "mono"  # 1 channel (uint8)
    DISPARITY = "disparity"  # disparity or depth (uint16)


class BucketType(Enum):
    """Whether bucket storage is internal to Luxonis or not."""

    INTERNAL = "internal"
    EXTERNAL = "external"


class BucketStorage(Enum):
    """Underlying object storage for a bucket."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure"
