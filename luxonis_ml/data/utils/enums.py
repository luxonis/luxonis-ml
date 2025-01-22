from enum import Enum


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


class UpdateMode(Enum):
    """Update mode for the dataset."""

    ALWAYS = "always"
    IF_EMPTY = "if_empty"
