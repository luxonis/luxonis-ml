from dataclasses import dataclass
from enum import Enum

from luxonis_ml.typing import PathType


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

    ALL = "all"
    MISSING = "missing"


class COCOFormat(str, Enum):
    FIFTYONE = "fiftyone"
    ROBOFLOW = "roboflow"


class ParserIssue(Enum):
    MISSING_IMAGE = "missing_image"
    COCO_ISCROWD = "coco_iscrowd"
    NON_NUMERIC_ANNOTATION = "non_numeric_annotation"
    MISSING_IMAGE_STEM = "missing_image_stem"


@dataclass(frozen=True, slots=True)
class ParserIssueMessage:
    parser_issue: ParserIssue
    reason: str
    source: PathType | None = None
    image: PathType | None = None
    annotation_id: str | int | None = None
