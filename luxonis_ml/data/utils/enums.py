from dataclasses import dataclass
from enum import Enum

from luxonis_ml.typing import PathType


class MediaType(Enum):
    """Individual file type.

    Attributes:
        IMAGE: Image media.
        VIDEO: Video media.
        POINTCLOUD: Point-cloud media.

    """

    IMAGE = "image"
    VIDEO = "video"
    POINTCLOUD = "pointcloud"


class ImageType(Enum):
    """Image type for image media.

    Attributes:
        COLOR: Color image, typically :math:`3` channels.
        MONO: Monochrome image with :math:`1` channel.
        DISPARITY: Disparity or depth image.

    """

    COLOR = "color"  # 3 channel BGR or RGB (uint8)
    MONO = "mono"  # 1 channel (uint8)
    DISPARITY = "disparity"  # disparity or depth (uint16)


class BucketType(Enum):
    """Whether bucket storage is internal to Luxonis.

    Attributes:
        INTERNAL: Bucket managed by Luxonis.
        EXTERNAL: User-provided bucket.

    """

    INTERNAL = "internal"
    EXTERNAL = "external"


class BucketStorage(Enum):
    """Underlying object storage for a bucket.

    Attributes:
        LOCAL: Local filesystem storage.
        S3: Amazon S3-compatible storage.
        GCS: Google Cloud Storage.
        AZURE_BLOB: Azure Blob Storage.

    """

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure"


class UpdateMode(Enum):
    """Dataset media update mode.

    Attributes:
        ALL: Update all files.
        MISSING: Update only missing files.

    """

    ALL = "all"
    MISSING = "missing"


class COCOFormat(str, Enum):
    """Supported COCO directory layouts.

    Attributes:
        FIFTYONE: COCO layout exported by FiftyOne.
        ROBOFLOW: COCO layout exported by Roboflow.

    """

    FIFTYONE = "fiftyone"
    ROBOFLOW = "roboflow"


class ParserIssue(Enum):
    """Parser issue categories reported during best-effort parsing.

    Attributes:
        MISSING_IMAGE: Annotation references an image that does not exist.
        COCO_ISCROWD: COCO annotation is marked with ``iscrowd=1``.
        NON_NUMERIC_ANNOTATION: Annotation contains non-numeric values.
        MISSING_IMAGE_STEM: Annotation image stem cannot be matched to an
            image file.

    """

    MISSING_IMAGE = "missing_image"
    COCO_ISCROWD = "coco_iscrowd"
    NON_NUMERIC_ANNOTATION = "non_numeric_annotation"
    MISSING_IMAGE_STEM = "missing_image_stem"


@dataclass(frozen=True, slots=True)
class ParserIssueMessage:
    """Structured message describing a skipped parser item.

    Attributes:
        parser_issue: Issue category.
        reason: Human-readable reason.
        source: Optional source annotation path.
        image: Optional image path involved in the issue.
        annotation_id: Optional annotation identifier.

    """

    parser_issue: ParserIssue
    reason: str
    source: PathType | None = None
    image: PathType | None = None
    annotation_id: str | int | None = None
