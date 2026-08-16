"""Re-exports of the LDF annotation schemas.

The models live in `luxonis_ml.ldf.annotation`; this module keeps imports such
as ``from luxonis_ml.data.datasets.annotation import Detection`` working.
"""

from luxonis_ml.ldf.annotation import (
    Annotation,
    ArrayAnnotation,
    BBoxAnnotation,
    Category,
    ClassificationAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    Keypoint,
    KeypointAnnotation,
    KeypointMetadata,
    KeypointVisibility,
    NormalizedFloat,
    PathOrArray,
    SegmentationAnnotation,
    load_annotation,
)
from luxonis_ml.ldf.parquet import ParquetRecord

__all__ = [
    "Annotation",
    "ArrayAnnotation",
    "BBoxAnnotation",
    "Category",
    "ClassificationAnnotation",
    "DatasetRecord",
    "Detection",
    "InstanceSegmentationAnnotation",
    "Keypoint",
    "KeypointAnnotation",
    "KeypointMetadata",
    "KeypointVisibility",
    "NormalizedFloat",
    "ParquetRecord",
    "PathOrArray",
    "SegmentationAnnotation",
    "load_annotation",
]
