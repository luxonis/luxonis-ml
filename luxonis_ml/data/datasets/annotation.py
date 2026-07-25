"""Backward-compatibility shim for the LDF annotation schemas.

The annotation record and payload models were extracted into the lightweight
`luxonis_ml.ldf` package so that both `luxonis_ml.data` and `luxonis_ml.vizlab`
can share the same annotation contracts without pulling each other's heavy
dependencies. This module re-exports them, so existing imports such as
``from luxonis_ml.data.datasets.annotation import Detection`` keep working.

See `luxonis_ml.ldf.annotation` for the full documentation and examples.
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
    KeypointAnnotation,
    KeypointVisibility,
    NormalizedFloat,
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
    "KeypointAnnotation",
    "KeypointVisibility",
    "NormalizedFloat",
    "ParquetRecord",
    "SegmentationAnnotation",
    "load_annotation",
]
