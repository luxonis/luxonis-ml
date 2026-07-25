"""Luxonis Data Format (LDF) annotation models.

This lightweight package owns the record and annotation payload contracts of the
Luxonis Data Format: `DatasetRecord`, `Detection`, and the annotation payload
schemas (`BBoxAnnotation`, `KeypointAnnotation`, `SegmentationAnnotation`,
`InstanceSegmentationAnnotation`, `ArrayAnnotation`, `ClassificationAnnotation`).

It is intentionally dependency-light (pydantic, numpy, pycocotools) so that both
the full data stack (`luxonis_ml.data`) and the visualization engine
(`luxonis_ml.vizlab`) can depend on the same annotation models without pulling
each other's heavy dependencies. `luxonis_ml.data.datasets.annotation` re-exports
these names, so existing imports keep working.
"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("ldf"):
    from .annotation import (
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
    from .parquet import ParquetRecord

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
