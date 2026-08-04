"""Annotation schemas of the Luxonis Data Format.

The `luxonis_ml.ldf` package describes a single sample of an LDF dataset:
a `DatasetRecord` with its `Detection`, the annotation payloads
(`BBoxAnnotation`, `KeypointAnnotation`, `SegmentationAnnotation`,
`InstanceSegmentationAnnotation`, `ArrayAnnotation`,
`ClassificationAnnotation`), and the `ParquetRecord` row they are stored as.

Example:
    Build a record with a single bounding box annotation. The referenced
    file has to exist -- `DatasetRecord` validates it and stores it as an
    absolute path.

    .. code-block:: python

        from luxonis_ml.ldf import DatasetRecord

        record = DatasetRecord(
            file="images/frame_001.jpg",
            annotation={
                "class": "person",
                "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            },
        )

Note:
    Install ``luxonis-ml[ldf]`` to work with the format without the rest of
    the data stack. The extra is a subset of ``luxonis-ml[data]``.

See:
    `luxonis_ml.ldf.annotation` for the record and annotation schemas and
    `luxonis_ml.ldf.parquet` for the row they are serialized to.

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
