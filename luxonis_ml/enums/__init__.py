"""Shared enumerations used by Luxonis ML public APIs.

The `luxonis_ml.enums` package exports `DatasetType`, the common dataset
format identifier used by parsers, exporters, CLI commands, and dataset
workflows. Each value also exposes
`DatasetType.supported_annotation_formats`, which describes the annotation
families supported by that external format.

Example:
    Use the package import when selecting a parser or exporter format.

    >>> from luxonis_ml.enums import DatasetType
    >>> DatasetType.COCO.value
    'coco'
    >>> "boundingbox" in DatasetType.COCO.supported_annotation_formats
    True

See:
    `luxonis_ml.enums.enums` for the complete enum definition and
    annotation-format mapping.

"""

from .enums import DatasetType

__all__ = ["DatasetType"]
