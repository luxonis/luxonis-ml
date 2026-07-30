"""Synthetic dataset builders, one module per parser.

Importing this package registers every builder in
`harness.BENCHMARK_DATASETS`.
"""

from tests.test_data.parsers.benchmarks.harness import BENCHMARK_DATASETS

from . import (
    classification_directory,
    coco,
    create_ml,
    darknet,
    fiftyone_classification,
    native,
    segmentation_mask_directory,
    solo,
    tensorflow_csv,
    ultralytics_ndjson,
    voc,
    yolov4,
    yolov6,
    yolov8,
)

__all__ = [
    "BENCHMARKED_TYPES",
    "classification_directory",
    "coco",
    "create_ml",
    "darknet",
    "fiftyone_classification",
    "native",
    "segmentation_mask_directory",
    "solo",
    "tensorflow_csv",
    "ultralytics_ndjson",
    "voc",
    "yolov4",
    "yolov6",
    "yolov8",
]

BENCHMARKED_TYPES = sorted(BENCHMARK_DATASETS)
