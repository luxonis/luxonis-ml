"""Exporters that convert LDF datasets to external formats.

This package contains exporter implementations for writing Luxonis Data Format
(LDF) datasets into common annotation formats used by training frameworks,
dataset tools, and interchange workflows. Exporters operate on prepared LDF
records and write format-specific image references, annotation files, labels,
and metadata as required by the target format.

`BaseExporter` defines the exporter interface and shared dataset-selection
behavior. `PreparedLDF` resolves the dataset dataframe, split mapping, class
metadata, and media paths used by concrete exporters.

.. list-table:: Export targets
   :header-rows: 1

   * - Exporter
     - Target format
   * - `CocoExporter`
     - COCO detection, segmentation, and keypoint annotations.
   * - `YoloV8Exporter`, `YoloV8InstanceSegmentationExporter`,
       `YoloV8KeypointsExporter`
     - Ultralytics YOLOv8 task formats.
   * - `YoloV6Exporter`, `YoloV4Exporter`, `DarknetExporter`
     - YOLO-family text-label formats.
   * - `VOCExporter`, `CreateMLExporter`, `TensorflowCSVExporter`
     - VOC XML, CreateML JSON, and TensorFlow CSV annotations.
   * - `ClassificationDirectoryExporter`,
       `FiftyOneClassificationExporter`
     - Classification datasets organized for directory or FiftyOne-style
       workflows.
   * - `SegmentationMaskDirectoryExporter`
     - Semantic segmentation masks stored as image files.
   * - `NativeExporter`, `UltralyticsNDJSONExporter`
     - Native LDF and Ultralytics NDJSON interchange formats.
"""

from .base_exporter import BaseExporter
from .classification_directory_exporter import ClassificationDirectoryExporter
from .coco_exporter import CocoExporter
from .createml_exporter import CreateMLExporter
from .darknet_exporter import DarknetExporter
from .exporter_utils import PreparedLDF
from .fiftyone_classification_exporter import FiftyOneClassificationExporter
from .native_exporter import NativeExporter
from .segmentation_mask_directory_exporter import (
    SegmentationMaskDirectoryExporter,
)
from .tensorflow_csv_exporter import TensorflowCSVExporter
from .ultralytics_ndjson_exporter import UltralyticsNDJSONExporter
from .voc_exporter import VOCExporter
from .yolov4_exporter import YoloV4Exporter
from .yolov6_exporter import YoloV6Exporter
from .yolov8_bbox_exporter import YoloV8Exporter
from .yolov8_instance_segmentation_exporter import (
    YoloV8InstanceSegmentationExporter,
)
from .yolov8_keypoints_exporter import YoloV8KeypointsExporter

__all__ = [
    "BaseExporter",
    "ClassificationDirectoryExporter",
    "CocoExporter",
    "CreateMLExporter",
    "DarknetExporter",
    "FiftyOneClassificationExporter",
    "NativeExporter",
    "PreparedLDF",
    "SegmentationMaskDirectoryExporter",
    "TensorflowCSVExporter",
    "UltralyticsNDJSONExporter",
    "VOCExporter",
    "YoloV4Exporter",
    "YoloV6Exporter",
    "YoloV8Exporter",
    "YoloV8InstanceSegmentationExporter",
    "YoloV8KeypointsExporter",
]
