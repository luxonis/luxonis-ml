r"""Parsers that convert external dataset formats to LDF.

This package owns the list of supported import formats. Additions or changes
to parser support should be documented here, alongside the parser classes that
implement them.

The high-level `LuxonisParser` dispatcher accepts local directories, remote
paths supported by ``LuxonisFileSystem``, ZIP archives, and Roboflow URLs in
``roboflow://workspace/project/version/format`` form. It can auto-detect
supported layouts or use an explicit ``DatasetType``, then delegates to the
matching parser implementation.

.. contents:: Table of Contents
   :depth: 2


Basic Usage
===========

.. python::

    from luxonis_ml.data import LuxonisParser
    from luxonis_ml.enums import DatasetType

    parser = LuxonisParser(
        "path/to/dataset",
        dataset_name="parking_lot",
        dataset_type=DatasetType.COCO,
        task_name="detection",
    )

    dataset = parser.parse()

When ``dataset_type`` is omitted, `LuxonisParser` tries to infer the dataset
format from the directory structure. ``task_name`` may be a single string used
for all records or a mapping from class names to task names.

.. python::

    parser = LuxonisParser(
        "path/to/person_dataset",
        task_name={
            "head": "head_pose",
            "neck": "head_pose",
            "torso": "body_pose",
            "leg": "body_pose",
        },
    )

Note:
    When parsing ZIP files, place the dataset layout directly at the archive
    root unless the selected parser explicitly expects a nested directory.

Warning:
    Format-specific parsers do not follow symbolic links. Datasets that rely
    on symlinked images or labels may not parse as expected.


Supported Formats
=================

.. list-table:: Supported parser formats
   :header-rows: 1

   * - Format
     - Dataset type
     - Parser
     - Typical annotations
   * - COCO JSON
     - ``DatasetType.COCO``
     - `COCOParser`
     - Bounding boxes, segmentation, instance segmentation, keypoints.
   * - Pascal VOC XML
     - ``DatasetType.VOC``
     - `VOCParser`
     - Bounding boxes.
   * - YOLO Darknet TXT
     - ``DatasetType.DARKNET``
     - `DarknetParser`
     - Bounding boxes.
   * - YOLOv4 PyTorch TXT
     - ``DatasetType.YOLOV4``
     - `YoloV4Parser`
     - Bounding boxes.
   * - MT YOLOv6
     - ``DatasetType.YOLOV6``
     - `YoloV6Parser`
     - Bounding boxes.
   * - YOLOv8 bounding boxes
     - ``DatasetType.YOLOV8BOUNDINGBOX``
     - `YOLOv8Parser`
     - Bounding boxes.
   * - YOLOv8 instance segmentation
     - ``DatasetType.YOLOV8INSTANCESEGMENTATION``
     - `YOLOv8Parser`
     - Instance segmentation.
   * - YOLOv8 keypoints
     - ``DatasetType.YOLOV8KEYPOINTS``
     - `YOLOv8Parser`
     - Keypoints.
   * - Ultralytics NDJSON detection
     - ``DatasetType.ULTRALYTICSNDJSON``
     - `UltralyticsNDJSONParser`
     - Detection records, including local paths or remote image URLs.
   * - Ultralytics NDJSON instance segmentation
     - ``DatasetType.ULTRALYTICSNDJSONINSTANCESEGMENTATION``
     - `UltralyticsNDJSONParser`
     - Segmentation records.
   * - Ultralytics NDJSON keypoints
     - ``DatasetType.ULTRALYTICSNDJSONKEYPOINTS``
     - `UltralyticsNDJSONParser`
     - Pose records.
   * - CreateML JSON
     - ``DatasetType.CREATEML``
     - `CreateMLParser`
     - Bounding boxes.
   * - TensorFlow Object Detection CSV
     - ``DatasetType.TFCSV``
     - `TensorflowCSVParser`
     - Bounding boxes.
   * - SOLO
     - ``DatasetType.SOLO``
     - `SOLOParser`
     - Synthetic data with boxes, masks, keypoints, and segmentation.
   * - Classification directory
     - ``DatasetType.CLSDIR``
     - `ClassificationDirectoryParser`
     - Class labels encoded by directory names.
   * - FiftyOne classification
     - ``DatasetType.FIFTYONECLS``
     - `FiftyOneClassificationParser`
     - Class labels from ``labels.json``.
   * - Segmentation mask directory
     - ``DatasetType.SEGMASK``
     - `SegmentationMaskDirectoryParser`
     - Grayscale masks with class mappings in ``_classes.csv``.
   * - Native LDF
     - ``DatasetType.NATIVE``
     - ``NativeParser``
     - Existing Luxonis native exports.

See:
    `luxonis_ml.data.datasets.annotation` for the LDF annotation schemas
    produced by these parsers.


Format Layout Notes
===================

The dispatcher can parse full dataset directories, individual split
directories for parser types that support it, ZIP archives whose extracted
root contains a supported layout, remote paths handled by ``LuxonisFileSystem``,
Roboflow URLs in ``roboflow://workspace/project/version/format`` form, and
Ultralytics format URLs in ``ultralytics://username/datasets/slug``.
Parser implementations do not follow symbolic links.

Common layout markers:

    - COCO JSON supports FiftyOne-style splits with ``train/data`` plus
      ``labels.json`` and Roboflow-style splits with images beside
      ``_annotations.coco.json``.
    - YOLOv8-v12 Roboflow layouts use split directories containing
      ``images/`` and ``labels/``; Ultralytics layouts use top-level
      ``images/<split>``, ``labels/<split>``, and a YAML file.
    - Ultralytics NDJSON uses a single ``.ndjson`` manifest. Records may
      reference local image paths or remote image URLs.
    - Pascal VOC XML places images and matching ``.xml`` annotations in each
      split directory.
    - YOLO Darknet uses split directories with image/``.txt`` pairs and
      ``_darknet.labels``.
    - YOLOv4 PyTorch uses ``_annotations.txt`` and ``_classes.txt`` in each
      split directory.
    - MT YOLOv6 uses top-level ``images/<split>``, ``labels/<split>``, and
      ``data.yaml``.
    - CreateML JSON uses ``_annotations.createml.json`` in each split
      directory.
    - TensorFlow Object Detection CSV uses ``_annotations.csv`` in each split
      directory.
    - SOLO expects Unity Perception metadata, sensor, annotation, metric, and
      sequence files per split.
    - Classification directory data may be split-based
      (``train/class_name/*.jpg``) or flat (``class_name/*.jpg``), with
      random splits applied to flat layouts.
    - FiftyOne classification data uses ``data/`` images and ``labels.json``.
      ``labels.json`` contains a ``classes`` list and a mapping from image stem
      to class index.
    - Segmentation mask directories pair images with ``*_mask`` images and
      define pixel-value classes in ``_classes.csv``.


Split Ratio Modes
=================

Parser split ratios support two modes:

    - Floating-point values, such as :math:`0.8`, :math:`0.1`, and
      :math:`0.1`, redistribute and shuffle samples across splits. Values
      should sum to :math:`1.0`.
    - Integer counts, such as :math:`1000`, :math:`100`, and :math:`50`, draw
      from the corresponding original split and preserve split boundaries. If
      a requested count exceeds the available samples, all available samples
      from that split are used.

Example:
    >>> ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    >>> round(sum(ratios.values()), 6)
    1.0

.. code-block:: bash

    luxonis_ml data parse ./dataset --name parking_lot --type coco
    luxonis_ml data parse ./dataset --split-ratio 0.8,0.1,0.1
    luxonis_ml data parse ./dataset --split-ratio 1000,100,50


COCO Keypoints
==============

For COCO-2017 style FiftyOne exports, bounding boxes and segmentations often
come from instance annotation files while person keypoints live in dedicated
``person_keypoints_*.json`` files. Use ``use_keypoint_ann=True`` when parsing
with the Python API:

.. python::

    parser = LuxonisParser("coco-2017", dataset_name="coco_keypoints")
    dataset = parser.parse(
        use_keypoint_ann=True,
        split_ratios={"train": 0.5, "val": 0.4, "test": 0.1},
    )

Parser issues that are skipped or recovered during parsing are reported as
`ParserIssueMessage` instances and categorized by `ParserIssue`.


Evaluation Dataset Notes
========================

COCO-2017
---------

COCO parsing handles both FiftyOne and Roboflow layouts. Bounding boxes are
normalized relative to image dimensions; polygon or RLE segmentations are
stored as RLE; instance segmentation is emitted from the same segmentation
source; keypoints are normalized and clipped; category identifiers are mapped
to class names.

For FiftyOne COCO exports, the standard ``labels.json`` usually contains
instance annotations for the 80 COCO categories but not person keypoints. Use
``use_keypoint_ann=True`` with the Python API to read dedicated
``raw/person_keypoints_train2017.json`` and
``raw/person_keypoints_val2017.json`` files. If test keypoints are missing,
``split_val_to_test=True`` splits validation samples into validation and test
sets. Roboflow COCO layouts ignore the keypoint-specific options.

The COCO parser also filters known corrupted COCO-2017 train images and can
write cleaned annotation files when source metadata requires repair.

ImageNet Sample
---------------

The ImageNet-sample parser handles FiftyOne image classification exports in
flat ``data/`` plus ``labels.json`` form or in split-based
``train/validation/test`` directories. Flat layouts are split randomly at parse
time. ``labels.json`` contains ``classes`` and ``labels`` keys, where
``labels`` maps image stems to class indices.

Known ImageNet-sample label issues are cleaned automatically: duplicate
``"crane"`` and ``"maillot"`` class names are disambiguated, and known
misindexed labels for images ``006742`` and ``031933`` are corrected. A
``labels_fixed.json`` file is saved next to the original labels.

ImageNet-2012
-------------

The original ImageNet-2012 archive layout is not directly parsed. Extract the
train and validation archives, group training images by class, use the devkit
metadata to map validation images to class labels, move validation images into
class folders, and parse the result as ``DatasetType.CLSDIR``.

"""

from luxonis_ml.data.utils.enums import ParserIssue, ParserIssueMessage

from .base_parser import BaseParser
from .classification_directory_parser import ClassificationDirectoryParser
from .coco_parser import COCOParser
from .create_ml_parser import CreateMLParser
from .darknet_parser import DarknetParser
from .fiftyone_classification_parser import FiftyOneClassificationParser
from .luxonis_parser import LuxonisParser
from .native_parser import NativeParser
from .segmentation_mask_directory_parser import SegmentationMaskDirectoryParser
from .solo_parser import SOLOParser
from .tensorflow_csv_parser import TensorflowCSVParser
from .ultralytics_ndjson_parser import UltralyticsNDJSONParser
from .voc_parser import VOCParser
from .yolov4_parser import YoloV4Parser
from .yolov6_parser import YoloV6Parser
from .yolov8_parser import YOLOv8Parser

__all__ = [
    "BaseParser",
    "COCOParser",
    "ClassificationDirectoryParser",
    "CreateMLParser",
    "DarknetParser",
    "FiftyOneClassificationParser",
    "LuxonisParser",
    "NativeParser",
    "ParserIssue",
    "ParserIssueMessage",
    "SOLOParser",
    "SegmentationMaskDirectoryParser",
    "TensorflowCSVParser",
    "UltralyticsNDJSONParser",
    "VOCParser",
    "YOLOv8Parser",
    "YoloV4Parser",
    "YoloV6Parser",
]
