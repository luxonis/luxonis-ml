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
   * - COCO JSON in the `FiftyOne
       <https://docs.voxel51.com/user_guide/export_datasets.html#cocodetectiondataset-export>`__
       layout or the `Roboflow
       <https://roboflow.com/formats/coco-json>`__ layout
     - ``DatasetType.COCO``
     - `COCOParser`
     - Bounding boxes, segmentation, instance segmentation, keypoints.
   * - `Pascal VOC XML <https://roboflow.com/formats/pascal-voc-xml>`__
     - ``DatasetType.VOC``
     - `VOCParser`
     - Bounding boxes.
   * - `YOLO Darknet TXT <https://roboflow.com/formats/yolo-darknet-txt>`__
     - ``DatasetType.DARKNET``
     - `DarknetParser`
     - Bounding boxes.
   * - `YOLOv4 PyTorch TXT <https://roboflow.com/formats/yolov4-pytorch-txt>`__
     - ``DatasetType.YOLOV4``
     - `YoloV4Parser`
     - Bounding boxes.
   * - `MT YOLOv6 <https://roboflow.com/formats/mt-yolov6>`__
     - ``DatasetType.YOLOV6``
     - `YoloV6Parser`
     - Bounding boxes.
   * - `YOLOv8 bounding boxes
       <https://roboflow.com/formats/yolov8-pytorch-txt>`__
     - ``DatasetType.YOLOV8BOUNDINGBOX``
     - `YOLOv8Parser`
     - Bounding boxes.
   * - `YOLOv8 instance segmentation
       <https://roboflow.com/formats/yolov8-pytorch-txt>`__
     - ``DatasetType.YOLOV8INSTANCESEGMENTATION``
     - `YOLOv8Parser`
     - Instance segmentation.
   * - `YOLOv8 keypoints
       <https://roboflow.com/formats/yolov8-pytorch-txt>`__
     - ``DatasetType.YOLOV8KEYPOINTS``
     - `YOLOv8Parser`
     - Keypoints.
   * - `Ultralytics NDJSON detection
       <https://docs.ultralytics.com/datasets/detect/#ultralytics-ndjson-format>`__
     - ``DatasetType.ULTRALYTICSNDJSON``
     - `UltralyticsNDJSONParser`
     - Detection records, including local paths or remote image URLs.
   * - `Ultralytics NDJSON instance segmentation
       <https://docs.ultralytics.com/datasets/detect/#ultralytics-ndjson-format>`__
     - ``DatasetType.ULTRALYTICSNDJSONINSTANCESEGMENTATION``
     - `UltralyticsNDJSONParser`
     - Segmentation records.
   * - `Ultralytics NDJSON keypoints
       <https://docs.ultralytics.com/datasets/detect/#ultralytics-ndjson-format>`__
     - ``DatasetType.ULTRALYTICSNDJSONKEYPOINTS``
     - `UltralyticsNDJSONParser`
     - Pose records.
   * - `CreateML JSON <https://roboflow.com/formats/createml-json>`__
     - ``DatasetType.CREATEML``
     - `CreateMLParser`
     - Bounding boxes.
   * - `TensorFlow Object Detection CSV
       <https://roboflow.com/formats/tensorflow-object-detection-csv>`__
     - ``DatasetType.TFCSV``
     - `TensorflowCSVParser`
     - Bounding boxes.
   * - `SOLO <https://docs.unity3d.com/Packages/com.unity.perception@1.0/manual/Schema/SoloSchema.html>`__
     - ``DatasetType.SOLO``
     - `SOLOParser`
     - Synthetic data with boxes, masks, keypoints, and segmentation.
   * - Classification directory
     - ``DatasetType.CLSDIR``
     - `ClassificationDirectoryParser`
     - Class labels encoded by directory names.
   * - `FiftyOne classification
       <https://docs.voxel51.com/user_guide/export_datasets.html>`__
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
     - Existing Luxonis native exports, including ``sample_metadata`` records.

See:
    `luxonis_ml.ldf.annotation` for the LDF annotation schemas produced by
    these parsers.


Expected Directory Layouts
==========================

The dispatcher can parse full dataset directories, individual split
directories for parser types that support it, ZIP archives whose extracted
root contains a supported layout, remote paths handled by ``LuxonisFileSystem``,
Roboflow URLs in ``roboflow://workspace/project/version/format`` form, and
Ultralytics format URLs in ``ultralytics://username/datasets/slug``.
Parser implementations do not follow symbolic links.

Each tree below shows one split in full. The other splits repeat the same
structure. Roboflow exports name the second split ``valid``, and FiftyOne
exports name it ``validation``.

COCO JSON
---------

The FiftyOne layout keeps the images in a ``data`` directory and the
annotations in ``labels.json``::

    dataset_dir/
    ├── train/
    │   ├── data/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── labels.json
    ├── validation/
    │   ├── data/
    │   └── labels.json
    └── test/
        ├── data/
        └── labels.json

The Roboflow layout keeps the images beside ``_annotations.coco.json``::

    dataset_dir/
    ├── train/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── ...
    │   └── _annotations.coco.json
    ├── valid/
    └── test/

YOLOv8-v12 and Ultralytics
--------------------------

The Roboflow layout gives each split its own ``images`` and ``labels``
directories::

    dataset_dir/
    ├── train/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img1.txt
    │       ├── img2.txt
    │       └── ...
    ├── valid/
    ├── test/
    └── *.yaml

The `Ultralytics layout <https://docs.ultralytics.com/datasets/>`__ puts the
split directories under one ``images`` directory and one ``labels``
directory::

    dataset_dir/
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   │   ├── img1.txt
    │   │   ├── img2.txt
    │   │   └── ...
    │   ├── val/
    │   └── test/
    └── *.yaml

Ultralytics NDJSON
------------------

The directory holds exactly one ``.ndjson`` manifest. The manifest holds one
record for each image::

    dataset_dir/
    ├── dataset.ndjson
    ├── train/
    ├── val/
    └── test/

A record names a local image through ``file``. A record names a remote image
through ``url``, and ``file`` then gives the name of the local cache file. You
can also give the path of the manifest itself. A manifest of remote images
alone needs no image directories::

    dataset.ndjson

Pascal VOC XML
--------------

Each split holds the images and the matching ``.xml`` files::

    dataset_dir/
    ├── train/
    │   ├── img1.jpg
    │   ├── img1.xml
    │   └── ...
    ├── valid/
    └── test/

YOLO Darknet TXT
----------------

Each split holds image and ``.txt`` pairs, plus ``_darknet.labels``::

    dataset_dir/
    ├── train/
    │   ├── img1.jpg
    │   ├── img1.txt
    │   ├── ...
    │   └── _darknet.labels
    ├── valid/
    └── test/

YOLOv4 PyTorch TXT
------------------

Each split holds ``_annotations.txt`` and ``_classes.txt``::

    dataset_dir/
    ├── train/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── ...
    │   ├── _annotations.txt
    │   └── _classes.txt
    ├── valid/
    └── test/

MT YOLOv6
---------

The split directories sit under one ``images`` directory and one ``labels``
directory, next to ``data.yaml``::

    dataset_dir/
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── valid/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   │   ├── img1.txt
    │   │   ├── img2.txt
    │   │   └── ...
    │   ├── valid/
    │   └── test/
    └── data.yaml

CreateML JSON
-------------

Each split holds ``_annotations.createml.json``::

    dataset_dir/
    ├── train/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── ...
    │   └── _annotations.createml.json
    ├── valid/
    └── test/

TensorFlow Object Detection CSV
-------------------------------

Each split holds ``_annotations.csv``::

    dataset_dir/
    ├── train/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   ├── ...
    │   └── _annotations.csv
    ├── valid/
    └── test/

SOLO
----

Each split holds the Unity Perception definition files and one directory for
each sequence::

    dataset_dir/
    ├── train/
    │   ├── metadata.json
    │   ├── sensor_definitions.json
    │   ├── annotation_definitions.json
    │   ├── metric_definitions.json
    │   └── sequence.<SequenceNUM>/
    │       ├── step<StepNUM>.camera.jpg
    │       ├── step<StepNUM>.frame_data.json
    │       └── step<StepNUM>.camera.semantic segmentation.jpg
    ├── valid/
    └── test/

The semantic segmentation image is optional. ``metadata.json`` declares
``totalSequences``, and the parser warns when the count of sequence
directories does not match it.

Classification Directory
------------------------

One directory holds the images of one class. The split layout keeps the class
directories inside the split directories::

    dataset_dir/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── class2/
    │   └── ...
    ├── valid/
    └── test/

The flat layout puts the class directories in the root. The parser then makes
random splits::

    dataset_dir/
    ├── class1/
    │   ├── img1.jpg
    │   └── ...
    ├── class2/
    │   └── ...
    └── info.json

``info.json`` is optional. It is the only file the root of a flat layout may
hold.

FiftyOne Classification
-----------------------

The images go in a ``data`` directory, and the labels go in ``labels.json``.
The split layout gives each split its own pair::

    dataset_dir/
    ├── train/
    │   ├── data/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── labels.json
    ├── validation/
    │   ├── data/
    │   └── labels.json
    └── test/
        ├── data/
        └── labels.json

The flat layout holds one pair, and the parser then makes random splits::

    dataset_dir/
    ├── data/
    │   ├── img1.jpg
    │   └── ...
    └── labels.json

``labels.json`` holds a list of class names and a map from the image stem to
the index of its class:

.. code-block:: json

    {
      "classes": ["class1", "class2"],
      "labels": {
        "img1": 0,
        "img2": 1
      }
    }

Segmentation Mask Directory
---------------------------

Each split holds the images, the matching masks, and ``_classes.csv``::

    dataset_dir/
    ├── train/
    │   ├── img1.jpg
    │   ├── img1_mask.png
    │   ├── ...
    │   └── _classes.csv
    ├── valid/
    └── test/

The mask of ``img1.jpg`` is ``img1_mask.png``. Auto-detection looks for the
image of each mask under the ``.jpg`` suffix. Give ``DatasetType.SEGMASK``
explicitly to use another image suffix. The masks are grayscale images, and
each pixel value is a class. ``_classes.csv`` maps the pixel values to the
class names:

.. code-block:: text

    Pixel Value, Class
    0, background
    1, class1
    2, class2

Native LDF
----------

``DatasetType.NATIVE`` reads back what `LuxonisDataset.export` writes with
``--type native``. Use it to move a dataset between machines, or to read an
LDF export without `LuxonisParser`::

    dataset_dir/
    ├── metadata.json
    ├── train/
    │   ├── annotations.json
    │   └── images/
    │       ├── 0.jpg
    │       ├── 1.jpg
    │       └── ...
    ├── val/
    │   ├── annotations.json
    │   └── images/
    └── test/
        ├── annotations.json
        └── images/

``metadata.json`` holds the LDF version of the export, such as
``{"ldf_version": "2.1.0"}``. It is optional, and the parser reads it only to
warn about an export from a newer version. An export larger than
``max_partition_size_gb`` is written as ``<name>_part0``, ``<name>_part1``,
and so on, and each part repeats this layout.

``annotations.json`` holds a list of records. Each record uses the same shape
that `LuxonisDataset.add` accepts, so you can read one without `LuxonisParser`:

.. code-block:: json

    [
      {
        "file": "images/0.jpg",
        "task_name": "detection",
        "sample_metadata": {
          "camera": "left"
        },
        "annotation": {
          "instance_id": 0,
          "class": "person",
          "boundingbox": {
            "x": 0.1,
            "y": 0.2,
            "w": 0.3,
            "h": 0.4
          }
        }
      }
    ]

The keys of a record are:

    - ``file`` holds the path of one image, relative to ``annotations.json``.
      A record with several synchronized sources uses ``files`` instead, which
      maps each source name to a path.
    - ``task_name`` names the task group of the annotation.
    - ``sample_metadata`` holds record-level metadata, such as a camera name
      or a frame number. It is not an annotation label.
    - ``annotation`` holds one detection. The export writes one record for
      each detection, so several records can share the same ``file``.

A detection holds a ``class``, an ``instance_id``, and one payload key. The
payload key is ``boundingbox``, ``keypoints``, ``segmentation``,
``instance_segmentation``, ``array``, or ``metadata``. Boxes and keypoints use
image-normalized coordinates. Masks are written as COCO RLE, with a
``height``, a ``width``, and a compressed ``counts`` string:

.. code-block:: json

    {
      "instance_id": 1,
      "class": "road",
      "segmentation": {
        "height": 720,
        "width": 1280,
        "counts": "b14<000000000^3"
      }
    }

See:
    `luxonis_ml.ldf.annotation` for every payload key, the accepted mask
    encodings, the coordinate conventions, and the metadata categories.

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
    luxonis_ml data parse ./dataset --train 0.8 --val 0.1 --test 0.1
    luxonis_ml data parse ./dataset --train 1000 --val 100 --test 50

Give ``--train``, ``--val``, and ``--test`` each their own value. If you set
only some of them, the remaining splits share the leftover records equally.
This applies to ratios only, not to counts.

The older ``--split-ratio`` option takes one Python list literal of three
values, such as ``--split-ratio "[0.8, 0.1, 0.1]"``. A comma-separated value
without brackets raises ``ValueError``. The option is deprecated. Use
``--train``, ``--val``, and ``--test`` instead.


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

The FiftyOne layout is what the ``fiftyone`` package writes. Install it, then
download one split at a time:

.. code-block:: bash

    fiftyone zoo datasets load coco-2017 \
        --splits validation \
        --kwargs max_samples=1000

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

The ImageNet-sample dataset holds 1,000 images across the 1,000 ImageNet
classes, for image classification. Download it with the ``fiftyone`` package:

.. code-block:: bash

    fiftyone zoo datasets load imagenet-sample

The ImageNet-sample parser handles FiftyOne image classification exports in
flat ``data/`` plus ``labels.json`` form or in split-based
``train/validation/test`` directories. The command above writes the flat form.
The parser splits a flat layout randomly at parse time.

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
