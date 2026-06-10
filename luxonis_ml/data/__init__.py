r"""Luxonis Data Format tools for datasets, parsers, loaders, and
augmentations.

The `luxonis_ml.data` package is the public entry point for working with the
Luxonis Data Format (LDF). LDF is a parquet-backed dataset format designed for
computer-vision training pipelines that need a single representation for
classification, detection, keypoints, semantic segmentation, instance
segmentation, arrays, metadata, local media, and remote media.

The package is organized around three user-facing classes:

    - `LuxonisDataset` creates, opens, mutates, splits, clones, merges,
      exports, deletes, and synchronizes LDF datasets.
    - `LuxonisParser` converts external dataset formats into LDF.
    - `LuxonisLoader` iterates LDF samples and optionally applies runtime
      preprocessing and augmentation through `AlbumentationsEngine`.

.. contents:: Table of Contents
   :depth: 2


Data Flow
=========

A typical dataset lifecycle is:

    1. Create or open a `LuxonisDataset`.
    2. Add records yielded by an iterable or generator.
    3. Define split membership with `LuxonisDataset.make_splits`.
    4. Optionally clone, merge, export, push, pull, inspect, or sanitize the
       dataset.
    5. Load one or more splits with `LuxonisLoader`.
    6. Optionally augment samples while loading.

The same storage abstraction is used for local and remote datasets. Local
datasets default to ``LUXONISML_BASE_PATH / data / LUXONISML_TEAM_ID /
datasets / dataset_name``. Unless configured otherwise,
``LUXONISML_BASE_PATH`` is ``Path.home() / "luxonis_ml"`` and
``LUXONISML_TEAM_ID`` is ``"offline"``.

.. list-table:: Main public entry points
   :header-rows: 1

   * - Object
     - Role
     - Typical import
   * - `LuxonisDataset`
     - Create, mutate, split, clone, merge, export, push, pull, and delete LDF
       datasets.
     - ``from luxonis_ml.data import LuxonisDataset``
   * - `LuxonisParser`
     - Convert external formats such as COCO, YOLO, VOC, SOLO, and
       Ultralytics NDJSON into LDF.
     - ``from luxonis_ml.data import LuxonisParser``
   * - `LuxonisLoader`
     - Return image-like inputs and task labels for a selected dataset split.
     - ``from luxonis_ml.data import LuxonisLoader``
   * - `AlbumentationsEngine`
     - Apply Albumentations and Luxonis custom transforms to LDF samples.
     - ``from luxonis_ml.data import AlbumentationsEngine``

Note:
    Most users should import from `luxonis_ml.data` rather than from the
    nested packages directly. The nested packages contain the same concrete
    implementations plus lower-level annotation, parser, loader, and utility
    models.


Quick Start
===========

Create or open a local dataset:

.. python::

    from luxonis_ml.data import LuxonisDataset

    dataset = LuxonisDataset("parking_lot")

Each call to `LuxonisDataset.add` consumes records. Records may describe a
single input file or a synchronized set of input files. Each record contains at
most one logical annotation payload; multiple annotations for the same image
are yielded as multiple records and associated through their shared file path
and optional ``instance_id``.

Example:
    Simple, side-effect-free record checks can be written as doctests.

    >>> record = {
    ...     "file": "image.jpg",
    ...     "task_name": "detection",
    ...     "annotation": {
    ...         "class": "car",
    ...         "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
    ...     },
    ... }
    >>> sorted(record)
    ['annotation', 'file', 'task_name']
    >>> tuple(record["annotation"]["boundingbox"])
    ('x', 'y', 'w', 'h')

Add records, then define splits:

.. python::

    def generator():
        yield {
            "file": "path/to/image.jpg",
            "task_name": "detection",
            "annotation": {
                "class": "car",
                "boundingbox": {
                    "x": 0.15,
                    "y": 0.20,
                    "w": 0.30,
                    "h": 0.25,
                },
            },
        }

    dataset.add(generator())
    dataset.make_splits({"train": 0.8, "val": 0.1, "test": 0.1})

Load the training split:

.. python::

    from luxonis_ml.data import LuxonisLoader

    loader = LuxonisLoader(dataset, view="train")

    for inputs, labels in loader:
        ...

Important:
    Dataset construction is intentionally idempotent by name. If a dataset
    with the requested name already exists, `LuxonisDataset` opens it. To
    recreate local state, pass ``delete_local=True``. For remote datasets, be
    explicit about both ``delete_local`` and ``delete_remote``.


Dataset Records
===============

Single-source datasets use the ``"file"`` key:

.. python::

    {
        "file": "path/to/rgb_image.png",
        "task_name": "detection",
        "annotation": {
            "class": "person",
            "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.4},
        },
    }

Multi-source datasets use the ``"files"`` key. The source names are arbitrary
identifiers that describe the input role or modality and later become loader
input keys:

.. python::

    {
        "files": {
            "rgb": "path/to/rgb_image.png",
            "depth": "path/to/depth_image.png",
        },
        "task_name": "detection",
        "annotation": {
            "class": "person",
            "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.4},
        },
    }

The ``task_name`` groups annotations that should be learned or consumed
together. If omitted, the empty string ``""`` is used, so loader label keys
start with ``"/"``.

Task names are especially useful when a dataset mixes incompatible label
groups. For example, semantic segmentation usually belongs in its own task
because `LuxonisLoader` adds a background class, while class-specific
keypoint tasks often need separate task names because different classes can
have different numbers of keypoints.

Example:
    ``task_name`` values become the prefix in loader label keys.

    >>> task_name = "pose_car"
    >>> task_type = "keypoints"
    >>> f"{task_name}/{task_type}"
    'pose_car/keypoints'
    >>> default_task_name = ""
    >>> f"{default_task_name}/boundingbox"
    '/boundingbox'


Annotation Format
=================

All coordinates stored in standard spatial annotations are normalized to the
image dimensions. For an image with width :math:`W` and height :math:`H`, an
absolute point :math:`(x, y)` is stored as
:math:`\left(x / W, y / H\right)`.

Classification
--------------

Classification assigns one class to the whole sample:

.. python::

    {"class": "vehicle"}

Classification annotations are always represented as a task type internally.
When multiple classes are present for a task, `LuxonisLoader` returns a
one-hot vector with shape :math:`\left(C\right)`.

Bounding Boxes
--------------

Bounding boxes use normalized ``xywh`` coordinates, where ``x`` and ``y`` are
the top-left corner:

.. python::

    {
        "class": "car",
        "instance_id": 17,
        "boundingbox": {
            "x": 0.20,
            "y": 0.10,
            "w": 0.35,
            "h": 0.25,
        },
    }

The loader combines boxes into an array with shape :math:`\left(N, 5\right)`.
Each row is :math:`\left[c, x, y, w, h\right]`, where :math:`c` is the class
index.

Example:
    The normalized area is :math:`w \cdot h`.

    >>> box = {"x": 0.2, "y": 0.1, "w": 0.35, "h": 0.25}
    >>> round(box["w"] * box["h"], 4)
    0.0875

Keypoints
---------

Keypoints are stored as ``(x, y, visibility)`` triplets. Coordinates are
normalized and visibility follows the common convention:

    - ``0``: not visible or outside the image.
    - ``1``: inside the image but occluded.
    - ``2``: visible.

.. python::

    {
        "class": "car",
        "instance_id": 17,
        "keypoints": {
            "keypoints": [
                (0.10, 0.20, 2),
                (0.30, 0.40, 1),
            ],
        },
    }

For :math:`K` keypoints and :math:`N` instances, `LuxonisLoader` returns
keypoints with shape :math:`\left(N, 3K\right)`.

Segmentation
------------

Semantic segmentation masks can be provided as polygons, binary arrays, or
run-length encoding.

Polyline segmentation stores normalized polygon points. The final point is
implicitly connected to the first one:

.. python::

    {
        "class": "road",
        "segmentation": {
            "height": 720,
            "width": 1280,
            "points": [
                (0.10, 0.10),
                (0.90, 0.10),
                (0.80, 0.80),
                (0.20, 0.80),
            ],
        },
    }

Binary masks are two-dimensional arrays where foreground pixels are
:math:`1` and background pixels are :math:`0`:

.. python::

    {
        "class": "road",
        "segmentation": {
            "mask": binary_mask,
        },
    }

Run-length encoded masks use the COCO RLE representation. The ``counts`` value
may be an uncompressed list of integers or a compressed byte string:

.. python::

    {
        "class": "road",
        "segmentation": {
            "height": 720,
            "width": 1280,
            "counts": [120, 8, 200, 12],
        },
    }

Note:
    Array masks are converted to RLE internally. RLE is primarily intended for
    interoperability with datasets that already store masks in that format.

`LuxonisLoader` combines semantic segmentation labels into
:math:`\left(C, H, W\right)` masks in channel-first format.

Instance Segmentation
---------------------

Instance segmentation uses the same mask encodings as semantic segmentation,
but associates masks with object instances. A record may also contain a
matching bounding box:

.. python::

    {
        "class": "car",
        "instance_id": 17,
        "boundingbox": {"x": 0.20, "y": 0.10, "w": 0.35, "h": 0.25},
        "instance_segmentation": {
            "height": 720,
            "width": 1280,
            "points": [
                (0.20, 0.10),
                (0.55, 0.10),
                (0.55, 0.35),
                (0.20, 0.35),
            ],
        },
    }

When bounding boxes, keypoints, and instance masks are yielded in separate
records for the same physical object, provide the same ``instance_id`` so they
can be associated.

Arrays
------

Array annotations reference arbitrary ``.npy`` data:

.. python::

    {
        "class": "embedding",
        "array": {
            "path": "path/to/embedding.npy",
        },
    }

Arrays are useful for data that should stay synchronized with media but does
not fit one of the standard spatial schemas.

Metadata
--------

Metadata stores flexible key-value data. Use `Category` when a metadata value
is categorical and should be tracked as a category rather than as an arbitrary
string:

.. python::

    from luxonis_ml.data import Category

    {
        "metadata": {
            "text": "ABC-123",
            "text_color": Category("white"),
            "track_id": 42,
        },
    }

Categorical metadata may be returned as encoded integers or as strings,
depending on `LuxonisLoader` configuration.

Warning:
    Metadata and arrays do not have universal geometric semantics. Built-in
    augmentations can discard metadata or arrays associated with boxes that
    leave the image, but arbitrary values are otherwise kept unchanged unless a
    custom augmentation explicitly handles them.


Dataset Storage and Management
==============================

An LDF dataset stores annotation rows in parquet shards and dataset-level state
in metadata files.

.. list-table:: LDF storage layout
   :header-rows: 1

   * - Path
     - Contents
   * - ``annotations/*.parquet``
     - Rows containing file paths, source names, task names, class names,
       instance IDs, task types, serialized annotation payloads, and UUIDs.
   * - ``media/``
     - Local copies of remote media, usually named by UUID. Local-only
       datasets may keep this directory empty and continue referencing the
       original files.
   * - ``metadata/metadata.json``
     - Classes, tasks, skeletons, source metadata, categorical encodings, and
       LDF version metadata.
   * - ``metadata/splits.json``
     - Mapping from split names to sample identifiers.

Local datasets use `BucketStorage.LOCAL`. Remote datasets can use supported
object stores such as `BucketStorage.GCS` or `BucketStorage.S3`, depending on
environment configuration.

.. python::

    from luxonis_ml.data import BucketStorage, LuxonisDataset

    local_dataset = LuxonisDataset(
        "parking_lot",
        bucket_storage=BucketStorage.LOCAL,
    )

    remote_dataset = LuxonisDataset(
        "parking_lot_cloud",
        bucket_storage=BucketStorage.GCS,
    )

Splits can be defined by ratios or by explicit file lists:

.. python::

    dataset.make_splits({"train": 0.7, "val": 0.2, "test": 0.1})

    dataset.make_splits(
        {
            "train": ["image_001.jpg", "image_002.jpg"],
            "val": ["image_003.jpg"],
            "test": ["image_004.jpg"],
        },
        replace_old_splits=True,
    )

Example:
    Split ratios are probabilities and should sum to :math:`1`.

    >>> ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    >>> round(sum(ratios.values()), 6)
    1.0

Datasets can be cloned or merged:

.. python::

    clone = dataset.clone(new_dataset_name="parking_lot_clone")

    dataset1.merge_with(dataset2, inplace=True)

    merged = dataset1.merge_with(
        dataset2,
        inplace=False,
        new_dataset_name="parking_lot_merged",
    )

Remote synchronization is explicit:

.. python::

    from luxonis_ml.data import UpdateMode

    dataset.pull_from_cloud(update_mode=UpdateMode.MISSING)
    dataset.push_to_cloud(
        bucket_storage=BucketStorage.GCS,
        update_mode=UpdateMode.ALL,
    )

Warning:
    ``delete_local`` and ``delete_remote`` affect different stores. To
    recreate a remote dataset from scratch, use both flags. To discard a
    damaged local cache while preserving a healthy remote dataset, use
    ``delete_local=True`` and ``delete_remote=False``.


Parsing External Formats
========================

`LuxonisParser` converts supported external formats into `LuxonisDataset`.
The input may be a local directory, a ``.zip`` archive, a remote path supported
by `LuxonisFileSystem`, or a Roboflow identifier in
``roboflow://workspace/project/version/format`` form.

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

When ``dataset_type`` is omitted, `LuxonisParser` attempts to infer the format
from the directory structure. When ``task_name`` is a mapping, class names are
mapped to task names:

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

.. list-table:: Supported parser formats
   :header-rows: 1

   * - Format
     - `DatasetType`
     - Typical annotations
   * - COCO JSON
     - `DatasetType.COCO`
     - Bounding boxes, segmentation, instance segmentation, keypoints.
   * - Pascal VOC XML
     - `DatasetType.VOC`
     - Bounding boxes.
   * - YOLO Darknet TXT
     - `DatasetType.DARKNET`
     - Bounding boxes.
   * - YOLOv4 PyTorch TXT
     - `DatasetType.YOLOV4`
     - Bounding boxes.
   * - MT YOLOv6
     - `DatasetType.YOLOV6`
     - Bounding boxes.
   * - YOLOv8 bounding boxes, instances, or keypoints
     - `DatasetType.YOLOV8BOUNDINGBOX`,
       `DatasetType.YOLOV8INSTANCESEGMENTATION`,
       `DatasetType.YOLOV8KEYPOINTS`
     - Task-specific YOLO labels.
   * - Ultralytics NDJSON
     - `DatasetType.ULTRALYTICSNDJSON`,
       `DatasetType.ULTRALYTICSNDJSONINSTANCESEGMENTATION`,
       `DatasetType.ULTRALYTICSNDJSONKEYPOINTS`
     - Detection, segmentation, or pose records, including remote image URLs.
   * - CreateML JSON
     - `DatasetType.CREATEML`
     - Bounding boxes.
   * - TensorFlow Object Detection CSV
     - `DatasetType.TFCSV`
     - Bounding boxes.
   * - SOLO
     - `DatasetType.SOLO`
     - Synthetic data with boxes, masks, keypoints, and segmentation.
   * - Classification directory
     - `DatasetType.CLSDIR`
     - Class labels encoded by directory names.
   * - FiftyOne classification
     - `DatasetType.FIFTYONECLS`
     - Class labels from ``labels.json``.
   * - Segmentation mask directory
     - `DatasetType.SEGMASK`
     - Grayscale masks with class mappings in ``_classes.csv``.
   * - Native LDF
     - `DatasetType.NATIVE`
     - Existing Luxonis native exports.

Note:
    When parsing ZIP files, place the train, validation, and test directories
    directly at the archive root. Do not wrap the dataset in an additional
    top-level ``dataset_dir`` folder unless that is the actual format expected
    by the selected parser.

Split Ratio Modes
-----------------

Parser split ratios support two modes:

    - Floating-point values, such as :math:`0.8`, :math:`0.1`, and
      :math:`0.1`, redistribute and shuffle samples across splits. Values
      should sum to :math:`1.0`.
    - Integer counts, such as :math:`1000`, :math:`100`, and :math:`50`, draw
      from the corresponding original split and preserve split boundaries. If
      a requested count exceeds the available samples, all available samples
      from that split are used.

.. code-block:: bash

    luxonis_ml data parse ./dataset --name parking_lot --type coco
    luxonis_ml data parse ./dataset --split-ratio 0.8,0.1,0.1
    luxonis_ml data parse ./dataset --split-ratio 1000,100,50

COCO Keypoints
--------------

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

Warning:
    Format-specific parsers do not follow symbolic links. Datasets that rely
    on symlinked images or labels may not parse as expected.


Loading Samples
===============

`LuxonisLoader` provides indexed and iterable access to one or more dataset
splits:

.. python::

    from luxonis_ml.data import LuxonisDataset, LuxonisLoader

    dataset = LuxonisDataset("parking_lot")
    loader = LuxonisLoader(dataset, view=["train", "val"])

    inputs, labels = loader[0]

For single-source datasets, ``inputs`` is usually one image array. For
multi-source datasets, ``inputs`` is a dictionary keyed by source name. Labels
are stored in a dictionary keyed by ``"task_name/task_type"``. Metadata labels
use ``"task_name/metadata/key"``.

.. list-table:: Common loader label layouts
   :header-rows: 1

   * - Task type
     - Shape or structure
     - Meaning
   * - ``classification``
     - :math:`\left(C\right)`
     - One-hot class vector.
   * - ``boundingbox``
     - :math:`\left(N, 5\right)`
     - Rows are :math:`\left[c, x, y, w, h\right]`.
   * - ``segmentation``
     - :math:`\left(C, H, W\right)`
     - One-hot semantic mask in channel-first layout.
   * - ``instance_segmentation``
     - :math:`\left(N, H, W\right)`
     - One binary mask per instance.
   * - ``keypoints``
     - :math:`\left(N, 3K\right)`
     - Flattened :math:`\left(x, y, v\right)` triplets.
   * - ``metadata``
     - Original value structure.
     - Values keyed by metadata field name.

Loader options include color-space conversion, output ``height`` and
``width``, aspect-ratio preservation, random seed, remote update mode,
task-name filtering, empty-annotation filtering, bbox visibility thresholds,
and categorical metadata encoding.

.. python::

    loader = LuxonisLoader(
        dataset,
        view="train",
        height=640,
        width=640,
        keep_aspect_ratio=True,
        color_space="RGB",
        filter_task_names=["detection"],
        exclude_empty_annotations=True,
    )


Augmentation
============

`LuxonisLoader` can construct an augmentation engine from an
``augmentation_config``. The default engine is `AlbumentationsEngine`, which
adapts LDF labels to Albumentations targets, applies supported transforms, and
converts labels back to loader output structures.

Augmentation configuration is a list of dictionaries with a transform
``name``, optional ``params``, optional ``apply_on_stages``, and optional
``use_for_resizing``:

.. python::

    augmentation_config = [
        {
            "name": "HueSaturationValue",
            "params": {
                "p": 0.5,
                "hue_shift_limit": 3,
                "sat_shift_limit": 70,
                "val_shift_limit": 40,
            },
        },
        {
            "name": "Rotate",
            "params": {
                "p": 0.6,
                "limit": 30,
                "border_mode": 0,
                "value": [0, 0, 0],
            },
        },
        {
            "name": "Mosaic4",
            "params": {
                "height": 640,
                "width": 640,
                "p": 1.0,
            },
        },
    ]

    loader = LuxonisLoader(
        dataset,
        view="train",
        augmentation_config=augmentation_config,
        augmentation_engine="albumentations",
        height=640,
        width=640,
    )

The engine groups transforms by behavior rather than preserving exact input
order:

    1. Batch transforms such as `MixUp` and `Mosaic4`.
    2. Spatial transforms such as Albumentations dual transforms.
    3. Custom basic transforms.
    4. Pixel-only transforms.

Batch transforms increase the number of source samples needed to produce one
augmented output. A pipeline containing `MixUp` and `Mosaic4` requires
:math:`8 = 2 \cdot 4` source samples per output.

Important:
    Standard Albumentations flips transform keypoint coordinates but do not
    swap semantic left/right keypoint identities. For symmetric keypoint
    structures, prefer the Luxonis custom symmetric keypoint flip transforms.

See:
    `AlbumentationsEngine` for details about target conversion, transform
    ordering, custom transform registration, unsupported target behavior, and
    batch augmentation semantics.


Command Line Interface
======================

Dataset operations are also available through ``luxonis_ml data``:

.. code-block:: bash

    luxonis_ml data parse path/to/dataset --name parking_lot --type coco
    luxonis_ml data ls
    luxonis_ml data info parking_lot
    luxonis_ml data inspect parking_lot
    luxonis_ml data health parking_lot
    luxonis_ml data sanitize parking_lot
    luxonis_ml data export parking_lot --type ultralytics-ndjson
    luxonis_ml data push parking_lot
    luxonis_ml data pull parking_lot
    luxonis_ml data clone parking_lot parking_lot_copy
    luxonis_ml data merge source_dataset target_dataset
    luxonis_ml data delete parking_lot

Run ``luxonis_ml data --help`` or ``luxonis_ml data COMMAND --help`` for the
full set of options.


Extension Points
================

Datasets and loaders are registry-backed. Third-party packages can expose
entry points in the ``dataset_plugins`` and ``loader_plugins`` groups. The
package imports those plugins at module import time and registers them in
`DATASETS_REGISTRY` and `LOADERS_REGISTRY`.

The public registries are useful when integrating LDF with another storage
implementation, framework-specific loader, or training environment while
preserving the same parser and annotation contracts.

See:
    `BaseDataset`, `BaseLoader`, `DatasetIterator`,
    `DATASETS_REGISTRY`, and `LOADERS_REGISTRY`.

"""

from importlib.metadata import entry_points

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("data"):
    from .augmentations import AlbumentationsEngine
    from .datasets import (
        DATASETS_REGISTRY,
        BaseDataset,
        Category,
        DatasetIterator,
        LuxonisComponent,
        LuxonisDataset,
        LuxonisSource,
        Metadata,
        UpdateMode,
    )
    from .loaders import LOADERS_REGISTRY, BaseLoader, LuxonisLoader
    from .parsers import LuxonisParser
    from .utils.enums import (
        BucketStorage,
        BucketType,
        ImageType,
        MediaType,
        ParserIssue,
        ParserIssueMessage,
    )
    from .utils.ldf_equivalence import LDFEquivalence, ldf_equivalent


def _load_dataset_plugins() -> None:  # pragma: no cover
    for entry_point in _get_entry_points_subset("dataset_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register(module=plugin_class)


def _load_loader_plugins() -> None:  # pragma: no cover
    for entry_point in _get_entry_points_subset("loader_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register(module=plugin_class)


def _get_entry_points_subset(key: str) -> list:
    entry_points_obj = entry_points()
    if isinstance(entry_points_obj, dict):
        # py3.8 specific
        selected_entry_points = entry_points_obj.get(key, [])
    else:
        selected_entry_points = entry_points_obj.select(group=key)
    return selected_entry_points


_load_dataset_plugins()
_load_loader_plugins()

__all__ = [
    "DATASETS_REGISTRY",
    "LOADERS_REGISTRY",
    "AlbumentationsEngine",
    "BaseDataset",
    "BaseLoader",
    "BucketStorage",
    "BucketType",
    "Category",
    "DatasetIterator",
    "ImageType",
    "LDFEquivalence",
    "LuxonisComponent",
    "LuxonisDataset",
    "LuxonisLoader",
    "LuxonisParser",
    "LuxonisSource",
    "MediaType",
    "Metadata",
    "ParserIssue",
    "ParserIssueMessage",
    "UpdateMode",
    "ldf_equivalent",
]
