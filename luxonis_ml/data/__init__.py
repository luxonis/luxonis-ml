r"""Public entry point for Luxonis Data Format workflows.

The `luxonis_ml.data` package brings together the high-level APIs used to
create, convert, load, and augment datasets in the Luxonis Data Format (LDF).
LDF is the dataset representation used across the Luxonis training stack for
vision datasets with one or more media sources, task groups, annotation types,
metadata fields, local files, and remote object storage.

This module is intentionally a map of the package rather than the canonical
home for every detailed contract. Details live next to the implementation that
owns them:

    - `luxonis_ml.data.datasets` documents dataset lifecycle, storage layout,
      splits, cloning, merging, remote synchronization, and dataset plugins.
    - `luxonis_ml.ldf.annotation` documents `DatasetRecord`, `Detection`,
      `Category`, and all annotation payload schemas.
    - `luxonis_ml.data.parsers` documents `LuxonisParser`, the supported
      external formats, the directory layout each one expects, the native LDF
      layout, split-ratio modes, and parser-specific caveats.
    - `luxonis_ml.data.loaders` documents `LuxonisLoader`, loader outputs,
      label key conventions, color spaces, filtering, and preprocessing.
    - `luxonis_ml.data.augmentations` documents `AlbumentationsEngine`,
      transform ordering, resizing, batch transforms, and custom transforms.

.. contents:: Table of Contents
   :depth: 2


Core Workflow
=============

Most data pipelines follow the same sequence:

    1. Create or open a `LuxonisDataset`.
    2. Add records from an iterable or parse an external dataset with
       `LuxonisParser`.
    3. Define dataset splits.
    4. Load one or more splits with `LuxonisLoader`.
    5. Optionally apply augmentations through `AlbumentationsEngine`.
    6. Optionally clone, merge, export, push, pull, inspect, sanitize, or
       delete the dataset.

.. list-table:: High-level APIs
   :header-rows: 1

   * - API
     - Use it when you need to
     - More detail
   * - `LuxonisDataset`
     - Create, mutate, split, clone, merge, export, synchronize, or delete LDF
       datasets.
     - `luxonis_ml.data.datasets`
   * - `LuxonisParser`
     - Convert a supported external dataset layout into LDF.
     - `luxonis_ml.data.parsers`
   * - `LuxonisLoader`
     - Iterate image-like inputs, labels, and sample metadata from one or
       more dataset splits.
     - `luxonis_ml.data.loaders`
   * - `AlbumentationsEngine`
     - Apply runtime image and label augmentation while loading samples.
     - `luxonis_ml.data.augmentations`

Example:
    A minimal flow starts with the dataset, then constructs a loader.

    .. python::

        from luxonis_ml.data import LuxonisDataset, LuxonisLoader

        dataset = LuxonisDataset("parking_lot")
        loader = LuxonisLoader(dataset, view="train")

        for sample in loader:
            images = sample.images
            labels = sample.labels
            metadata = sample.metadata

Note:
    Importing from `luxonis_ml.data` is the recommended public API for common
    workflows. Import from lower modules when you need implementation-specific
    models such as annotation schemas, parser classes, or loader base classes.


Tutorial Dataset
================

Most examples in the data package use a small ``parking_lot`` dataset with
cars and motorcycles. It contains object-detection boxes, instance keypoints,
semantic segmentation masks for color/type/brand/binary vehicle classes, and
metadata suitable for trying the full LDF workflow.

The dataset can be used to exercise task naming conventions:

    - keypoint annotations for classes with different skeletons should be
      separated into task groups such as ``"instance_keypoints_car"`` and
      ``"instance_keypoints_motorbike"``;
    - semantic segmentation is usually placed in its own task group, such as
      ``"segmentation"``, because loaders add a background class for
      segmentation tasks.

Hands-on notebooks and scripts for preparing and interacting with LuxonisML
datasets are maintained in the Luxonis AI tutorials repository:
``https://github.com/luxonis/ai-tutorials/tree/main/training``.

The original ``parking_lot`` sample archive used by these examples is
available at
``https://drive.google.com/uc?export=download&id=1OAuLlL_4wRSzZ33BuxM6Uw2QYYgv_19N``.


Records, Tasks, and Labels
==========================

Dataset ingestion is record-based. A record points to one file or to multiple
synchronized files, optionally assigns a task name, and optionally provides an
annotation payload.

Example:
    The two supported media-key styles are easy to distinguish.

    >>> single_source = {"file": "image.jpg", "annotation": None}
    >>> multi_source = {"files": {"rgb": "rgb.png", "depth": "depth.png"}}
    >>> "file" in single_source, "files" in multi_source
    (True, True)

Task names group annotations that should be consumed together by a model or
loader. Loader label keys use ``"task_name/task_type"``. If no task name is
provided, the task name is the empty string and keys start with ``"/"``.

Example:
    >>> task_name = "detection"
    >>> task_type = "boundingbox"
    >>> f"{task_name}/{task_type}"
    'detection/boundingbox'
    >>> f"{''}/segmentation"
    '/segmentation'

See:
    `luxonis_ml.ldf.annotation` for the exact record model, annotation
    payload schemas, normalized coordinate conventions, metadata categories,
    and instance-association rules.


Adding Records with a Generator
===============================

`LuxonisDataset.add` accepts any iterable of records. A generator function
is the usual choice, because it holds one record in memory at a time.

The generator below reads one annotation file for each directory. It yields
the bounding box and the keypoints of one object under a shared
``instance_id``. It yields each segmentation mask under a separate task
name.

.. python::

    import json
    from pathlib import Path

    import cv2
    import numpy as np

    from luxonis_ml.data import LuxonisDataset

    dataset_root = Path("data/parking_lot")


    def generator():
        for annotation_file in dataset_root.rglob("annotations.json"):
            data = json.loads(annotation_file.read_text())
            width = data["dimensions"]["width"]
            height = data["dimensions"]["height"]
            image = str(annotation_file.parent / data["filename"])

            for instance_id, box in data["BoundingBoxAnnotation"].items():
                x, y = box["origin"]
                w, h = box["dimension"]
                yield {
                    "file": image,
                    "task_name": f"instance_keypoints_{box['labelName']}",
                    "annotation": {
                        "class": box["labelName"],
                        "instance_id": int(instance_id),
                        "boundingbox": {
                            "x": x / width,
                            "y": y / height,
                            "w": w / width,
                            "h": h / height,
                        },
                    },
                }

            for instance_id, pose in data["KeypointsAnnotation"].items():
                yield {
                    "file": image,
                    "task_name": f"instance_keypoints_{pose['labelName']}",
                    "annotation": {
                        "instance_id": int(instance_id),
                        "keypoints": {
                            "keypoints": [
                                (
                                    keypoint["location"][0] / width,
                                    keypoint["location"][1] / height,
                                    keypoint["visibility"],
                                )
                                for keypoint in pose["keypoints"]
                            ]
                        },
                    },
                }

            masks = data["VehicleTypeSegmentation"]
            mask_path = annotation_file.parent / masks["filename"]
            mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)

            for instance in masks["instances"]:
                color = np.array(instance["pixelValue"], dtype=np.uint8)
                yield {
                    "file": image,
                    "task_name": "segmentation",
                    "annotation": {
                        "class": instance["labelName"],
                        "segmentation": {
                            "mask": (mask == color)
                            .all(axis=-1)
                            .astype(np.uint8)
                        },
                    },
                }


    dataset = LuxonisDataset("parking_lot")
    dataset.add(generator())

Note:
    The shared ``instance_id`` links the box and the keypoints of one
    object. Give the keypoints of each class a separate task name, because
    the loader stacks the keypoints of one task into a single array.


Defining Splits
===============

`LuxonisDataset.make_splits` assigns the records to named splits. Pass the
ratios of the splits, or pass the exact files of each split.

.. python::

    dataset.make_splits({"train": 0.7, "val": 0.2, "test": 0.1})

    dataset.make_splits(
        {
            "train": ["image_1.jpg", "image_2.jpg"],
            "val": ["image_3.jpg"],
            "test": ["image_4.jpg"],
        }
    )

A call with no argument splits 80/10/10. A ratio is a number from 0 to 1,
and the ratios must sum to 1.

A split from ratios uses only the records that no split holds yet. A second
call with ratios thus needs new records, or it raises an error. A call with
explicit file lists only logs a warning and keeps the old splits. Thus you
can parse the same data twice. Pass ``replace_old_splits=True`` to drop the
old splits and to split every record again.


Annotation Payloads
===================

LDF supports a small set of annotation payload families:

    - classification through a ``"class"`` value;
    - normalized ``xywh`` bounding boxes through ``"boundingbox"``;
    - normalized ``(x, y, visibility)`` keypoint triplets through
      ``"keypoints"``;
    - semantic segmentation masks through polygon points, binary masks, or
      COCO RLE values under ``"segmentation"``;
    - instance segmentation masks through the same mask encodings under
      ``"instance_segmentation"``;
    - arbitrary ``.npy`` array targets through ``"array"``;
    - flexible metadata values through ``"metadata"``.

Any annotation with a class contributes a classification target. When separate
records describe the same physical object, use the same ``instance_id`` so
boxes, keypoints, and instance masks can be associated reliably.

See:
    `luxonis_ml.ldf.annotation` for examples, exact field names, mask
    encoding details, metadata categories, and loader output shapes.


Command Line Interface
======================

The data package also provides dataset operations through ``luxonis_ml data``.
The CLI mirrors the Python APIs for parsing, listing, inspecting, validating,
sanitizing, exporting, synchronizing, cloning, merging, and deleting datasets.

.. code-block:: bash

    luxonis_ml data --help
    luxonis_ml data parse --help
    luxonis_ml data parse <data_directory>
    luxonis_ml data ls
    luxonis_ml data info <dataset_name>
    luxonis_ml data inspect <dataset_name>
    luxonis_ml data health <dataset_name>
    luxonis_ml data sanitize <dataset_name>
    luxonis_ml data export <dataset_name> --type native
    luxonis_ml data export <dataset_name> --type coco
    luxonis_ml data export <dataset_name> --type ultralyticsndjson
    luxonis_ml data push <dataset_name>
    luxonis_ml data pull <dataset_name>
    luxonis_ml data clone <dataset_name> <new_name>
    luxonis_ml data merge <source_name> <target_name>
    luxonis_ml data delete <dataset_name>

See:
    `luxonis_ml.data.__main__` for command implementation details, and
    `luxonis_ml.data.parsers` for the directory layout of every supported
    format, including the native LDF layout that ``--type native`` writes.


Extension Points
================

Datasets and loaders are registry-backed. Third-party packages can expose
entry points in the ``dataset_plugins`` and ``loader_plugins`` groups. This
module loads those plugins at import time and registers them in
`DATASETS_REGISTRY` and `LOADERS_REGISTRY`.

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
    """Register external dataset ``BaseDataset`` plugins."""
    for entry_point in _get_entry_points_subset("dataset_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register(module=plugin_class)


def _load_loader_plugins() -> None:  # pragma: no cover
    """Register external dataset ``BaseLoader`` plugins."""
    for entry_point in _get_entry_points_subset("loader_plugins"):
        plugin_class = entry_point.load()
        LOADERS_REGISTRY.register(module=plugin_class)


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
