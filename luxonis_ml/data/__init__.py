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
    - `luxonis_ml.data.datasets.annotation` documents `DatasetRecord`,
      `Detection`, `Category`, and all annotation payload schemas.
    - `luxonis_ml.data.parsers` documents `LuxonisParser`, supported external
      formats, split-ratio modes, and parser-specific caveats.
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
     - Iterate image-like inputs and labels from one or more dataset splits.
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

        for inputs, labels in loader:
            ...

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
    `luxonis_ml.data.datasets.annotation` for the exact record model,
    annotation payload schemas, normalized coordinate conventions, metadata
    categories, and instance-association rules.


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
    luxonis_ml data export <dataset_name> --type ultralytics-ndjson
    luxonis_ml data export <dataset_name> --type ultralytics-ndjson-instancesegmentation
    luxonis_ml data export <dataset_name> --type ultralytics-ndjson-keypoints
    luxonis_ml data push <dataset_name>
    luxonis_ml data pull <dataset_name>
    luxonis_ml data clone <dataset_name> <new_name>
    luxonis_ml data merge <source_name> <target_name>
    luxonis_ml data delete <dataset_name>

See:
    `luxonis_ml.data.__main__` for command implementation details.


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
