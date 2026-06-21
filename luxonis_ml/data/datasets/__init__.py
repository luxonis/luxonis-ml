r"""Dataset handles, records, metadata, and storage abstractions for LDF.

This package owns the persistent Luxonis Data Format (LDF) dataset contract.
The primary entry point is `LuxonisDataset`, which creates or opens a dataset
and provides methods for adding records, defining splits, setting class order
and keypoint skeletons, merging or cloning datasets, exporting datasets, and
synchronizing remote media.

The exact annotation payload schemas live in
`luxonis_ml.data.datasets.annotation`. This package-level documentation focuses
on dataset lifecycle and storage.

.. contents:: Table of Contents
   :depth: 2


Dataset Lifecycle
=================

A dataset is identified by ``dataset_name`` and optional storage settings.
Constructing `LuxonisDataset` opens an existing dataset when one is present, or
initializes a new one when no matching dataset exists.

.. python::

    from luxonis_ml.data import BucketStorage, LuxonisDataset

    dataset = LuxonisDataset(
        "parking_lot",
        bucket_storage=BucketStorage.LOCAL,
    )

Typical mutation flow:

    1. Yield `DatasetRecord`-compatible dictionaries from an iterable.
    2. Pass the iterable to `LuxonisDataset.add`.
    3. Call `LuxonisDataset.make_splits` to define split membership.
    4. Optionally clone, merge, export, push, pull, inspect, sanitize, or
       delete through `LuxonisDataset` methods or the CLI.

.. python::

    def records():
        yield {
            "file": "path/to/image.jpg",
            "task_name": "detection",
            "annotation": {
                "class": "car",
                "boundingbox": {
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.3,
                    "h": 0.4,
                },
            },
        }

    dataset.add(records())
    dataset.make_splits({"train": 0.8, "val": 0.1, "test": 0.1})

Note:
    Although ``"train"``, ``"val"``, and ``"test"`` are conventional split
    names, `LuxonisDataset.make_splits` accepts arbitrary split names when a
    project needs custom views.


Creation and Append Modes
=========================

`LuxonisDataset` creates local datasets by default. Pass
``bucket_storage=BucketStorage.GCS``, ``BucketStorage.S3``, or another
supported storage backend to create a remote-backed dataset. Remote datasets
keep the same local metadata structure as local datasets, while media files
are synchronized with object storage.

Opening an existing dataset with the same name reuses it. To force a clean
local dataset, pass ``delete_local=True``. To force a clean remote dataset,
delete both sides with ``delete_local=True`` and ``delete_remote=True`` before
adding records.

.. python::

    dataset = LuxonisDataset(
        "parking_lot",
        bucket_storage=BucketStorage.LOCAL,
        delete_local=True,
    )
    dataset.add(records(), batch_size=100_000_000)
    dataset.make_splits((0.8, 0.1, 0.1))

To append new data, open the dataset with ``delete_local=False`` and call
`LuxonisDataset.add` again. New annotation records are appended, while records
for media with the same informational UUID replace previous annotations for
that media item.

The `LuxonisDataset.add` ``batch_size`` controls how many annotation records
are buffered before writing a Parquet shard. For remote datasets, the same
batch boundary also controls when media and annotation shards are pushed to
cloud storage.


Dataset Records
===============

`LuxonisDataset.add` accepts any iterable yielding `DatasetRecord`-compatible
items. A record may use ``"file"`` for a single source or ``"files"`` for
multiple synchronized sources.

.. list-table:: Record-level fields
   :header-rows: 1

   * - Field
     - Meaning
   * - ``file``
     - Path to a single media source.
   * - ``files``
     - Mapping from source names to synchronized media paths.
   * - ``task_name``
     - Optional group name used by loaders and metadata.
   * - ``annotation``
     - Optional annotation payload validated by `Detection`.

Multi-source records preserve source names for `LuxonisLoader`, allowing
training code to receive dictionaries such as ``{"rgb": ..., "depth": ...}``.

See:
    `DatasetRecord` for record validation, `Detection` for payload grouping,
    and `luxonis_ml.data.datasets.annotation` for detailed annotation schemas.


Storage Layout
==============

By default, local datasets live under ``LUXONISML_BASE_PATH / data /
LUXONISML_TEAM_ID / datasets / dataset_name``. The default base path is
``Path.home() / "luxonis_ml"`` and the default team identifier is
``"offline"``.

.. list-table:: Dataset storage layout
   :header-rows: 1

   * - Path
     - Contents
   * - ``annotations/*.parquet``
     - Parquet shards containing media paths, source names, task names, class
       names, instance IDs, task types, serialized annotation payloads, and
       UUIDs.
   * - ``media/``
     - Local copies of remote media. Local-only datasets may keep this
       directory empty and continue referencing the original files.
   * - ``metadata/metadata.json``
     - Dataset metadata, source descriptors, class mappings, task metadata,
       categorical encodings, skeleton definitions, and LDF version metadata.
   * - ``metadata/splits.json``
     - Mapping from split names to dataset sample identifiers.

Remote datasets use the same local metadata structure and synchronize media
and annotation state with the configured object store.

Warning:
    Deletion flags control local and remote state independently. To recreate a
    remote dataset completely, pass ``delete_local=True`` and
    ``delete_remote=True``. To rebuild a damaged local copy from an existing
    remote dataset, pass ``delete_local=True`` and ``delete_remote=False``.


Cloning, Merging, and Synchronization
=====================================

`LuxonisDataset.clone` creates a copy under a new dataset name.
`LuxonisDataset.merge_with` combines two datasets either in place or into a new
dataset.

.. python::

    clone = dataset.clone(new_dataset_name="parking_lot_clone")

    dataset1.merge_with(dataset2, inplace=True)

    merged = dataset1.merge_with(
        dataset2,
        inplace=False,
        new_dataset_name="parking_lot_merged",
    )

Remote synchronization is explicit and controlled by `UpdateMode`:

.. python::

    from luxonis_ml.data import BucketStorage, UpdateMode

    dataset.pull_from_cloud(update_mode=UpdateMode.MISSING)
    dataset.push_to_cloud(
        bucket_storage=BucketStorage.GCS,
        update_mode=UpdateMode.ALL,
    )

Important:
    Annotation shards and metadata are always synchronized. Media update mode
    controls whether all media files or only missing media files are
    transferred.


Class Ordering
==============

`LuxonisDataset.set_class_order_per_task` applies a view-time class order per
task without rewriting stored metadata. The provided mapping must use exact
task names and exact class names already present in the dataset.

.. python::

    dataset.set_class_order_per_task(
        {
            "vehicle_detection": ["car", "motorcycle"],
            "color_segmentation": ["background", "red", "green", "blue"],
        }
    )

Call this before constructing `LuxonisLoader`, because loader initialization
uses the dataset's active class ordering. If later additions introduce new
classes, call `set_class_order_per_task` again with the complete desired
order.

"""

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
    SegmentationAnnotation,
    load_annotation,
)
from .base_dataset import DATASETS_REGISTRY, BaseDataset, DatasetIterator
from .luxonis_dataset import LuxonisDataset, UpdateMode
from .metadata import Metadata
from .source import LuxonisComponent, LuxonisSource

__all__ = [
    "DATASETS_REGISTRY",
    "Annotation",
    "ArrayAnnotation",
    "BBoxAnnotation",
    "BaseDataset",
    "Category",
    "ClassificationAnnotation",
    "DatasetIterator",
    "DatasetRecord",
    "Detection",
    "InstanceSegmentationAnnotation",
    "KeypointAnnotation",
    "LuxonisComponent",
    "LuxonisDataset",
    "LuxonisSource",
    "Metadata",
    "SegmentationAnnotation",
    "UpdateMode",
    "load_annotation",
]
