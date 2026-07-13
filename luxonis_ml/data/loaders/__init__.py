r"""Dataset loaders for LDF samples.

This package owns runtime sample access. `LuxonisLoader` reads one or more
dataset splits, resolves media paths, assembles labels by task key, optionally
applies augmentations, and returns data in a shape suitable for training
pipelines.

.. contents:: Table of Contents
   :depth: 2


Basic Usage
===========

.. python::

    from luxonis_ml.data import LuxonisDataset, LuxonisLoader

    dataset = LuxonisDataset("parking_lot")
    loader = LuxonisLoader(dataset, view="train")

    inputs, labels = loader[0]

`LuxonisLoader` implements indexed access and iteration. The returned value is
always ``(inputs, labels)``.

For single-source datasets, ``inputs`` is a single image-like array. For
multi-source datasets, ``inputs`` is a dictionary mapping source names to
arrays.


Constructor Options
===================

`LuxonisLoader` is configured at runtime and does not mutate stored dataset
state. Common options include:

    - ``view`` to load one split or a list of splits.
    - ``augmentation_engine`` and ``augmentation_config`` to enable
      augmentations.
    - ``height``, ``width``, and ``keep_aspect_ratio`` to define the resize
      behavior expected by the augmentation engine.
    - ``color_space`` to request ``"RGB"``, ``"BGR"``, or ``"GRAY"`` output
      globally or per source name.
    - ``seed`` for reproducible random augmentations.
    - ``exclude_empty_annotations`` to omit empty labels.
    - ``keep_categorical_as_strings`` to preserve categorical metadata values.
    - ``update_mode`` to control media synchronization for remote datasets.
    - ``filter_task_names`` to load only selected task groups.

When a remote dataset is loaded, annotations and metadata are refreshed. Media
files are downloaded according to `UpdateMode`: ``ALL`` overwrites local media
and ``MISSING`` downloads only media that cannot be resolved locally.


Label Keys
==========

The ``labels`` dictionary is keyed by ``"task_name/task_type"``. If a dataset
was created without a task name, the default task name is the empty string and
keys start with ``"/"``.

Example:
    >>> task_name = "detection"
    >>> task_type = "boundingbox"
    >>> f"{task_name}/{task_type}"
    'detection/boundingbox'
    >>> f"{''}/metadata/camera_angle"
    '/metadata/camera_angle'

Metadata labels use ``"task_name/metadata/key"`` so each metadata field can be
consumed independently.


Output Layouts
==============

.. list-table:: Common label layouts
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
     - Flattened :math:`\left(x, y, v\right)` keypoint triplets.
   * - ``metadata``
     - Original value structure.
     - Values keyed by metadata field name.

See:
    `luxonis_ml.data.datasets.annotation` for the ingestion schemas that are
    converted into these loader outputs.


Runtime Options
===============

`LuxonisLoader` owns runtime concerns that are intentionally separate from
dataset storage:

    - selected views through ``view``;
    - color-space conversion through ``color_space``;
    - optional resizing through ``height`` and ``width``;
    - aspect-ratio preservation through ``keep_aspect_ratio``;
    - augmentation engine construction through ``augmentation_engine`` and
      ``augmentation_config``;
    - remote media synchronization through ``update_mode``;
    - empty-annotation filtering through ``exclude_empty_annotations``;
    - metadata category encoding through ``keep_categorical_as_strings``;
    - task filtering through ``filter_task_names``.

.. python::

    loader = LuxonisLoader(
        dataset,
        view=["train", "val"],
        height=640,
        width=640,
        keep_aspect_ratio=True,
        color_space="RGB",
        filter_task_names=["detection"],
        exclude_empty_annotations=True,
    )

Important:
    Augmentations require output ``height`` and ``width`` so the loader can
    construct a deterministic resizing stage.

See:
    `luxonis_ml.data.augmentations` for augmentation configuration and target
    conversion details.

"""

from .base_loader import LOADERS_REGISTRY, BaseLoader
from .luxonis_loader import LuxonisLoader

__all__ = ["LOADERS_REGISTRY", "BaseLoader", "LuxonisLoader"]
