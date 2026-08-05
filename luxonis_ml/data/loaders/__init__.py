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

    sample = loader[0]

    images = sample.images
    labels = sample.labels
    metadata = sample.metadata

`LuxonisLoader` implements indexed access and iteration. The returned value is
always `LoaderOutput`.

`LoaderOutput.images` maps source names to arrays. Single-source datasets use
the conventional ``"image"`` source name; multi-source datasets preserve the
names from the record ``"files"`` mapping.

`LoaderOutput.metadata` contains **record-level metadata** from
``DatasetRecord.sample_metadata``. By default it also includes a
``"filenames"`` mapping from source name to the loaded file basename.

Legacy two-value unpacking still works when code only needs inputs and labels:

.. python::

    image_or_images, labels = loader[0]

Metadata is not yielded by tuple unpacking. Access it through
``loader[0].metadata``.


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
    - ``autopopulate_metadata`` to include automatic metadata such as source
      filenames in `LoaderOutput.metadata`.

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

Sample metadata does not use label keys. It is returned separately through
`LoaderOutput.metadata`:

.. python::

    sample = loader[0]
    print(sample.metadata)

    # {
    #     "filenames": {"image": "frame_001.jpg"},
    #     "record_id": 123,
    #     "camera": "left",
    # }


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
     - :math:`\left(N\right)`
     - One value per instance, in the same row order as the task's spatial
       labels. Instances that do not carry the field hold ``None``.
   * - ``array``
     - :math:`\left(N, C, \ldots\right)`
     - Whatever each ``.npy`` held, wrapped in an instance and a class axis.

Array labels
------------

``array`` is the only layout whose trailing shape is not fixed: it is whatever
the ``.npy`` files held. The two leading axes come from
`ArrayAnnotation.combine_to_numpy`, which stacks one row per annotation and puts
each row's data in its own class slot, leaving the rest of that row zero.

This makes the storage layout matter. A :math:`\left(2, H, W\right)` array
written as one ``.npy`` under a single class arrives as
:math:`\left(1, 1, 2, H, W\right)`, with nothing naming its channels. Written as
two annotations with class names ``u`` and ``v``, it arrives as
:math:`\left(2, 2, H, W\right)` populated down the diagonal, and summing away
the instance axis recovers :math:`\left(2, H, W\right)` in class order. Use the
per-channel form when the channels have names worth keeping.

See:
    `luxonis_ml.ldf.annotation` for the ingestion schemas that are converted
    into these loader outputs.


Back to LDF Records
===================

`LoaderOutput.to_ldf` reverses the conversion, rebuilding the label arrays into
one `DatasetRecord` per task name for visualization:

.. python::

    sample = loader[0]

    for task_name, record in sample.to_ldf().items():
        for detection in record.annotation or []:
            print(task_name, detection.class_name, detection.boundingbox)

The loader attaches the class and categorical metadata mappings needed for the
conversion. Images and array annotations remain in memory, so the records
cannot be added to a dataset.


Sample Metadata
===============

**Record-level metadata** comes from ``DatasetRecord.sample_metadata`` and is
available as `LoaderOutput.metadata`.

.. python::

    sample = loader[0]

    sample.metadata
    # {
    #     "filenames": {"image": "frame_001.jpg"},
    #     "record_id": 123,
    #     "camera": "left",
    # }

Pass ``autopopulate_metadata=False`` to return only stored metadata:

.. python::

    loader = LuxonisLoader(dataset, autopopulate_metadata=False)
    metadata = loader[0].metadata

Augmented outputs additionally carry an ``"augmentations"`` mapping from
configured augmentation paths to the runtime parameters the engine selected.
Unlike ``"filenames"``, it is added regardless of ``autopopulate_metadata``,
as it describes the returned arrays rather than the dataset record:

.. python::

    {
        "record_id": 123,
        "augmentations": {
            "HorizontalFlip": {},
            "OneOf/RandomBrightnessContrast": {"alpha": 1.02, "beta": -0.03},
        },
    }

Both keys are reserved. A record defining one of them keeps its own value,
and the loader warns rather than overwriting it.

When a batch augmentation combines several samples, metadata from the input
samples is preserved in ``"batch_augmentation_metadata"``:

.. python::

    {
        "record_id": 123,

        "batch_augmentation_metadata": [
            {
                "input_index": 0,
                "sample_metadata": {"record_id": 123},
            },
            {
                "input_index": 1,
                "sample_metadata": {"record_id": 456},
            },
        ],
    }

**Annotation metadata is different:** values under
``annotation["metadata"]`` are converted into label tasks such as
``"detection/metadata/weather"``.


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

__all__ = [
    "LOADERS_REGISTRY",
    "BaseLoader",
    "LuxonisLoader",
]
