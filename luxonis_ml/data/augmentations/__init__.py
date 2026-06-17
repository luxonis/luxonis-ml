r"""Augmentation engines and custom transforms for LDF samples.

This package provides the augmentation interface used by `LuxonisLoader`.
The default implementation is `AlbumentationsEngine`, which adapts LDF labels
to Albumentations targets before transformation and converts them back after
transformation.

Augmentation configuration is a list of records. Each record contains a
``name`` identifying an Albumentations transform or a transform registered in
`TRANSFORMATIONS`, optional ``params``, optional ``use_for_resizing``, and
optional stage filtering through ``apply_on_stages``.

.. python::

    [
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {
            "name": "Mosaic4",
            "params": {"height": 640, "width": 640, "p": 1.0},
        },
    ]

The engine groups transforms by behavior rather than preserving the exact
input order:

    1. Batch transforms, such as `MixUp` and `Mosaic4`.
    2. Spatial transforms, such as Albumentations dual transforms.
    3. Custom basic transforms.
    4. Pixel-only transforms.

Resize handling is part of the engine. A transform marked with
``use_for_resizing`` is used as the resize stage; otherwise the engine falls
back to a regular resize or `LetterboxResize`, depending on the loader's
aspect-ratio setting.

Standard Albumentations flip transforms such as ``HorizontalFlip``,
``VerticalFlip``, and ``Transpose`` flip keypoint coordinates but do not swap
semantic left/right keypoint labels. For symmetric keypoint structures, use the
Luxonis custom transforms `HorizontalSymmetricKeypointsFlip`,
`VerticalSymmetricKeypointsFlip`, and `TransposeSymmetricKeypoints`.

Batch transforms multiply the number of source samples required by the loader.
For example, a pipeline that contains `MixUp` and `Mosaic4` requires
:math:`8 = 2 \cdot 4` samples for each augmented output.

Custom augmentation engines can be added by subclassing `AugmentationEngine`.
Subclasses are automatically registered in `AUGMENTATION_ENGINES`.


Custom Transforms
=================

Custom transforms follow Albumentations conventions. Subclass an appropriate
base class such as ``DualTransform`` or ``ImageOnlyTransform``, implement the
target methods needed by your labels, register the class in
`luxonis_ml.data.augmentations.custom.TRANSFORMATIONS`, and reference the
class name in loader configuration.

.. python::

    from albumentations import DualTransform
    from luxonis_ml.data.augmentations.custom import TRANSFORMATIONS

    class CustomTransform(DualTransform):
        def apply(self, image, **kwargs):
            return image

        def apply_to_mask(self, mask, **kwargs):
            return mask

        def apply_to_bboxes(self, bboxes, **kwargs):
            return bboxes

        def apply_to_keypoints(self, keypoints, **kwargs):
            return keypoints

    TRANSFORMATIONS.register(module=CustomTransform)

    augmentation_config = [
        {"name": "CustomTransform", "params": {"p": 1.0}},
    ]


Engine Interface
================

A custom engine should subclass `AugmentationEngine` and implement:

    - ``__init__`` to consume output size, class count, configuration,
      aspect-ratio behavior, pipeline stage, and target metadata;
    - ``apply`` to transform a batch of images and labels and return the
      transformed values;
    - ``batch_size`` to tell `LuxonisLoader` how many source samples are
      needed per augmented output.
"""

from .albumentations_engine import AlbumentationsEngine
from .base_engine import AUGMENTATION_ENGINES, AugmentationEngine
from .batch_compose import BatchCompose
from .batch_transform import BatchTransform
from .custom import LetterboxResize, MixUp, Mosaic4

__all__ = [
    "AUGMENTATION_ENGINES",
    "AlbumentationsEngine",
    "AugmentationEngine",
    "BatchCompose",
    "BatchTransform",
    "LetterboxResize",
    "MixUp",
    "Mosaic4",
]
