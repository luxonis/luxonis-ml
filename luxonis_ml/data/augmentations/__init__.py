r"""Augmentation engines and custom transforms for LDF samples.

This package provides the augmentation interface used by `LuxonisLoader`.
The default implementation is `AlbumentationsEngine`, which adapts LDF labels
to Albumentations targets before transformation and converts them back after
transformation.

.. contents:: Table of Contents
   :depth: 2


Configuration
=============

Augmentation configuration is a list of records. Each record contains a
``name`` identifying an Albumentations transform or a transform registered in
`TRANSFORMATIONS`, optional ``params``, optional ``use_for_resizing``, and
optional stage filtering through ``apply_on_stages``. When
``apply_on_stages`` is omitted, the transform applies to ``"train"``.

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

    1. Batch transforms, such as `MixUp` or `Mosaic4`.
    2. Spatial transforms, such as Albumentations dual transforms.
    3. Custom basic transforms.
    4. Pixel-only transforms.

Resize handling is part of the engine. A transform marked with
``use_for_resizing`` is used as the resize stage; otherwise the engine falls
back to a regular resize or `LetterboxResize`, depending on the loader's
aspect-ratio setting. If the selected resize transform has probability
``p < 1``, it stays in the resize stage and the remaining probability mass is
filled by the default resize in an always-on ``OneOf``. The resize stage is
applied before pixel-only transforms when downscaling saves work, and after
pixel-only transforms when upscaling or preserving size.

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

`LuxonisLoader` builds the engine for you. Pass the configuration as a list
of records, or as the path of a YAML or JSON file that holds the same list:

.. python::

    from luxonis_ml.data import LuxonisDataset, LuxonisLoader

    loader = LuxonisLoader(
        LuxonisDataset("parking_lot"),
        view="train",
        augmentation_engine="albumentations",
        augmentation_config="augmentations.yaml",
        height=256,
        width=320,
        keep_aspect_ratio=True,
        color_space="RGB",
    )

    for sample in loader:
        images = sample.images
        labels = sample.labels


Custom Transforms
=================

Custom transforms follow Albumentations conventions. Subclass an appropriate
base class such as ``DualTransform`` or ``ImageOnlyTransform``, implement the
target methods needed by your labels, register the class in
`luxonis_ml.data.augmentations.custom.TRANSFORMATIONS`, and reference the
class name in loader configuration. The `Albumentations guide
<https://albumentations.ai/docs/4-advanced-guides/creating-custom-transforms/>`__
explains the base classes and the target methods.

To combine several samples into one, subclass `BatchTransform` instead of an
Albumentations base class. `CutMix`, `MixUp`, and `Mosaic4` are the built-in
examples. Registration and configuration work the same way.

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

Engines may also override `AugmentationEngine.applied_augmentations` to
report the configured paths and runtime parameters of their latest call.


Tips and Tricks
===============

Rotated Bounding Boxes Are Too Large
------------------------------------

``Affine``, ``Rotate``, ``SafeRotate``, and ``ShiftScaleRotate`` can leave
bounding boxes much larger than the objects in them once they rotate or shear
the image. Tall and wide objects, such as standing people, show it the most.

.. figure::
   https://raw.githubusercontent.com/luxonis/luxonis-ml/main/luxonis_ml/data/augmentations/media/bbox_rotation_original.png
   :width: 600px
   :loading: embed

   The sample before augmentation.

.. figure::
   https://raw.githubusercontent.com/luxonis/luxonis-ml/main/luxonis_ml/data/augmentations/media/bbox_rotation_largest_box.png
   :width: 600px
   :loading: embed

   Rotated with the default ``rotate_method="largest_box"``.

.. figure::
   https://raw.githubusercontent.com/luxonis/luxonis-ml/main/luxonis_ml/data/augmentations/media/bbox_rotation_ellipse.png
   :width: 600px
   :loading: embed

   The same rotation with ``rotate_method="ellipse"``.

The boxes are not wrong. A box says nothing about the shape of the object
inside it, so the true box after a rotation or a shear cannot be known.
Albumentations has to guess, and the ``rotate_method`` parameter of these
transforms selects how:

``"largest_box"`` (default)
    Assumes the object fills the whole box, corners included, and takes the
    box around its four transformed corners. The box is often too large but
    never too small, which suits rectangular objects such as cars, signs,
    and screens. A :math:`w \times h` box rotated by :math:`\theta` becomes
    :math:`w \left|\cos\theta\right| + h \left|\sin\theta\right|` wide.

``"ellipse"``
    Assumes the object is the ellipse inscribed in the box, and takes the
    box around the transformed ellipse. The box is close but can be too
    small, which suits round or irregular objects such as people, animals,
    and balls. The same box becomes
    :math:`\sqrt{w^2 \cos^2\theta + h^2 \sin^2\theta}` wide.

A :math:`120 \times 315` person box rotated by 20° becomes 221 pixels wide
with ``"largest_box"`` and 156 with ``"ellipse"``. Without rotation and shear
the two methods agree; scaling, translation, and flips are not affected.

``"largest_box"`` is the default because it never cuts off part of the
object. ``"ellipse"`` has two drawbacks:

- It cuts off the corners of objects that fill their box. A square object
  rotated by 45° spans :math:`1.41 w`, but ``"ellipse"`` gives :math:`w`.
- It changes the training labels of an existing configuration, so compare
  metrics only between runs that use the same method.

Prefer ``"ellipse"`` when most objects are round or irregular, which holds
for most natural images. Oversized boxes can make a detector worse than
training without rotation at all.

.. python::

    [
        {
            "name": "Affine",
            "params": {
                "rotate": [-30, 30],
                "shear": [-15, 15],
                "rotate_method": "ellipse",
            },
        },
    ]

Check the augmented boxes before you train:

.. code-block:: bash

    luxonis_ml data inspect <dataset> --aug-config augmentations.yaml

``--list-augmentations`` shows the sampled rotation and scale of ``Affine``,
but not the sampled shear, which Albumentations does not report.

See:
    `Towards Rotation Invariance in Object Detection
    <https://arxiv.org/abs/2109.13488>`_.
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
