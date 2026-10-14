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


Keypoint Names
==============

A spatial transform moves each keypoint, but it keeps the column of the
keypoint, so it keeps its name. After a horizontal flip, the keypoint
called ``left_eye`` sits where the right eye of the person was. The engine
therefore renames the keypoints after the whole pipeline, from what each
name means (see `KeypointAnnotation`):

    - A mirror partner, such as ``left_eye``, swaps with its partner when
      the pipeline mirrored the picture: after a horizontal or a vertical
      flip, a transpose, or any odd number of them. A rotation keeps it.
    - A keypoint with a place in the picture, such as the ``top_left``
      corner of a parking space, takes the name of the place where it is.
    - Any other keypoint keeps its name.

A configuration therefore uses the plain transforms, also for keypoints:

.. python::

    [
        {
            "name": "HorizontalFlip",
            "params": {
                "p": 0.5,
            },
        },
        {
            "name": "RandomRotate90",
            "params": {
                "p": 0.5,
            },
        },
    ]

The engine does not need to know what a transform does. Three hidden probe
points ride along with each instance, also through `Mosaic4`. If the
triangle they form comes out reversed, the map at that instance was a
mirror. A custom transform thus works as well as a built-in one. A keypoint
that a transform drops, as ``ElasticTransform`` drops the keypoints outside
the image, comes back as unlabeled.

The renaming has these limits:

    - A place in the picture is named by the nearest corner or side. Near a
      rotation by 45°, two corners are about as near to the top
      left, and the name can change from one sample to the next. Keep the
      rotations under 45° when the names must be stable. Mirror
      partners have no such limit.
    - A warp so strong that it folds an object over itself, so that its
      outline crosses, leaves no side to tell. The probes then give one of
      the two answers.

`HorizontalSymmetricKeypointsFlip`, `VerticalSymmetricKeypointsFlip` and
`TransposeSymmetricKeypoints` are deprecated. In the engine, they flip like
the plain transforms, and their ``keypoint_pairs`` give the mirror partners
of a task that stores none.


Class Names
===========

A class name can say a side or a direction too, as ``left_hand`` and
``arrow_up`` do. After a horizontal flip, the box of a ``left_hand`` holds
a right hand. The engine therefore gives each class a new name after the
pipeline, from the class meanings of its task (see
`luxonis_ml.ldf.class_meaning` for the forms):

    - A mirror pair, such as ``left_hand`` and ``right_hand``, swaps after
      each mirror of the image. A turn keeps it.
    - A direction set, such as ``arrow_up`` to ``arrow_left``, turns and
      mirrors with the image. A quarter turn to the left makes an
      ``arrow_up`` an ``arrow_left``.
    - An upright class, such as ``digit_6``, has no valid sample after a
      turn or a mirror. An upright pair, such as ``b`` and ``d``, allows
      the horizontal mirror and swaps after it.
    - A symmetry table gives the class after each turn and mirror, or
      says that the result is no valid sample.
    - Any other class keeps its name.

The engine gives the new class to each box, to the classes of the image
(the classification vector), and to each channel of a semantic
segmentation. The configuration uses the plain transforms, and the class
meanings come from the dataset:

.. python::

    dataset.set_class_metadata(
        task="hands",
        classes={
            "left_hand": {
                "mirror": "right_hand",
            },
        },
    )
    loader = LuxonisLoader(
        dataset,
        augmentation_config=[
            {
                "name": "HorizontalFlip",
                "params": {
                    "p": 0.5,
                },
            },
        ],
    )

Three hidden probe points at the center of the image measure what the
spatial transforms did to the whole image. The engine rounds the result
to the nearest turn or mirror, so a turn under 45° keeps every
class, and a turn over 45° counts as a quarter turn.

When a class of the sample has no valid sample after the measured turn or
mirror, the engine applies one more flip or turn to the image and to all
of its labels. It takes the first one, mirrors before turns, after which
every class of the sample is valid. An upright ``digit_6`` thus comes back
as it was, and an upright ``b`` after a vertical flip comes back as a
``d``, turned by 180°. The engine logs this once.

The class names have these limits:

    - A transform that moves parts of the image in different ways, such
      as ``RandomGridShuffle``, has no turn or mirror for the whole image.
      Such a sample keeps its classes, and the engine logs a warning once.
    - The engine measures the whole image, not each object. A crop or a
      warp that turns one part of the image more than another can give an
      object a class that does not fit it.
    - A custom engine gets the class meanings only when its ``__init__``
      takes the ``class_symmetry`` argument. `LuxonisLoader` logs a warning
      for any other engine.
    - An instance segmentation and the keypoints of a box keep the class of
      the box row that they belong to. Their own arrays hold no class.


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

When the ``__init__`` of an engine takes a ``keypoint_metadata`` parameter,
`LuxonisLoader` passes the `KeypointMetadata` of each keypoint task, keyed by
task name.


Tips and Tricks
===============

Rotated Bounding Boxes Are Too Large
------------------------------------

``Affine``, ``Rotate``, ``SafeRotate``, and ``ShiftScaleRotate`` can leave
bounding boxes much larger than the objects in them once they rotate or shear
the image. Tall or wide objects, such as standing people, show it the most.

.. figure::
   https://raw.githubusercontent.com/luxonis/luxonis-ml/30a7530c1725538e1e9816b3d188e9df400db636/luxonis_ml/data/augmentations/media/bbox_rotation_original.png
   :width: 600px

   The sample before augmentation.

.. figure::
   https://raw.githubusercontent.com/luxonis/luxonis-ml/30a7530c1725538e1e9816b3d188e9df400db636/luxonis_ml/data/augmentations/media/bbox_rotation_largest_box.png
   :width: 600px

   Rotated with the default ``rotate_method="largest_box"``.

.. figure::
   https://raw.githubusercontent.com/luxonis/luxonis-ml/30a7530c1725538e1e9816b3d188e9df400db636/luxonis_ml/data/augmentations/media/bbox_rotation_ellipse.png
   :width: 600px

   The same rotation with ``rotate_method="ellipse"``.

A box says nothing about the shape of the object inside it, so after a
rotation or a shear Albumentations has to guess the new box. The
``rotate_method`` parameter of these transforms selects the guess:

``"largest_box"`` (default)
    Takes the box around the four transformed corners, as if the object
    filled its whole box. The box is never too small, but often too large.
    A :math:`w \times h` box rotated by :math:`\theta` becomes
    :math:`w \left|\cos\theta\right| + h \left|\sin\theta\right|` wide.

``"ellipse"``
    Takes the box around the transformed ellipse inscribed in the box. The
    box is tighter, but cuts off the corners of objects that fill their box.
    The same box becomes
    :math:`\sqrt{w^2 \cos^2\theta + h^2 \sin^2\theta}` wide.

A :math:`120 \times 315` person box rotated by 20° becomes 221 pixels wide
with ``"largest_box"`` and 156 with ``"ellipse"``. A square object rotated
by 45° spans :math:`1.41 w`, but ``"ellipse"`` gives it :math:`w`. Without
rotation or shear, both methods give the same box.

Use ``"ellipse"`` when objects do not reach the corners of their boxes, as
with people or animals. Keep ``"largest_box"`` for rectangular objects such
as cars or screens. `Towards Rotation Invariance in Object Detection
<https://arxiv.org/abs/2109.13488>`_ shows that oversized boxes can make a
detector worse than training without rotation at all.

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

Add ``--list-augmentations`` to print the sampled rotation and scale of
each ``Affine``. Albumentations does not report the sampled shear.
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
