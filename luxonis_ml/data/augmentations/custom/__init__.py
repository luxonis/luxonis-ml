r"""Built-in custom Albumentations transforms.

The `TRANSFORMATIONS` registry contains Luxonis transforms that can be used by
`AlbumentationsEngine` through augmentation configuration records. Registered
transforms are referenced by class name in the ``name`` field, the same way as
standard Albumentations transforms.

Built-in custom transforms include:

    - `LetterboxResize` for aspect-ratio preserving resize and padding.
    - `MixUp` for blending pairs of images and compatible labels.
    - `Mosaic4` for composing a :math:`2 \times 2` image mosaic.
    - Symmetric keypoint flips and transposition helpers for keypoint tasks.

User-defined transforms can be registered with ``TRANSFORMATIONS.register`` and
then used by name in loader augmentation configuration.

.. python::

    TRANSFORMATIONS.register(module=CustomTransform)

    augmentation_config = [
        {
            "name": "CustomTransform",
            "params": {"p": 1.0},
        },
    ]

"""

import albumentations as A

from luxonis_ml.utils import Registry

from .letterbox_resize import LetterboxResize
from .mixup import MixUp
from .mosaic import Mosaic4
from .symmetric_keypoints_flip import (
    HorizontalSymmetricKeypointsFlip,
    TransposeSymmetricKeypoints,
    VerticalSymmetricKeypointsFlip,
)

TRANSFORMATIONS: Registry[type[A.BasicTransform]] = Registry(
    "albumentations_transformations"
)

TRANSFORMATIONS.register(module=LetterboxResize)
TRANSFORMATIONS.register(module=MixUp)
TRANSFORMATIONS.register(module=Mosaic4)
TRANSFORMATIONS.register(module=HorizontalSymmetricKeypointsFlip)
TRANSFORMATIONS.register(module=VerticalSymmetricKeypointsFlip)
TRANSFORMATIONS.register(module=TransposeSymmetricKeypoints)

__all__ = [
    "TRANSFORMATIONS",
    "HorizontalSymmetricKeypointsFlip",
    "LetterboxResize",
    "MixUp",
    "Mosaic4",
    "TransposeSymmetricKeypoints",
    "VerticalSymmetricKeypointsFlip",
]
