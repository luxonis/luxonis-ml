from typing import Type

import albumentations as A

from luxonis_ml.utils import Registry

from .letterbox_resize import LetterboxResize
from .mixup import MixUp
from .mosaic import Mosaic4
from .symetric_keypoints_flip import (
    HorizontalSymetricKeypointsFlip,
    VerticalSymetricKeypointsFlip,
)

TRANSFORMATIONS: Registry[Type[A.BasicTransform]] = Registry(
    "albumentations_transformations"
)

TRANSFORMATIONS.register(module=LetterboxResize)
TRANSFORMATIONS.register(module=MixUp)
TRANSFORMATIONS.register(module=Mosaic4)
TRANSFORMATIONS.register(module=HorizontalSymetricKeypointsFlip)
TRANSFORMATIONS.register(module=VerticalSymetricKeypointsFlip)

__all__ = [
    "TRANSFORMATIONS",
    "HorizontalSymetricKeypointsFlip",
    "LetterboxResize",
    "MixUp",
    "Mosaic4",
    "VerticalSymetricKeypointsFlip",
]
