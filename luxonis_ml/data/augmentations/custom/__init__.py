from typing import Type

import albumentations as A

from luxonis_ml.utils import Registry

from .letterbox_resize import LetterboxResize
from .mixup import MixUp
from .mosaic import Mosaic4

TRANSFORMATIONS: Registry[Type[A.BasicTransform]] = Registry(
    "albumentation_transformations"
)

TRANSFORMATIONS.register_module(module=LetterboxResize)
TRANSFORMATIONS.register_module(module=MixUp)
TRANSFORMATIONS.register_module(module=Mosaic4)

__all__ = ["LetterboxResize", "MixUp", "Mosaic4"]
