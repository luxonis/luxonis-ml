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
