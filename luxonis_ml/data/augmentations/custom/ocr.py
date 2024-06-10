from typing import Tuple
import random
import albumentations as A

from albumentations.core.transforms_interface import ImageOnlyTransform

import numpy as np
from ..utils import AUGMENTATIONS


@AUGMENTATIONS.register_module()
class OCRAugmentation(ImageOnlyTransform):

    def __init__(self, image_size: Tuple[int, int], is_rgb: bool, is_train: bool):
        """

        @param image_size: OCR model input shape.
        @type image_size: Tuple[int, int]
        @param is_rgb: True if image is RGB. False if image is GRAY.
        @type is_rgb: bool
        @param is_train: True if image is train. False if image is val/test.
        @type is_train: bool
        """
        super(OCRAugmentation, self).__init__()
        self.transforms = A.Compose(
            transforms=[
                A.OneOf(
                    transforms=[
                        A.RandomScale(
                            scale_limit=(-0.5, 0),  # (1 + low, 1 + high) => (0.5, 1) : random downscale (min half size)
                            always_apply=True,
                            p=1
                        ),
                        A.MotionBlur(
                            blur_limit=(19, 21),
                            p=1.0,
                            always_apply=True,
                            allow_shifted=False
                        ),
                        A.Affine(
                            translate_percent=(0.05, 0.07),
                            scale=(0.7, 1.0),
                            rotate=(-7, 7),
                            shear=(5, 35),
                            always_apply=True,
                            p=1
                        )
                    ],
                    p=0.2
                ),
                A.OneOf(
                    transforms=[
                        A.ISONoise(
                            color_shift=(0.01, 0.1),
                            intensity=(0.1, 1.0),
                            always_apply=True
                        ),
                        A.GaussianBlur(
                            blur_limit=(7, 9),  # kernel
                            sigma_limit=(0.1, 0.5),
                            always_apply=True,
                            p=0.2
                        ),
                        A.ColorJitter(
                            brightness=(0.11, 1.0),
                            contrast=0.5,
                            saturation=0.5
                        )
                    ],
                    p=0.2
                ),
                A.Compose(  # resize to image_size with aspect ratio, pad if needed
                    transforms=[
                        A.LongestMaxSize(max_size=max(image_size), interpolation=1),
                        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=0,
                                      value=(0, 0, 0))
                    ]
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ) if is_rgb else A.Normalize(
                    mean=0.4453,
                    std=0.2692
                )

            ]
        ) if is_train else A.Compose(
            transforms=[
                A.Compose(  # resize to image_size with aspect ratio, pad if needed
                    transforms=[
                        A.LongestMaxSize(max_size=max(image_size), interpolation=1),
                        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=0,
                                      value=(0, 0, 0))
                    ]
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ) if is_rgb else A.Normalize(
                    mean=0.4453,
                    std=0.2692
                )
            ]
        )

    def apply(
        self,
        img: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Applies a series of OCR augmentations.

        @param img: Input image to which resize is applied.
        @type img: np.ndarray
        @return: Image with applied OCR augmentations.
        @rtype: np.ndarray
        """

        img_out = self.transforms(image=img)["image"]
        return img_out
