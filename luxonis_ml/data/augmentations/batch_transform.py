from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from albumentations.core.transforms_interface import (
    BasicTransform,
    BoxType,
    KeypointType,
)


class BatchBasedTransform(BasicTransform):
    def __init__(self, batch_size: int, **kwargs):
        """Transform for multi-image.

        @param batch_size: Batch size needed for augmentation to work
        @type batch_size: int
        @param kwargs: Additional BasicTransform parameters
        @type kwargs: Any
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image_batch": self.apply_to_image_batch,
            "mask_batch": self.apply_to_mask_batch,
            "bboxes_batch": self.apply_to_bboxes_batch,
            "keypoints_batch": self.apply_to_keypoints_batch,
        }

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # This overwrites the `super().update_params(...)`
        return params

    def apply_to_image_batch(
        self, image_batch: Sequence[BoxType], **params
    ) -> List[np.ndarray]:
        raise NotImplementedError(
            "Method apply_to_image_batch is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_mask_batch(
        self, mask_batch: Sequence[BoxType], **params
    ) -> List[np.ndarray]:
        raise NotImplementedError(
            "Method apply_to_mask_batch is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_bboxes_batch(
        self, bboxes_batch: Sequence[BoxType], **params
    ) -> List[BoxType]:
        raise NotImplementedError(
            "Method apply_to_bboxes_batch is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_keypoints_batch(
        self, keypoints_batch: Sequence[BoxType], **params
    ) -> List[KeypointType]:
        raise NotImplementedError(
            "Method apply_to_keypoints_batch is not implemented in class "
            + self.__class__.__name__
        )
