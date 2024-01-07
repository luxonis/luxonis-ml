import numpy as np
from typing import Dict, Any, Sequence, Callable, List
from albumentations.core.transforms_interface import (
    BoxType,
    KeypointType,
    BasicTransform,
)


class BatchBasedTransform(BasicTransform):
    """Transform for multi-image."""

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image_batch": self.apply_to_image_batch,
            "mask_batch": self.apply_to_mask_batch,
            "bboxes_batch": self.apply_to_bboxes_batch,
            "keypoints_batch": self.apply_to_keypoints_batch,
        }

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # This overwrites the `supre().update_params(...)`
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
