from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import albumentations as A
import numpy as np
from typing_extensions import override


class BatchBasedTransform(ABC, A.DualTransform):
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
    def targets(self) -> Dict[str, Callable[..., Any]]:
        targets = super().targets
        targets["masks"] = self.apply_to_instance_masks
        return targets

    @abstractmethod
    def apply(self, image_batch: List[np.ndarray], **kwargs) -> np.ndarray: ...

    @abstractmethod
    def apply_to_mask(
        self, mask_batch: List[np.ndarray], **params
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_bboxes(
        self, bboxes_batch: List[np.ndarray], **params
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_keypoints(
        self, keypoints_batch: List[np.ndarray], **params
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_instance_masks(
        self, masks_batch: List[List[np.ndarray]], **params
    ) -> List[np.ndarray]: ...

    @override
    def update_params(self, params: Dict[str, Any], **_) -> Dict[str, Any]:
        return params

    @override
    def update_params_shape(
        self, params: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        shape = data["image"][0].shape
        params["shape"] = shape
        params.update({"cols": shape[1], "rows": shape[0]})
        return params
