from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import numpy as np
from albumentations.core.transforms_interface import BasicTransform
from typing_extensions import override


class BatchBasedTransform(ABC, BasicTransform):
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
    @override
    def targets(self) -> Dict[str, Callable]:
        return {
            "image_batch": self.apply_to_image_batch,
            "mask_batch": self.apply_to_mask_batch,
            "bboxes_batch": self.apply_to_bboxes_batch,
            "keypoints_batch": self.apply_to_keypoints_batch,
        }

    @property
    @override
    def targets_as_params(self) -> List[str]:
        """List of augmentation targets.

        @rtype: List[str]
        @return: Output list of augmentation targets.
        """
        return ["image_batch"]

    @override
    def update_params(self, params: Dict[str, Any], **_) -> Dict[str, Any]:
        return params

    @override
    def update_params_shape(
        self, params: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        shape = (
            data["image"].shape
            if "image" in data
            else data["image_batch"][0].shape
        )
        params["shape"] = shape
        params.update({"cols": shape[1], "rows": shape[0]})
        return params

    @abstractmethod
    def apply_to_image_batch(
        self, image_batch: List[np.ndarray], **params
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_mask_batch(
        self, mask_batch: List[np.ndarray], **params
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_bboxes_batch(
        self, bboxes_batch: List[np.ndarray], **params
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_keypoints_batch(
        self, keypoints_batch: List[np.ndarray], **params
    ) -> np.ndarray: ...
