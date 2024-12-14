from abc import ABC, abstractmethod
from typing import Any, Dict, List

import albumentations as A
import numpy as np
from typing_extensions import override


class BatchTransform(ABC, A.DualTransform):
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
    def targets(self) -> Dict[str, Any]:
        targets = super().targets
        targets["instance_mask"] = self.apply_to_instance_mask
        return targets

    @abstractmethod
    def apply(self, image_batch: List[np.ndarray], **kwargs) -> np.ndarray: ...

    @abstractmethod
    def apply_to_mask(
        self, mask_batch: List[np.ndarray], **kwargs
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_bboxes(
        self, bboxes_batch: List[np.ndarray], **kwargs
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_keypoints(
        self, keypoints_batch: List[np.ndarray], **kwargs
    ) -> np.ndarray: ...

    @abstractmethod
    def apply_to_instance_mask(
        self, masks_batch: List[np.ndarray], **kwargs
    ) -> np.ndarray: ...

    def apply_to_array(self, array_batch: List[np.ndarray], **_) -> np.ndarray:
        raise NotImplementedError

    def apply_to_classification(
        self, classification_batch: List[np.ndarray], **_
    ) -> np.ndarray:
        raise NotImplementedError

    def apply_to_metadata(
        self, metadata_batch: List[Dict[str, Any]], **_
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

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
