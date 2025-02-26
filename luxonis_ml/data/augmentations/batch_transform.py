from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import albumentations as A
import numpy as np
from typing_extensions import override


class BatchTransform(ABC, A.DualTransform):
    def __init__(self, batch_size: int, **kwargs):
        """Batch transformation that combines multiple images and
        associated labels into one.

        @param batch_size: Batch size needed for augmentation to work
        @type batch_size: int
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size

    @property
    @override
    def targets(self) -> Dict[str, Callable]:
        targets = super().targets
        targets.update(
            {
                "instance_mask": self.apply_to_instance_mask,
                "array": self.apply_to_array,
                "classification": self.apply_to_classification,
                "metadata": self.apply_to_metadata,
            }
        )
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
        return np.concatenate([arr for arr in array_batch if arr.size > 0])

    def apply_to_classification(
        self, classification_batch: List[np.ndarray], **_
    ) -> np.ndarray:
        for i in range(len(classification_batch)):
            if classification_batch[i].size == 0:
                classification_batch[i] = np.zeros(1)
        return np.clip(sum(classification_batch), 0, 1)

    def apply_to_metadata(
        self, metadata_batch: List[np.ndarray], **_
    ) -> np.ndarray:
        if all(arr.size == 0 for arr in metadata_batch):
            return np.array([])
        return np.concatenate([arr for arr in metadata_batch if arr.size > 0])

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
