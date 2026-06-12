from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import albumentations as A
import numpy as np
from typing_extensions import override


class BatchTransform(ABC, A.DualTransform):
    """Base class for transforms that combine multiple samples.

    Attributes:
        batch_size: Number of samples consumed by one application of the
            transform.

    """

    def __init__(self, batch_size: int, **kwargs):
        """Create a batch transformation.

        Batch transformations combine multiple images and their labels into
        one sample.

        Args:
            batch_size: Number of samples required by the augmentation.
            kwargs: Additional arguments passed to the parent
                Albumentations transform.

        """
        super().__init__(**kwargs)

        self.batch_size = batch_size

    @property
    @override
    def targets(self) -> dict[str, Callable]:
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
    def apply(self, image_batch: list[np.ndarray], **kwargs) -> np.ndarray:
        r"""Apply the transformation to a batch of images.

        Args:
            image_batch: Images to transform. Each image should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Single transformed image resulting from the combination of the input
            batch.

        """
        ...

    @abstractmethod
    def apply_to_mask(
        self, masks_batch: list[np.ndarray], **kwargs
    ) -> np.ndarray:
        r"""Apply the transformation to a batch of semantic segmentation masks.

        Args:
            masks_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Single transformed mask resulting from the combination of the input
            batch.

        """

    @abstractmethod
    def apply_to_bboxes(
        self, bboxes_batch: list[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Apply the transformation to a batch of bounding boxes.

        Args:
            bboxes_batch: A batch of bounding boxes to transform.
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Transformed bounding boxes resulting from the combination of
            the input batch.

        """
        ...

    @abstractmethod
    def apply_to_keypoints(
        self, keypoints_batch: list[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Apply the transformation to a batch of keypoints.

        Args:
            keypoints_batch: A batch of keypoints to transform.
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Transformed keypoints resulting from the combination of the input
            batch.

        """
        ...

    @abstractmethod
    def apply_to_instance_mask(
        self, masks_batch: list[np.ndarray], **kwargs
    ) -> np.ndarray:
        r"""Apply the transformation to a batch of instance segmentation masks.

        Args:
            masks_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, N\right)`, where :math:`N`
                is the number of instances.
            **kwargs: Additional implementation-specific arguments.

        Returns:
            A single instance masks of shape
            :math:`\left(H_{out}, W_{out}, N\right)`.

        """
        ...

    def apply_to_array(
        self, array_batch: list[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Apply the transformation to a batch of generic arrays.

        Note:
            The default implementation simply concatenates non-empty
            arrays. Override this method if a different behavior is desired.

        Args:
            array_batch: A batch of arrays to transform.
            **kwargs: Additional implementation-specific arguments.

        """
        return np.concatenate([arr for arr in array_batch if arr.size > 0])

    def apply_to_classification(
        self, classification_batch: list[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Apply the transformation to a batch of classification labels.

        Note:
            The default implementation treats classification labels as binary
            and returns their logical OR. Override this method if a different
            behavior is desired.

        Args:
            classification_batch: A batch of classification labels to transform.
            **kwargs: Additional implementation-specific arguments.

        """
        for i in range(len(classification_batch)):
            if classification_batch[i].size == 0:
                classification_batch[i] = np.zeros(1)
        return np.clip(sum(classification_batch), 0, 1)

    def apply_to_metadata(
        self, metadata_batch: list[np.ndarray], **kwargs
    ) -> np.ndarray:
        """Apply the transformation to a batch of metadata arrays.

        Note:
            The default implementation concatenates non-empty metadata arrays.
            Override this method if a different behavior is desired.

        Args:
            metadata_batch: A batch of metadata arrays to transform.
            **kwargs: Additional implementation-specific arguments.

        """
        if all(arr.size == 0 for arr in metadata_batch):
            return np.array([])
        return np.concatenate([arr for arr in metadata_batch if arr.size > 0])

    @override
    def update_transform_params(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Update the parameters dictionary with the shape of the input images.

        Args:
            params: Parameters to be updated
            data: Input data dictionary containing images/volumes

        Returns:
            Updated parameters dictionary with shape and
            transform-specific parameters.

        """
        image_batch = data["image"]
        params["image_shapes"] = [
            tuple(image.shape[:2]) for image in image_batch
        ]
        return params
