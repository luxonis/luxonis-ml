from itertools import chain
from typing import Any

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override


class HorizontalSymmetricKeypointsFlip(A.DualTransform):
    def __init__(self, keypoint_pairs: list[tuple[int, int]], p: float = 0.5):
        """Flip an image and symmetric keypoints horizontally.

        Bounding boxes and segmentation masks are flipped as well.

        Args:
            keypoint_pairs: Pairs of keypoint indices to swap after the
                flip.
            p: Probability of applying the augmentation.

        """
        super().__init__(p=p)
        self.keypoint_pairs = keypoint_pairs
        self.n_keypoints = len(set(chain.from_iterable(keypoint_pairs)))

    @property
    @override
    def targets(self) -> dict[str, Any]:
        targets = super().targets
        targets["instance_mask"] = self.apply_to_mask
        targets["segmentation"] = self.apply_to_mask
        return targets

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get parameters dependent on the targets.

        Args:
            params: Existing augmentation parameters.
            data: Input data.

        Returns:
            Parameters derived from the input targets.

        """
        orig_height, orig_width, _ = params["shape"]
        return {
            "orig_width": orig_width,
            "orig_height": orig_height,
        }

    @override
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Flip an image horizontally.

        Args:
            img: Image to flip.
            params: Additional transform parameters.

        Returns:
            Flipped image.

        """
        return cv2.flip(img, 1)

    @override
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """Flip a segmentation mask horizontally.

        Args:
            img: Segmentation mask to flip.
            params: Additional transform parameters.

        Returns:
            Flipped segmentation mask.

        """
        return cv2.flip(img, 1)

    @override
    def apply_to_bboxes(self, bboxes: np.ndarray, **params) -> np.ndarray:
        """Flip bounding boxes horizontally.

        Args:
            bboxes: Bounding boxes to flip.
            params: Additional transform parameters.

        Returns:
            Flipped bounding boxes.

        """
        if bboxes.size == 0:
            return bboxes

        flipped = bboxes.copy()
        flipped[:, [0, 2]] = 1 - flipped[:, [2, 0]]
        return flipped

    @override
    def apply_to_keypoints(
        self, keypoints: np.ndarray, orig_width: int, **params
    ) -> np.ndarray:
        """Flip keypoints horizontally and swap symmetric pairs.

        Args:
            keypoints: Keypoints to flip.
            orig_width: Original image width.
            params: Additional transform parameters.

        Returns:
            Flipped keypoints.

        """
        if keypoints.size == 0:
            return keypoints

        keypoints = keypoints.copy()

        keypoints[:, 0] = orig_width - keypoints[:, 0]

        total_keypoints = keypoints.shape[0]
        if total_keypoints % self.n_keypoints != 0:
            raise ValueError(
                "Total number of keypoints is not a multiple of n_keypoints defined by keypoint_pairs."
            )
        num_instances = total_keypoints // self.n_keypoints

        for instance in range(num_instances):
            offset = instance * self.n_keypoints
            for i, j in self.keypoint_pairs:
                idx1, idx2 = offset + i, offset + j
                tmp = keypoints[idx1].copy()
                keypoints[idx1] = keypoints[idx2]
                keypoints[idx2] = tmp

        return keypoints


class VerticalSymmetricKeypointsFlip(A.DualTransform):
    def __init__(self, keypoint_pairs: list[tuple[int, int]], p: float = 0.5):
        """Flip an image and symmetric keypoints vertically.

        Bounding boxes and segmentation masks are flipped as well.

        Args:
            keypoint_pairs: Pairs of keypoint indices to swap after the
                flip.
            p: Probability of applying the augmentation.

        """
        super().__init__(p=p)
        self.keypoint_pairs = keypoint_pairs
        self.n_keypoints = len(set(chain.from_iterable(keypoint_pairs)))

    @property
    @override
    def targets(self) -> dict[str, Any]:
        targets = super().targets
        targets["instance_mask"] = self.apply_to_mask
        targets["segmentation"] = self.apply_to_mask
        return targets

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get parameters dependent on the targets.

        Args:
            params: Existing augmentation parameters.
            data: Input data.

        Returns:
            Parameters derived from the input targets.

        """
        orig_width, orig_height, _ = params["shape"]
        return {
            "orig_width": orig_width,
            "orig_height": orig_height,
        }

    @override
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Flip an image vertically.

        Args:
            img: Image to flip.
            params: Additional transform parameters.

        Returns:
            Flipped image.

        """
        return cv2.flip(img, 0)

    @override
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """Flip a segmentation mask vertically.

        Args:
            img: Segmentation mask to flip.
            params: Additional transform parameters.

        Returns:
            Flipped segmentation mask.

        """
        return cv2.flip(img, 0)

    @override
    def apply_to_bboxes(self, bboxes: np.ndarray, **params) -> np.ndarray:
        """Flip bounding boxes vertically.

        Args:
            bboxes: Bounding boxes to flip.
            params: Additional transform parameters.

        Returns:
            Flipped bounding boxes.

        """
        if bboxes.size == 0:
            return bboxes
        flipped = bboxes.copy()
        flipped[:, [1, 3]] = 1 - flipped[:, [3, 1]]
        return flipped

    @override
    def apply_to_keypoints(
        self, keypoints: np.ndarray, orig_height: int, **params
    ) -> np.ndarray:
        """Flip keypoints vertically and swap symmetric pairs.

        Args:
            keypoints: Keypoints to flip.
            orig_height: Original image height.
            params: Additional transform parameters.

        Returns:
            Flipped keypoints.

        """
        if keypoints.size == 0:
            return keypoints

        keypoints = keypoints.copy()

        keypoints[:, 1] = orig_height - keypoints[:, 1]

        total_keypoints = keypoints.shape[0]
        if total_keypoints % self.n_keypoints != 0:
            raise ValueError(
                "Total number of keypoints is not a multiple of n_keypoints defined by keypoint_pairs."
            )
        num_instances = total_keypoints // self.n_keypoints

        for instance in range(num_instances):
            offset = instance * self.n_keypoints
            for i, j in self.keypoint_pairs:
                idx1, idx2 = offset + i, offset + j
                tmp = keypoints[idx1].copy()
                keypoints[idx1] = keypoints[idx2]
                keypoints[idx2] = tmp

        return keypoints


class TransposeSymmetricKeypoints(A.DualTransform):
    def __init__(
        self,
        keypoint_pairs: list[tuple[int, int]],
        p: float = 0.5,
    ):
        """Transpose an image and symmetric keypoints.

        Equivalent to 90 degree rotation followed by horizontal flip.

        Args:
            keypoint_pairs: Pairs of keypoint indices to swap after the
                transpose.
            p: Probability of applying the augmentation.

        """
        super().__init__(p=p)
        self.keypoint_pairs = keypoint_pairs
        self.n_keypoints = len(set(chain.from_iterable(keypoint_pairs)))

    @property
    @override
    def targets(self) -> dict[str, Any]:
        targets = super().targets
        targets["instance_mask"] = self.apply_to_mask
        targets["segmentation"] = self.apply_to_mask
        return targets

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get parameters dependent on the targets.

        Args:
            params: Existing augmentation parameters.
            data: Input data.

        Returns:
            Parameters derived from the input targets.

        """
        orig_width, orig_height, _ = params["shape"]
        return {"orig_width": orig_width, "orig_height": orig_height}

    @override
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Transpose an image.

        Args:
            img: Image to transpose.
            params: Additional transform parameters.

        Returns:
            Transposed image.

        """
        axes = (1, 0, *tuple(range(2, img.ndim)))
        return img.transpose(axes)

    @override
    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """Transpose a segmentation mask.

        Args:
            mask: Segmentation mask to transpose.
            params: Additional transform parameters.

        Returns:
            Transposed segmentation mask.

        """
        axes = (1, 0, *tuple(range(2, mask.ndim)))
        return mask.transpose(axes)

    @override
    def apply_to_bboxes(self, bboxes: np.ndarray, **params) -> np.ndarray:
        """Transpose bounding boxes.

        Args:
            bboxes: Bounding boxes to transpose.
            params: Additional transform parameters.

        Returns:
            Transposed bounding boxes.

        """
        if bboxes.size == 0:
            return bboxes
        t = bboxes.copy()
        t[:, [0, 1, 2, 3]] = t[:, [1, 0, 3, 2]]
        return t

    @override
    def apply_to_keypoints(
        self, keypoints: np.ndarray, **params
    ) -> np.ndarray:
        """Transpose keypoints and swap symmetric pairs.

        Args:
            keypoints: Keypoints to transpose.
            params: Additional transform parameters.

        Returns:
            Transposed keypoints.

        """
        if keypoints.size == 0:
            return keypoints
        keypoints = keypoints.copy()
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        total_keypoints = keypoints.shape[0]
        if total_keypoints % self.n_keypoints != 0:
            raise ValueError(
                "Total number of keypoints is not a multiple of n_keypoints defined by keypoint_pairs."
            )
        num_instances = total_keypoints // self.n_keypoints

        for instance in range(num_instances):
            offset = instance * self.n_keypoints
            for i, j in self.keypoint_pairs:
                idx1, idx2 = offset + i, offset + j
                tmp = keypoints[idx1].copy()
                keypoints[idx1] = keypoints[idx2]
                keypoints[idx2] = tmp
        return keypoints
