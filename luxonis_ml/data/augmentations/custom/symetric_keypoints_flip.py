from itertools import chain
from typing import Any

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override


class HorizontalSymetricKeypointsFlip(A.DualTransform):
    def __init__(self, keypoint_pairs: list[tuple[int, int]], p: float = 0.5):
        """Augmentation to horizontally flip an image along with bboxes,
        segmentation masks and symmetric keypoints.

        @param keypoint_pairs: List of tuples with indices to swap after
            flipping.
        @param p: Probability of applying the augmentation.
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

        @param params: Dictionary containing parameters.
        @type params: dict[str, Any]
        @param data: Dictionary containing data.
        @type data: dict[str, Any]
        """
        orig_height, orig_width, _ = params["shape"]
        return {
            "orig_width": orig_width,
            "orig_height": orig_height,
        }

    @override
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Flips the image horizontally.

        @param img: Image to be flipped.
        @type img: np.ndarray
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
        """
        return cv2.flip(img, 1)

    @override
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """Flips segmentation masks horizontally.

        @param img: Segmentation mask to be flipped.
        @param params: Parameters for the transformation.
        """
        return cv2.flip(img, 1)

    @override
    def apply_to_bboxes(self, bboxes: np.ndarray, **params) -> np.ndarray:
        """Flips bounding boxes horizontally.

        @param bboxes: Bounding boxes to be flipped.
        @type bboxes: np.ndarray
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
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
        """Flips keypoints horizontally and then swaps symmetric ones.

        @param keypoints: Keypoints to be flipped.
        @type keypoints: np.ndarray
        @param orig_width: Width of the original image.
        @type orig_width: int
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
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


class VerticalSymetricKeypointsFlip(A.DualTransform):
    def __init__(self, keypoint_pairs: list[tuple[int, int]], p: float = 0.5):
        """Augmentation to vertically flip an image along with bboxes,
        segmentation masks and symmetric keypoints.

        @param keypoint_pairs: List of tuples with indices to swap after
            vertical flip.
        @type keypoint_pairs: list[tuple[int, int]]
        @param p: Probability of applying the augmentation.
        @type p: float
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

        @param params: Dictionary containing parameters.
        @type params: dict[str, Any]
        @param data: Dictionary containing data.
        @type data: dict[str, Any]
        """
        orig_width, orig_height, _ = params["shape"]
        return {
            "orig_width": orig_width,
            "orig_height": orig_height,
        }

    @override
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Flips the image vertically.

        @param img: Image to be flipped.
        @type img: np.ndarray
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
        """
        return cv2.flip(img, 0)

    @override
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """Flips segmentation masks vertically.

        @param img: Segmentation mask to be flipped.
        @type img: np.ndarray
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
        """
        return cv2.flip(img, 0)

    @override
    def apply_to_bboxes(self, bboxes: np.ndarray, **params) -> np.ndarray:
        """Flips bounding boxes vertically.

        @param bboxes: Bounding boxes to be flipped.
        @type bboxes: np.ndarray
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
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
        """Flips keypoints vertically and then swaps symmetric ones.

        @param keypoints: Keypoints to be flipped.
        @type keypoints: np.ndarray
        @param orig_height: Original height in the image.
        @type orig_height: int
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
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
        """Augmentation to transpose an image along with bboxes,
        segmentation masks and symmetric keypoints.

        Equivalent to 90 degree rotation followed by horizontal flip.

        @param keypoint_pairs: List of tuples with indices to swap after
            transposing.
        @param p: Probability of applying the augmentation.
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

        @param params: Dictionary containing parameters.
        @type params: dict[str, Any]
        @param data: Dictionary containing data.
        @type data: dict[str, Any]
        """
        orig_width, orig_height, _ = params["shape"]
        return {"orig_width": orig_width, "orig_height": orig_height}

    @override
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Flips the image horizontally.

        @param img: Image to be flipped.
        @type img: np.ndarray
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
        """
        axes = (1, 0, *tuple(range(2, img.ndim)))
        return img.transpose(axes)

    @override
    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """Flips segmentation masks horizontally.

        @param img: Segmentation mask to be flipped.
        @param params: Parameters for the transformation.
        """
        axes = (1, 0, *tuple(range(2, mask.ndim)))
        return mask.transpose(axes)

    @override
    def apply_to_bboxes(self, bboxes: np.ndarray, **params) -> np.ndarray:
        """Flips bounding boxes horizontally.

        @param bboxes: Bounding boxes to be flipped.
        @type bboxes: np.ndarray
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
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
        """Flips keypoints horizontally and then swaps symmetric ones.

        @param keypoints: Keypoints to be flipped.
        @type keypoints: np.ndarray
        @param orig_width: Original width of the image.
        @type orig_width: int
        @param params: Parameters for the transformation.
        @type params: Dict[str, Any]
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
