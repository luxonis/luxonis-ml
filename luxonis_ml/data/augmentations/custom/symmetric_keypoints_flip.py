"""Deprecated flips and a transpose that swap configured keypoint pairs.

`AlbumentationsEngine` now gives every augmented keypoint the name that its
meaning asks for, after any transform: a plain ``HorizontalFlip``,
``VerticalFlip`` or ``Transpose`` swaps the mirror partners of each task by
itself. See `luxonis_ml.data.augmentations.keypoint_naming`. These
transforms remain for old configurations:

    - In the engine, they flip like the plain transforms. Their
      ``keypoint_pairs`` give the mirror partners of a task that stores
      none.
    - Outside the engine, they flip and swap their ``keypoint_pairs``, as
      before.
"""

from itertools import chain
from typing import Any

import albumentations as A
import cv2
import numpy as np
from loguru import logger
from typing_extensions import override

from luxonis_ml.utils.logging import log_once


class SymmetricKeypointsTransform(A.DualTransform):
    """Base of the deprecated flips that swap configured keypoint pairs.

    A subclass moves the pixels and the coordinates. This class swaps the
    keypoints of each pair in ``keypoint_pairs``, instance by instance,
    unless `AlbumentationsEngine` runs it: the engine names the keypoints
    itself after the whole pipeline. Albumentations hands over the
    keypoints as a flat ``(N * K, D)`` array, and ``K`` is one more than
    the largest index in ``keypoint_pairs``.

    Attributes:
        keypoint_pairs: Index pairs of mirror partners.
        n_keypoints: The keypoint count of an instance.
        swaps_pairs: Whether the transform swaps ``keypoint_pairs``
            itself. `AlbumentationsEngine` turns it off.

    """

    def __init__(
        self,
        keypoint_pairs: list[tuple[int, int]] | None = None,
        p: float = 0.5,
    ):
        """Mirror an image and swap the configured keypoint pairs.

        Bounding boxes and segmentation masks move with the image.

        Args:
            keypoint_pairs: Index pairs of mirror partners. In the engine,
                they give the partners of a task that stores none, for
                every mirror of the pipeline. Identity pairs such as
                ``[(0, 0)]`` swap nothing.
            p: Probability of applying the augmentation.

        """
        super().__init__(p=p)
        log_once(
            logger.warning,
            f"`{type(self).__name__}` is deprecated. Use the plain "
            "Albumentations flip: the augmentation gives every keypoint the "
            "name its meaning asks for after any transform.",
        )
        self.keypoint_pairs = [(i, j) for i, j in keypoint_pairs or []]
        self.n_keypoints = (
            max(chain.from_iterable(self.keypoint_pairs), default=-1) + 1
        )
        self.swaps_pairs = True

    @property
    @override
    def targets(self) -> dict[str, Any]:
        """Return the target functions, with masks moved like the image."""
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
            The height and the width of the image.

        """
        orig_height, orig_width = params["shape"][:2]
        return {"orig_width": orig_width, "orig_height": orig_height}

    @override
    def apply_to_keypoints(
        self, keypoints: np.ndarray, **params
    ) -> np.ndarray:
        """Mirror the keypoints, and swap the configured pairs.

        Args:
            keypoints: Keypoints to mirror.
            params: Additional transform parameters.

        Returns:
            Mirrored keypoints.

        Raises:
            ValueError: If the keypoints do not split into instances of
                `n_keypoints`.

        """
        if keypoints.size == 0:
            return keypoints
        keypoints = self._move(keypoints.copy(), **params)
        pairs = [(i, j) for i, j in self.keypoint_pairs if i != j]
        if not (self.swaps_pairs and pairs):
            return keypoints
        size = self.n_keypoints
        if len(keypoints) % size:
            raise ValueError(
                f"{len(keypoints)} keypoints do not split into instances "
                f"of {size} keypoints."
            )
        order = np.arange(len(keypoints)).reshape(-1, size)
        swapped = order.copy()
        for i, j in pairs:
            swapped[:, [i, j]] = order[:, [j, i]]
        return keypoints[swapped.reshape(-1)]

    def _move(self, keypoints: np.ndarray, **params) -> np.ndarray:
        """Move the keypoint coordinates the way the image moves."""
        raise NotImplementedError


class HorizontalSymmetricKeypointsFlip(SymmetricKeypointsTransform):
    """Flip images horizontally, and swap the configured keypoint pairs.

    Deprecated: use ``HorizontalFlip``, which renames the keypoints of every
    task from their meaning.
    """

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
    def _move(
        self, keypoints: np.ndarray, orig_width: int, **params
    ) -> np.ndarray:
        """Mirror the x coordinates across the image width."""
        keypoints[:, 0] = orig_width - keypoints[:, 0]
        return keypoints


class VerticalSymmetricKeypointsFlip(SymmetricKeypointsTransform):
    """Flip images vertically, and swap the configured keypoint pairs.

    Deprecated: use ``VerticalFlip``, which renames the keypoints of every
    task from their meaning.
    """

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
    def _move(
        self, keypoints: np.ndarray, orig_height: int, **params
    ) -> np.ndarray:
        """Mirror the y coordinates across the image height."""
        keypoints[:, 1] = orig_height - keypoints[:, 1]
        return keypoints


class TransposeSymmetricKeypoints(SymmetricKeypointsTransform):
    """Transpose images, and swap the configured keypoint pairs.

    Deprecated: use ``Transpose``, which renames the keypoints of every
    task from their meaning.

    A transpose is a rotation by 90 degrees followed by a horizontal flip.
    """

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
    def _move(self, keypoints: np.ndarray, **params) -> np.ndarray:
        """Swap the x and the y coordinates."""
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        return keypoints
