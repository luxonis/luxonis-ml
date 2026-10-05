"""Flips and a transpose that also swap the names of symmetric keypoints.

Mirroring an image moves the left wrist of a person to where the right wrist
was. A plain Albumentations flip moves the coordinates but keeps the names,
so the keypoint called ``left_wrist`` ends up on the right side of the body.
These transforms move the coordinates and also swap the keypoints of each
symmetric pair.

Each keypoint task can have its own skeleton. `AlbumentationsEngine` tells
the transforms, for each keypoint task, how many keypoints an instance has
and which pairs the task stores in its `KeypointMetadata`. Pairs given in the
configuration replace the stored pairs of each task that they fit.
"""

from functools import partial
from itertools import chain
from typing import Any

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override

#: The keypoint count of an instance and the pairs to swap, for one target.
Layout = tuple[int, list[tuple[int, int]]]


class SymmetricKeypointsTransform(A.DualTransform):
    """Base of the transforms that mirror keypoints and swap symmetric ones.

    A subclass moves the pixels and the coordinates; this class swaps the
    keypoints of each symmetric pair, instance by instance. Albumentations
    hands over the keypoints of one target as a flat ``(N * K, D)`` array, so
    the transform has to know ``K``, the number of keypoints of an instance.
    `set_layouts` gives it ``K`` and the stored pairs for each target. A
    target without a layout takes ``K`` as one more than the largest index
    in ``keypoint_pairs``.

    Attributes:
        keypoint_pairs: Index pairs from the configuration. They replace the
            stored pairs of each target whose instances have more keypoints
            than their largest index. A target that they do not fit swaps
            the pairs that its task stores.
        n_keypoints: The keypoint count assumed for a target without a
            layout.
        layouts: For each keypoint target, its keypoint count and the pairs
            its task stores.

    """

    def __init__(
        self,
        keypoint_pairs: list[tuple[int, int]] | None = None,
        p: float = 0.5,
    ):
        """Mirror an image and swap symmetric keypoints.

        Bounding boxes and segmentation masks move with the image.

        Args:
            keypoint_pairs: Pairs of keypoint indices to swap. They
                replace the stored flip pairs of each task that they fit,
                so identity pairs keep the keypoints in place. Without
                them, or for a task that they do not fit, the transform
                swaps the pairs that the task stores.
            p: Probability of applying the augmentation.

        """
        super().__init__(p=p)
        self.keypoint_pairs = [(i, j) for i, j in keypoint_pairs or []]
        self.n_keypoints = (
            max(chain.from_iterable(self.keypoint_pairs), default=-1) + 1
        )
        self.layouts: dict[str, Layout] = {}

    def set_layouts(self, layouts: dict[str, Layout]) -> None:
        """Set the keypoint count and the stored pairs of each target.

        Args:
            layouts: The layout of each keypoint target, by target name.

        """
        self.layouts = dict(layouts)

    @property
    @override
    def targets(self) -> dict[str, Any]:
        targets = super().targets
        targets["instance_mask"] = self.apply_to_mask
        targets["segmentation"] = self.apply_to_mask
        return targets

    @override
    def add_targets(self, additional_targets: dict[str, str]) -> None:
        """Register extra targets, and bind each keypoint target by name.

        Albumentations calls one function for every target of a kind. Each
        keypoint target here gets a function that knows the target's name,
        so that it can swap with the layout of its own task.

        Args:
            additional_targets: The kind of each extra target, by name.

        """
        super().add_targets(additional_targets)
        for key, kind in additional_targets.items():
            if kind == "keypoints" and key in self._key2func:
                self._key2func[key] = partial(self._mirror_keypoints, key)

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
        """Mirror the keypoints of the default target, and swap the pairs.

        Args:
            keypoints: Keypoints to mirror.
            params: Additional transform parameters.

        Returns:
            Mirrored keypoints.

        """
        return self._mirror_keypoints("keypoints", keypoints, **params)

    def _move(self, keypoints: np.ndarray, **params) -> np.ndarray:
        """Move the keypoint coordinates the way the image moves."""
        raise NotImplementedError

    def _mirror_keypoints(
        self, target: str, keypoints: np.ndarray, **params
    ) -> np.ndarray:
        """Move the keypoints of one target and swap their pairs.

        Raises:
            ValueError: If the keypoints do not split into instances of the
                keypoint count.

        """
        if keypoints.size == 0:
            return keypoints
        keypoints = self._move(keypoints.copy(), **params)
        size, stored = self.layouts.get(target, (self.n_keypoints, []))
        fits = bool(self.keypoint_pairs) and self.n_keypoints <= size
        pairs = self.keypoint_pairs if fits else stored
        if not pairs:
            return keypoints
        if len(keypoints) % size:
            raise ValueError(
                f"{len(keypoints)} keypoints of target '{target}' do not "
                f"split into instances of {size} keypoints."
            )
        order = np.arange(len(keypoints)).reshape(-1, size)
        swapped = order.copy()
        for i, j in pairs:
            swapped[:, [i, j]] = order[:, [j, i]]
        return keypoints[swapped.reshape(-1)]


class HorizontalSymmetricKeypointsFlip(SymmetricKeypointsTransform):
    """Flip images and symmetric keypoints horizontally.

    Examples:
        >>> import numpy as np
        >>> flip = HorizontalSymmetricKeypointsFlip([(1, 2)], p=1.0)
        >>> flip.n_keypoints  # the unpaired keypoint 0 still counts
        3
        >>> keypoints = np.array([[5.0, 1, 2], [2, 1, 2], [8, 1, 2]])
        >>> flip.apply_to_keypoints(keypoints, orig_width=10)[:, 0].tolist()
        [5.0, 2.0, 8.0]

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
    """Flip images and symmetric keypoints vertically."""

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
    """Transpose images and symmetric keypoints.

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
