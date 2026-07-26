from itertools import chain
from typing import Any

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override


class HorizontalSymmetricKeypointsFlip(A.DualTransform):
    """Mirror left-to-right, re-indexing symmetric keypoints so labels stay correct.

    A plain horizontal flip mirrors the pixels (and boxes and masks) but leaves
    the keypoint *order* untouched, which mislabels left/right-symmetric objects:
    after mirroring, the object's left side is on the right, yet a fixed "left
    shoulder" index still points at the now-right-side shoulder. This transform
    additionally swaps each left/right keypoint pair, so every index keeps naming
    the same part — the left-shoulder point travels with the mirrored left side
    and is still labeled the left shoulder. Use it instead of Albumentations'
    ``HorizontalFlip`` whenever a task has symmetric keypoints.

    .. image:: TODO-HOST/aug_horizontal_flip.png
       :alt: A pose with left/right joints colored, before and after a horizontal symmetric flip.

    ``keypoint_pairs`` describes one object's symmetry as ``(left_index,
    right_index)`` pairs. A keypoint that sits on the mirror axis (e.g. the nose)
    has no partner and pairs with itself, ``(i, i)``. Every keypoint of the object
    must appear in exactly one pair, so ``n_keypoints`` — the number of distinct
    indices — equals the object's keypoint count. Keypoints for several objects
    are passed as one flat array of length ``num_objects * n_keypoints``, and each
    object's block is re-indexed independently. Bounding boxes and
    segmentation/instance masks are mirrored as well.

    Attributes:
        keypoint_pairs: ``(left_index, right_index)`` symmetry pairs for a single
            object; on-axis keypoints pair with themselves as ``(i, i)``.
        n_keypoints: Distinct keypoints per object, derived from
            ``keypoint_pairs``.

    Examples:
        A center point (self-paired) plus one left/right pair: the coordinates
        mirror and the pair swaps, so each index keeps its body part.

        >>> import numpy as np
        >>> import albumentations as A
        >>> transform = A.Compose(
        ...     [
        ...         HorizontalSymmetricKeypointsFlip(
        ...             keypoint_pairs=[(0, 0), (1, 2)], p=1.0
        ...         )
        ...     ],
        ...     keypoint_params=A.KeypointParams(
        ...         format="xy", remove_invisible=False
        ...     ),
        ... )
        >>> out = transform(
        ...     image=np.zeros((10, 10, 3), np.uint8),
        ...     keypoints=[(2.0, 5.0), (1.0, 3.0), (9.0, 4.0)],
        ... )["keypoints"]
        >>> [(round(float(k[0]), 1), round(float(k[1]), 1)) for k in out]
        [(8.0, 5.0), (1.0, 4.0), (9.0, 3.0)]

    """

    def __init__(self, keypoint_pairs: list[tuple[int, int]], p: float = 0.5):
        """Mirror an image and its symmetric keypoints left-to-right.

        Bounding boxes and segmentation/instance masks are mirrored too.

        Args:
            keypoint_pairs: ``(left_index, right_index)`` symmetry pairs for one
                object. On-axis keypoints with no mirror partner pair with
                themselves, ``(i, i)``. Every keypoint must be covered exactly
                once (see the class docstring).
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

        Raises:
            ValueError: If the total number of keypoints is not a multiple
                of ``n_keypoints``.

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
    """Mirror top-to-bottom, re-indexing symmetric keypoints so labels stay correct.

    The vertical counterpart of
    `HorizontalSymmetricKeypointsFlip`: the pixels, boxes, and masks are
    mirrored top-to-bottom, and each pair in ``keypoint_pairs`` is swapped so
    every index keeps naming the same part after the mirror (a plain vertical
    flip would leave the order untouched and mislabel symmetric keypoints).

    .. image:: TODO-HOST/aug_vertical_flip.png
       :alt: A pose with left/right joints colored, before and after a vertical symmetric flip.

    Give the pairs of keypoints that map onto each other under a *vertical*
    mirror; a keypoint on the axis pairs with itself as ``(i, i)``, and every
    keypoint of the object must appear exactly once (so ``n_keypoints`` is the
    object's keypoint count). Several objects are passed as one flat array of
    length ``num_objects * n_keypoints`` and re-indexed per block. See
    `HorizontalSymmetricKeypointsFlip` for the full explanation.

    Attributes:
        keypoint_pairs: Index pairs that swap under the vertical mirror; on-axis
            keypoints pair with themselves as ``(i, i)``.
        n_keypoints: Distinct keypoints per object, derived from
            ``keypoint_pairs``.

    Examples:
        The coordinates mirror top-to-bottom and the pair swaps:

        >>> import numpy as np
        >>> import albumentations as A
        >>> transform = A.Compose(
        ...     [
        ...         VerticalSymmetricKeypointsFlip(
        ...             keypoint_pairs=[(0, 0), (1, 2)], p=1.0
        ...         )
        ...     ],
        ...     keypoint_params=A.KeypointParams(
        ...         format="xy", remove_invisible=False
        ...     ),
        ... )
        >>> out = transform(
        ...     image=np.zeros((10, 10, 3), np.uint8),
        ...     keypoints=[(2.0, 5.0), (1.0, 3.0), (9.0, 4.0)],
        ... )["keypoints"]
        >>> [(round(float(k[0]), 1), round(float(k[1]), 1)) for k in out]
        [(2.0, 5.0), (9.0, 6.0), (1.0, 7.0)]

    """

    def __init__(self, keypoint_pairs: list[tuple[int, int]], p: float = 0.5):
        """Mirror an image and its symmetric keypoints top-to-bottom.

        Bounding boxes and segmentation/instance masks are mirrored too.

        Args:
            keypoint_pairs: Index pairs that swap under the vertical mirror.
                On-axis keypoints pair with themselves, ``(i, i)``; every
                keypoint must be covered exactly once (see the class docstring).
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

        Raises:
            ValueError: If the total number of keypoints is not a multiple
                of ``n_keypoints``.

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
    """Reflect across the main diagonal, re-indexing symmetric keypoints.

    Transposing swaps the x and y axes (equivalent to a 90 degree rotation
    followed by a horizontal flip), reflecting the image, boxes, and masks across
    the top-left/bottom-right diagonal. As with the flips, the geometry alone
    would leave keypoint order untouched and mislabel symmetric objects, so each
    pair in ``keypoint_pairs`` is swapped to keep every index on the same part.

    .. image:: TODO-HOST/aug_transpose.png
       :alt: A pose with left/right joints colored, before and after a symmetric transpose.

    Give the pairs of keypoints that map onto each other under the diagonal
    reflection; a keypoint on the axis pairs with itself as ``(i, i)``, and every
    keypoint of the object must appear exactly once (so ``n_keypoints`` is the
    object's keypoint count). Several objects are passed as one flat array of
    length ``num_objects * n_keypoints`` and re-indexed per block. See
    `HorizontalSymmetricKeypointsFlip` for the full explanation.

    Attributes:
        keypoint_pairs: Index pairs that swap under the diagonal reflection;
            on-axis keypoints pair with themselves as ``(i, i)``.
        n_keypoints: Distinct keypoints per object, derived from
            ``keypoint_pairs``.

    Examples:
        Each keypoint's x and y are swapped and the pair is swapped:

        >>> import numpy as np
        >>> import albumentations as A
        >>> transform = A.Compose(
        ...     [
        ...         TransposeSymmetricKeypoints(
        ...             keypoint_pairs=[(0, 0), (1, 2)], p=1.0
        ...         )
        ...     ],
        ...     keypoint_params=A.KeypointParams(
        ...         format="xy", remove_invisible=False
        ...     ),
        ... )
        >>> out = transform(
        ...     image=np.zeros((10, 10, 3), np.uint8),
        ...     keypoints=[(2.0, 5.0), (1.0, 3.0), (9.0, 4.0)],
        ... )["keypoints"]
        >>> [(round(float(k[0]), 1), round(float(k[1]), 1)) for k in out]
        [(5.0, 2.0), (4.0, 9.0), (3.0, 1.0)]

    """

    def __init__(
        self,
        keypoint_pairs: list[tuple[int, int]],
        p: float = 0.5,
    ):
        """Reflect an image and its symmetric keypoints across the main diagonal.

        Transposing is equivalent to a 90 degree rotation followed by a
        horizontal flip. Bounding boxes and segmentation/instance masks are
        transposed too.

        Args:
            keypoint_pairs: Index pairs that swap under the diagonal reflection.
                On-axis keypoints pair with themselves, ``(i, i)``; every
                keypoint must be covered exactly once (see the class docstring).
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

        Raises:
            ValueError: If the total number of keypoints is not a multiple
                of ``n_keypoints``.

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
