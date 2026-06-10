r"""Utilities for adapting LDF annotations to Albumentations.

The augmentation engine receives annotations in Luxonis Data Format (LDF),
converts them to Albumentations-compatible arrays before spatial transforms,
and converts them back afterward. This module keeps those boundary conversions
centralized.

.. list-table:: Layout conversions
   :header-rows: 1

   * - Target
     - LDF layout
     - Albumentations layout
   * - Masks
     - :math:`\left(C, H, W\right)` or :math:`\left(N, H, W\right)`
     - :math:`\left(H, W, C\right)` or :math:`\left(H, W, N\right)`
   * - Bounding boxes
     - :math:`\left[c, x, y, w, h\right]`
     - :math:`\left[x_{\min}, y_{\min}, x_{\max}, y_{\max}, c, i\right]`
   * - Keypoints
     - :math:`\left(N, 3K\right)` normalized rows
     - :math:`\left(NK, 3\right)` pixel-space rows

The appended bbox index :math:`i` is used during postprocessing to keep
bbox-associated labels, such as instance masks and keypoints, aligned with the
bounding boxes that survive augmentation and filtering.
"""

from collections.abc import Iterator
from typing import TypeVar

import numpy as np


def preprocess_mask(seg: np.ndarray) -> np.ndarray:
    r"""Convert an LDF mask to the Albumentations mask layout.

    Luxonis Data Format stores semantic masks as
    :math:`\left(C, H, W\right)` and instance masks as
    :math:`\left(N, H, W\right)`. Albumentations expects the spatial
    dimensions first, so this function moves the channel or instance
    dimension to the end.

    Args:
        seg: Mask array of shape :math:`\left(C, H, W\right)` or
            :math:`\left(N, H, W\right)`.

    Returns:
        Mask array of shape :math:`\left(H, W, C\right)` or
        :math:`\left(H, W, N\right)`.

    Examples:
        >>> seg = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> preprocess_mask(seg).shape
        (2, 2, 2)
        >>> preprocess_mask(seg).tolist()
        [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    """
    return seg.transpose(1, 2, 0)


def preprocess_bboxes(bboxes: np.ndarray, bbox_counter: int) -> np.ndarray:
    r"""Convert LDF bounding boxes to Albumentations format.

    LDF stores bounding boxes as normalized
    :math:`\left[c, x, y, w, h\right]` rows, where :math:`c` is the
    class ID and :math:`x` and :math:`y` are the top-left corner.
    Albumentations expects normalized
    :math:`\left[x_{\min}, y_{\min}, x_{\max}, y_{\max}, c\right]`
    rows. This function also appends a stable per-box index used after
    augmentation to keep bbox-associated labels aligned with boxes that
    survived filtering.

    Args:
        bboxes: Bounding boxes of shape :math:`\left(N, 5\right)` in
            :math:`\left[c, x, y, w, h\right]` format.
        bbox_counter: Offset used to create the appended bbox indices.
            The first output row receives this value, the next receives
            ``bbox_counter + 1``, and so on.

    Returns:
        Bounding boxes of shape :math:`\left(N, 6\right)` in
        :math:`\left[x_{\min}, y_{\min}, x_{\max}, y_{\max}, c, i\right]`
        format, where :math:`i` is the stable bbox index.

    Examples:
        >>> bboxes = np.array([[2, 0.1, 0.2, 0.3, 0.4]])
        >>> out = preprocess_bboxes(bboxes, bbox_counter=5)
        >>> np.round(out[:, :4], 4).tolist()
        [[0.1, 0.2, 0.4, 0.6]]
        >>> out[:, 4:].astype(int).tolist()
        [[2, 5]]

    """
    bboxes = bboxes[:, [1, 2, 3, 4, 0]]

    # Adding 1e-6 to avoid zero width or height.
    bboxes[:, 2] += bboxes[:, 0] + 1e-6
    bboxes[:, 3] += bboxes[:, 1] + 1e-6

    # Used later to filter out instance tasks associated
    # with bboxes that were removed during augmentations.
    indices = np.arange(
        bbox_counter, bboxes.shape[0] + bbox_counter, dtype=bboxes.dtype
    )[:, None]
    return np.concatenate((bboxes, indices), axis=1)


def preprocess_keypoints(
    keypoints: np.ndarray, height: int, width: int
) -> np.ndarray:
    r"""Convert normalized LDF keypoints to absolute coordinates.

    LDF stores keypoints as flattened rows of normalized
    :math:`\left(x, y, v\right)` triplets. Albumentations expects one
    keypoint per row and uses pixel-space coordinates for spatial
    transforms.

    Args:
        keypoints: Keypoints of shape :math:`\left(N, 3K\right)`, where
            :math:`N` is the number of annotations and :math:`K` is the
            number of keypoints per annotation. Each row has format
            :math:`\left[x_1, y_1, v_1, x_2, y_2, v_2,
            \ldots\right]`.
        height: Image height used to scale normalized :math:`y`
            coordinates.
        width: Image width used to scale normalized :math:`x`
            coordinates.

    Returns:
        Keypoints of shape :math:`\left(NK, 3\right)` in pixel-space
        :math:`\left[x, y, v\right]` format.

    Examples:
        >>> keypoints = np.array([[0.5, 0.25, 2, 1.0, 0.0, 1]])
        >>> preprocess_keypoints(keypoints, height=8, width=4).tolist()
        [[2.0, 2.0, 2.0], [4.0, 0.0, 1.0]]
        >>> preprocess_keypoints(np.empty((0, 3)), 8, 4).shape
        (0, 3)

    """
    keypoints = np.reshape(keypoints, (-1, 3))
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return keypoints


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    r"""Convert an augmented mask back to the LDF layout.

    Args:
        mask: Augmented mask. Two-dimensional inputs are interpreted as a
            single mask of shape :math:`\left(H, W\right)`. Three-dimensional
            inputs are expected to have shape
            :math:`\left(H, W, C\right)` or :math:`\left(H, W, N\right)`.

    Returns:
        Mask array in LDF channels-first layout. Two-dimensional inputs
        become :math:`\left(1, H, W\right)`, and three-dimensional inputs
        become :math:`\left(C, H, W\right)` or
        :math:`\left(N, H, W\right)`.

    Examples:
        >>> postprocess_mask(np.array([[1, 0], [0, 1]])).shape
        (1, 2, 2)
        >>> mask = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])
        >>> postprocess_mask(mask).tolist()
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    """
    if mask.ndim == 2:
        return mask[None, ...]

    return mask.transpose(2, 0, 1)


def postprocess_bboxes(
    bboxes: np.ndarray, area_threshold: float = 0.0004
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert augmented bounding boxes back to LDF format.

    Bounding boxes smaller than ``area_threshold`` are discarded. The
    appended bbox indices are returned alongside the boxes so labels tied to
    boxes, such as instance masks, keypoints, arrays, and metadata, can be
    filtered in the same way.

    Args:
        bboxes: Augmented bounding boxes of shape
            :math:`\left(N, 6\right)` in
            :math:`\left[x_{\min}, y_{\min}, x_{\max}, y_{\max}, c,
            i\right]` format, where :math:`c` is the class ID and
            :math:`i` is the stable bbox index.
        area_threshold: Minimum normalized box area
            :math:`w \cdot h` required for a box to remain valid.

    Returns:
        A tuple containing:

            - Bounding boxes of shape :math:`\left(M, 5\right)` in
              :math:`\left[c, x, y, w, h\right]` format, where
              :math:`M \leq N` and :math:`c` is the class ID.
            - Integer indices of shape :math:`\left(M\right)` identifying
              which original bbox-associated labels survived filtering.

    Examples:
        >>> bboxes = np.array([[0.1, 0.2, 0.4, 0.6, 2, 5]], dtype=float)
        >>> out_bboxes, ordering = postprocess_bboxes(bboxes)
        >>> np.round(out_bboxes, 2).tolist()
        [[2.0, 0.1, 0.2, 0.3, 0.4]]
        >>> ordering.tolist()
        [5]
        >>> tiny = np.array([[0.0, 0.0, 0.01, 0.01, 1, 3]], dtype=float)
        >>> postprocess_bboxes(tiny, area_threshold=0.1)[0].shape
        (0, 5)
        >>> postprocess_bboxes(np.empty((0, 6)), area_threshold=0.1)[0].shape
        (0, 5)

    """
    if bboxes.size == 0:
        return np.zeros((0, 5)), np.zeros((0,), dtype=np.uint8)
    ordering = bboxes[:, -1]
    raw_bboxes = bboxes[:, :-1]
    raw_bboxes[:, 2] -= raw_bboxes[:, 0]
    raw_bboxes[:, 3] -= raw_bboxes[:, 1]
    widths = raw_bboxes[:, 2]
    heights = raw_bboxes[:, 3]
    areas = widths * heights

    valid_mask = areas >= area_threshold
    raw_bboxes = raw_bboxes[valid_mask]
    refined_ordering = ordering[valid_mask]

    out_bboxes = raw_bboxes[:, [4, 0, 1, 2, 3]]

    return out_bboxes, refined_ordering.astype(int)


def postprocess_keypoints(
    keypoints: np.ndarray,
    bboxes_ordering: np.ndarray,
    image_height: int,
    image_width: int,
    n_keypoints: int,
) -> np.ndarray:
    r"""Convert augmented keypoints back to normalized LDF rows.

    The input contains pixel-space keypoints after Albumentations has applied
    spatial transforms. This function restores the original annotation
    grouping, keeps only annotations associated with surviving bounding boxes,
    marks out-of-image keypoints as invisible, clips coordinates to the image
    extent, and normalizes coordinates back to :math:`\left[0, 1\right]`.

    Args:
        keypoints: Augmented keypoints of shape
            :math:`\left(NK, D\right)`, where the first
            :math:`3K` values per annotation are
            :math:`\left(x, y, v\right)` triplets and any extra columns are
            discarded.
        bboxes_ordering: Indices of bbox-associated annotations that survived
            bbox postprocessing.
        image_height: Height of the augmented image used to normalize
            :math:`y` coordinates.
        image_width: Width of the augmented image used to normalize
            :math:`x` coordinates.
        n_keypoints: Number of keypoints :math:`K` in each annotation.

    Returns:
        Keypoints of shape :math:`\left(M, 3K\right)`, where
        :math:`M` is the number of retained annotations. Each row is in
        normalized LDF format
        :math:`\left[x_1, y_1, v_1, x_2, y_2, v_2, \ldots\right]`.

    Examples:
        >>> keypoints = np.array([[5, 5, 2], [12, -1, 2]], dtype=float)
        >>> out = postprocess_keypoints(keypoints, np.array([0]), 10, 10, 2)
        >>> np.round(out, 2).tolist()
        [[0.5, 0.5, 2.0, 1.0, 0.0, 0.0]]
        >>> keypoints = np.array([[1, 1, 2], [9, 9, 1]], dtype=float)
        >>> postprocess_keypoints(keypoints, np.array([1, 0]), 10, 10, 1).tolist()
        [[0.9, 0.9, 1.0], [0.1, 0.1, 2.0]]

    """
    keypoints = keypoints[:, : (n_keypoints * 3)]
    keypoints = keypoints.reshape(-1, n_keypoints, 3)

    keypoints = keypoints[bboxes_ordering]

    x = keypoints[..., 0]
    y = keypoints[..., 1]
    v = keypoints[..., 2]

    in_bounds = (x >= 0) & (x < image_width) & (y >= 0) & (y < image_height)

    v[~in_bounds] = 0

    x = np.clip(x, 0, image_width)
    y = np.clip(y, 0, image_height)

    x /= image_width
    y /= image_height

    keypoints[..., 0] = x
    keypoints[..., 1] = y
    keypoints[..., 2] = v

    return keypoints.reshape(-1, n_keypoints * 3)


T = TypeVar("T")


def yield_batches(
    data_batch: list[dict[str, T]], batch_size: int
) -> Iterator[dict[str, list[T]]]:
    r"""Yield fixed-size chunks of sample dictionaries.

    Args:
        data_batch: Dictionaries containing data for each sample. All
            dictionaries are expected to expose the same keys.
        batch_size: Maximum number of samples in each yielded batch.

    Returns:
        Iterator over dictionaries grouped by key. For an input chunk of
        :math:`B` samples, each yielded value maps every key to a list of
        :math:`B` values.

    Examples:
        >>> samples = [
        ...     {"image": "a", "label": 1},
        ...     {"image": "b", "label": 2},
        ...     {"image": "c", "label": 3},
        ... ]
        >>> list(yield_batches(samples, 2))
        [{'image': ['a', 'b'], 'label': [1, 2]}, {'image': ['c'], 'label': [3]}]
        >>> next(yield_batches(samples, 10))["label"]
        [1, 2, 3]

    """
    for i in range(0, len(data_batch), batch_size):
        yield {
            target: [data[target] for data in data_batch[i : i + batch_size]]
            for target in data_batch[0]
        }
