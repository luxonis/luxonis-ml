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
     - :math:`\left[c, x, y, w, h, a\right]`
     - :math:`\left[x_{\min}, y_{\min}, x_{\max}, y_{\max}, a, c, i\right]`
   * - Keypoints
     - :math:`\left(N, 3K\right)` normalized rows
     - :math:`\left(NK, 3\right)` pixel-space rows

The appended bbox index :math:`i` is used during postprocessing to keep
bbox-associated labels, such as instance masks and keypoints, aligned with the
bounding boxes that survive augmentation and filtering.

For oriented boxes, a private keypoint target carries the four OBB corners
through Albumentations. The axis-aligned bbox row is still used for visibility
filtering and label ordering.
"""

import math
from collections.abc import Iterator
from typing import TypeVar

import numpy as np

from luxonis_ml.data.datasets.annotation import normalize_angle_degrees


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
    :math:`\left[c, x, y, w, h, a\right]` rows, where :math:`c` is the
    class ID, :math:`x` and :math:`y` are the top-left corner, and
    :math:`a` is the angle in degrees counter-clockwise. Albumentations
    expects normalized axis-aligned bbox rows. The angle is preserved as an
    extra label column and a private keypoint target carries the OBB corners
    through spatial transforms. This function also appends a stable per-box
    index used after augmentation to keep bbox-associated labels aligned with
    boxes that survived filtering.

    Args:
        bboxes: Bounding boxes of shape :math:`\left(N, 6\right)` in
            :math:`\left[c, x, y, w, h, a\right]` format.
        bbox_counter: Offset used to create the appended bbox indices.
            The first output row receives this value, the next receives
            ``bbox_counter + 1``, and so on.

    Returns:
        Bounding boxes of shape :math:`\left(N, 7\right)` in
        :math:`\left[x_{\min}, y_{\min}, x_{\max}, y_{\max}, a, c, i\right]`
        format, where :math:`i` is the stable bbox index.

    Examples:
        >>> bboxes = np.array([[2, 0.1, 0.2, 0.3, 0.4, 15]])
        >>> out = preprocess_bboxes(bboxes, bbox_counter=5)
        >>> np.round(out[:, :4], 4).tolist()
        [[0.1, 0.2, 0.4, 0.6]]
        >>> out[:, 4:].astype(int).tolist()
        [[15, 2, 5]]

    """
    if bboxes.size == 0:
        return np.zeros((0, 7), dtype=bboxes.dtype)

    bboxes = bboxes[:, [1, 2, 3, 4, 5, 0]]

    # Adding 1e-6 to avoid zero width or height.
    bboxes[:, 2] += bboxes[:, 0] + 1e-6
    bboxes[:, 3] += bboxes[:, 1] + 1e-6

    # Used later to filter out instance tasks associated
    # with bboxes that were removed during augmentations.
    indices = np.arange(
        bbox_counter, bboxes.shape[0] + bbox_counter, dtype=bboxes.dtype
    )[:, None]
    return np.concatenate((bboxes, indices), axis=1)


def preprocess_bbox_corners(
    bboxes: np.ndarray, height: int, width: int
) -> np.ndarray:
    r"""Convert LDF OBB rows to private Albumentations keypoints.

    Args:
        bboxes: Bounding boxes of shape :math:`\left(N, 6\right)` in
            :math:`\left[c, x, y, w, h, a\right]` format.
        height: Image height used to scale normalized coordinates.
        width: Image width used to scale normalized coordinates.

    Returns:
        OBB corners of shape :math:`\left(4N, 3\right)` in pixel-space
        ``[x, y, marker]`` format.

    """
    if bboxes.size == 0:
        return np.zeros((0, 3), dtype=bboxes.dtype)

    corners = np.empty((bboxes.shape[0] * 4, 3), dtype=float)
    for i, bbox in enumerate(bboxes):
        _class_id, x, y, box_width, box_height, angle = bbox
        cx = (x + box_width / 2) * width
        cy = (y + box_height / 2) * height
        pixel_width = box_width * width
        pixel_height = box_height * height
        theta = math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        offsets = np.array(
            [
                [-pixel_width / 2, -pixel_height / 2],
                [pixel_width / 2, -pixel_height / 2],
                [pixel_width / 2, pixel_height / 2],
                [-pixel_width / 2, pixel_height / 2],
            ]
        )
        start = i * 4
        end = start + 4
        corners[start:end, 0] = (
            cx + offsets[:, 0] * cos_t + offsets[:, 1] * sin_t
        )
        corners[start:end, 1] = (
            cy - offsets[:, 0] * sin_t + offsets[:, 1] * cos_t
        )
        corners[start:end, 2] = -1

    return corners


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
    bboxes: np.ndarray,
    area_threshold: float = 0.0004,
    bbox_corners: np.ndarray | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert augmented bounding boxes back to LDF format.

    Bounding boxes smaller than ``area_threshold`` are discarded. The
    appended bbox indices are returned alongside the boxes so labels tied to
    boxes, such as instance masks, keypoints, arrays, and metadata, can be
    filtered in the same way.

    Args:
        bboxes: Augmented bounding boxes of shape
            :math:`\left(N, 7\right)` in
            :math:`\left[x_{\min}, y_{\min}, x_{\max}, y_{\max}, a, c,
            i\right]` format, where :math:`a` is the LDF OBB angle,
            :math:`c` is the class ID and :math:`i` is the stable bbox
            index.
        area_threshold: Minimum normalized box area
            :math:`w \cdot h` required for a box to remain valid.
        bbox_corners: Optional transformed private OBB corner keypoints.
        image_height: Height of the augmented image. Required when
            ``bbox_corners`` is provided.
        image_width: Width of the augmented image. Required when
            ``bbox_corners`` is provided.

    Returns:
        A tuple containing:

            - Bounding boxes of shape :math:`\left(M, 6\right)` in
              :math:`\left[c, x, y, w, h, a\right]` format, where
              :math:`M \leq N` and :math:`c` is the class ID.
            - Integer indices of shape :math:`\left(M\right)` identifying
              which original bbox-associated labels survived filtering.

    Examples:
        >>> bboxes = np.array([[0.1, 0.2, 0.4, 0.6, 15, 2, 5]], dtype=float)
        >>> out_bboxes, ordering = postprocess_bboxes(bboxes)
        >>> np.round(out_bboxes, 2).tolist()
        [[2.0, 0.1, 0.2, 0.3, 0.4, 15.0]]
        >>> ordering.tolist()
        [5]
        >>> tiny = np.array([[0.0, 0.0, 0.01, 0.01, 0, 1, 3]], dtype=float)
        >>> postprocess_bboxes(tiny, area_threshold=0.1)[0].shape
        (0, 6)
        >>> postprocess_bboxes(np.empty((0, 7)), area_threshold=0.1)[0].shape
        (0, 6)

    """
    if bboxes.size == 0:
        return np.zeros((0, 6)), np.zeros((0,), dtype=int)
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

    if raw_bboxes.size == 0:
        return np.zeros((0, 6)), refined_ordering.astype(int)

    if bbox_corners is not None and bbox_corners.size > 0:
        if image_height is None or image_width is None:
            raise ValueError(
                "`image_height` and `image_width` are required when "
                "`bbox_corners` is provided."
            )
        corners = bbox_corners.reshape(-1, 4, 3)
        selected_corners = corners[refined_ordering.astype(int), :, :2]
        out_bboxes = _corners_to_ldf_bboxes(
            selected_corners,
            raw_bboxes[:, 5],
            image_height,
            image_width,
        )
    else:
        raw_bboxes[:, 4] = [
            normalize_angle_degrees(angle) for angle in raw_bboxes[:, 4]
        ]
        out_bboxes = raw_bboxes[:, [5, 0, 1, 2, 3, 4]]

    return out_bboxes, refined_ordering.astype(int)


def _corners_to_ldf_bboxes(
    corners: np.ndarray,
    class_ids: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    centers = corners.mean(axis=1)
    width_vectors = corners[:, 1] - corners[:, 0]
    height_vectors = corners[:, 2] - corners[:, 1]
    box_widths = np.linalg.norm(width_vectors, axis=1) / image_width
    box_heights = np.linalg.norm(height_vectors, axis=1) / image_height
    angles = np.array(
        [
            normalize_angle_degrees(
                math.degrees(math.atan2(-vector[1], vector[0]))
            )
            for vector in width_vectors
        ]
    )

    box_widths = np.clip(box_widths, 0.0, 1.0)
    box_heights = np.clip(box_heights, 0.0, 1.0)
    x = centers[:, 0] / image_width - box_widths / 2
    y = centers[:, 1] / image_height - box_heights / 2
    x = np.minimum(np.clip(x, 0.0, 1.0), 1.0 - box_widths)
    y = np.minimum(np.clip(y, 0.0, 1.0), 1.0 - box_heights)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    box_widths = np.minimum(box_widths, 1.0 - x)
    box_heights = np.minimum(box_heights, 1.0 - y)

    return np.column_stack((class_ids, x, y, box_widths, box_heights, angles))


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
