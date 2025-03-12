from typing import Dict, Iterator, List, Tuple, TypeVar

import numpy as np


def preprocess_mask(seg: np.ndarray) -> np.ndarray:
    return seg.transpose(1, 2, 0)


def preprocess_bboxes(bboxes: np.ndarray, bbox_counter: int) -> np.ndarray:
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
    keypoints = np.reshape(keypoints, (-1, 3))
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return keypoints


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return mask[None, ...]

    return mask.transpose(2, 0, 1)


def postprocess_bboxes(bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    area_threshold = 0.0004  # 0.02 * 0.02 Small area threshold to remove invalid bboxes and respective keypoints.
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
    data_batch: List[Dict[str, T]], batch_size: int
) -> Iterator[Dict[str, List[T]]]:
    """Yield batches of data.

    @type data_batch: List[Dict[str, Any]]
    @param data_batch: List of dictionaries containing data.
    @type batch_size: int
    @param batch_size: Size of the batch.
    @rtype: Iterator[Dict[str, List[Any]]]
    @return: Generator of batches of data.
    """
    for i in range(0, len(data_batch), batch_size):
        yield {
            target: [data[target] for data in data_batch[i : i + batch_size]]
            for target in data_batch[0]
        }
