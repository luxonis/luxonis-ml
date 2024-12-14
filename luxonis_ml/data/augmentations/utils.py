from typing import Any, Dict, Iterator, List, Tuple

import numpy as np


def preprocess_mask(seg: np.ndarray) -> np.ndarray:
    return seg.transpose(1, 2, 0)


def preprocess_bboxes(bboxes: np.ndarray, bbox_counter: int) -> np.ndarray:
    bboxes = bboxes[:, [1, 2, 3, 4, 0]]
    # adding 1e-6 to avoid zero width or height
    bboxes[:, 2] += bboxes[:, 0] + 1e-6
    bboxes[:, 3] += bboxes[:, 1] + 1e-6
    ordering = np.arange(
        bbox_counter, bboxes.shape[0] + bbox_counter, dtype=bboxes.dtype
    )[:, None]
    return np.concatenate((bboxes, ordering), axis=1)


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
    if bboxes.size == 0:
        return np.zeros((0, 6)), np.zeros((0, 1), dtype=np.uint8)

    ordering = bboxes[:, -1]
    out_bboxes = bboxes[:, :-1]
    out_bboxes[:, 2] -= out_bboxes[:, 0]
    out_bboxes[:, 3] -= out_bboxes[:, 1]

    return out_bboxes[:, [4, 0, 1, 2, 3]], ordering.astype(np.uint8)


def postprocess_keypoints(
    keypoints: np.ndarray,
    bboxes_ordering: np.ndarray,
    image_height: int,
    image_width: int,
    n_keypoints: int,
) -> np.ndarray:
    keypoints = np.reshape(keypoints[:, :3], (-1, n_keypoints * 3))[
        bboxes_ordering
    ]
    np.maximum(keypoints, 0, out=keypoints)
    keypoints[..., ::3] /= image_width
    keypoints[..., 1::3] /= image_height
    return keypoints


def yield_batches(
    data_batch: List[Dict[str, Any]], batch_size: int
) -> Iterator[Dict[str, List[Any]]]:
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
