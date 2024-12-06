from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple, TypeVar

import numpy as np


def preprocess_mask(seg: np.ndarray) -> np.ndarray:
    mask = np.argmax(seg, axis=0) + 1

    # only background has value 0
    mask[np.sum(seg, axis=0) == 0] = 0

    return mask


def preprocess_bboxes(bboxes: np.ndarray, bbox_counter: int) -> np.ndarray:
    bboxes = bboxes[:, [1, 2, 3, 4, 0]]
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    ordering = np.arange(bbox_counter, bboxes.shape[0] + bbox_counter)
    return np.concatenate((bboxes, ordering[:, None]), axis=1)


def preprocess_keypoints(
    keypoints: np.ndarray, height: int, width: int
) -> np.ndarray:
    keypoints = np.reshape(keypoints, (-1, 3))
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height
    return keypoints


def postprocess_mask(mask: np.ndarray, n_classes: int) -> np.ndarray:
    out_mask = np.zeros((n_classes, *mask.shape))
    for key in np.unique(mask):
        if key != 0:
            out_mask[int(key) - 1, ...] = mask == key
    out_mask[out_mask > 0] = 1
    return out_mask


def postprocess_bboxes(bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if bboxes.shape[0] == 0:
        return np.zeros((0, 6)), np.zeros((0, 1))
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
    keypoints = _mark_invisible_keypoints(keypoints, image_height, image_width)
    keypoints[..., 0] /= image_width
    keypoints[..., 1] /= image_height
    keypoints = np.reshape(keypoints[:, :3], (-1, n_keypoints * 3))[
        bboxes_ordering
    ]
    return keypoints


def _mark_invisible_keypoints(
    keypoints: np.ndarray, height: int, width: int
) -> np.ndarray:
    for kp in keypoints:
        if not (0 <= kp[0] < width and 0 <= kp[1] < height):
            kp[2] = 0

        # per COCO format invisible points have x=y=0
        if kp[2] == 0:
            kp[0] = kp[1] = 0

    return keypoints


K = TypeVar("K")
V = TypeVar("V")


def reverse_dictionary(d: Dict[K, V]) -> Dict[V, K]:
    return {v: k for k, v in d.items()}


def yield_batches(
    data: List[Dict[str, Any]], batch_size: int
) -> Iterator[Dict[str, List[Any]]]:
    """Yield batches of data."""
    for i in range(0, len(data), batch_size):
        yield list2batch(data[i : i + batch_size])


def list2batch(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert from a list of normal target dicts to a batched target
    dict."""

    batch = defaultdict(list)
    for item in data:
        for k, v in item.items():
            batch[k].append(v)

    return dict(batch)
