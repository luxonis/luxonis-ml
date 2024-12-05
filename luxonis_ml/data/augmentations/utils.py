from typing import Tuple

import numpy as np

from luxonis_ml.typing import Labels


def prepare_mask(labels: Labels, height: int, width: int) -> np.ndarray:
    seg = labels.get("segmentation", np.zeros((1, height, width)))
    mask = np.argmax(seg, axis=0) + 1

    # only background has value 0
    mask[np.sum(seg, axis=0) == 0] = 0

    return mask


def prepare_bboxes(
    labels: Labels, height: int, width: int
) -> Tuple[np.ndarray, np.ndarray]:
    bboxes = labels.get("boundingbox", np.zeros((0, 5)))
    bboxes_points = bboxes[:, 1:]
    bboxes_points[:, 0::2] *= width
    bboxes_points[:, 1::2] *= height
    bboxes_points = _check_bboxes(bboxes_points)
    bboxes_classes = bboxes[:, 0]
    return bboxes_points, bboxes_classes


def prepare_keypoints(
    labels: Labels, height: int, width: int, n_keypoints: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoints = labels.get("keypoints", np.zeros((1, n_keypoints * 3 + 1)))
    keypoints_unflat = np.reshape(keypoints[:, 1:], (-1, 3))
    keypoints_points = keypoints_unflat[:, :2]
    keypoints_points[:, 0] *= width
    keypoints_points[:, 1] *= height
    keypoints_visibility = keypoints_unflat[:, 2]

    keypoints_classes = np.repeat(keypoints[:, 0], n_keypoints)

    return keypoints_points, keypoints_visibility, keypoints_classes


def post_process_mask(mask: np.ndarray, n_classes: int) -> np.ndarray:
    out_mask = np.zeros((n_classes, *mask.shape))
    for key in np.unique(mask):
        if key != 0:
            out_mask[int(key) - 1, ...] = mask == key
    out_mask[out_mask > 0] = 1
    return out_mask


def post_process_bboxes(
    bboxes: np.ndarray,
    classes: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    if bboxes.shape[0] > 0:
        out_bboxes = np.concatenate([classes, bboxes], axis=1)
        out_bboxes[:, 1::2] /= image_width
        out_bboxes[:, 2::2] /= image_height
        return out_bboxes
    return np.zeros((0, 5))


def post_process_keypoints(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    classes: np.ndarray,
    n_keypoints: int,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    visibility = np.expand_dims(visibility, axis=-1)

    if n_keypoints == 0:
        n_keypoints = 1

    if keypoints.shape[0] > 0:
        out_keypoints = np.concatenate(
            (keypoints, visibility),
            axis=1,
        )
    else:
        out_keypoints = np.zeros((0, n_keypoints * 3 + 1))

    out_keypoints = _mark_invisible_keypoints(
        out_keypoints, image_height, image_width
    )
    out_keypoints[..., 0] /= image_width
    out_keypoints[..., 1] /= image_height
    out_keypoints = np.reshape(out_keypoints, (-1, n_keypoints * 3))
    classes = classes[0::n_keypoints]
    classes = np.expand_dims(classes, axis=-1)
    out_keypoints = np.concatenate((classes, out_keypoints), axis=1)
    return out_keypoints


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


def _check_bboxes(bboxes: np.ndarray) -> np.ndarray:
    for i in range(bboxes.shape[0]):
        if bboxes[i, 2] == 0:
            bboxes[i, 2] = 1
        if bboxes[i, 3] == 0:
            bboxes[i, 3] = 1
    return bboxes
