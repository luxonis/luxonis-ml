from copy import deepcopy

import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.typing import Labels


def assert_boxes_match_masks(
    boxes: np.ndarray, masks: np.ndarray, tol: float = 0.05
) -> None:
    """Assert every instance mask is paired with the correct bounding box.

    Box and instance mask undergo the same geometric transforms, so a
    correctly paired mask always has its nonzero pixels inside its box.
    A reindexing bug that scrambles the box-to-mask mapping (e.g. pairing a
    box with a mask blob located elsewhere in the image) is caught here even
    when the row counts still match.
    """
    assert boxes.shape[0] == masks.shape[0]
    _, height, width = masks.shape
    for i, (box, mask) in enumerate(zip(boxes, masks, strict=True)):
        ys, xs = np.nonzero(mask)
        assert ys.size > 0, f"instance {i} has an empty mask"
        cx = xs.mean() / width
        cy = ys.mean() / height
        x, y, w, h = box[1:5]
        assert x - tol <= cx <= x + w + tol, (
            f"instance {i}: mask centroid x={cx:.3f} outside box "
            f"[{x:.3f}, {x + w:.3f}] — box paired with the wrong mask"
        )
        assert y - tol <= cy <= y + h + tol, (
            f"instance {i}: mask centroid y={cy:.3f} outside box "
            f"[{y:.3f}, {y + h:.3f}] — box paired with the wrong mask"
        )


@pytest.fixture(
    params=[
        {"image": np.zeros((320, 320, 3), dtype=np.uint8)},
        {
            "rgb_image": np.zeros((320, 320, 3), dtype=np.uint8),
            "ir_image": np.zeros((320, 320, 1), dtype=np.uint8),
        },
        {
            "left_img": np.zeros((320, 320, 1), dtype=np.uint8),
            "right_img": np.zeros((320, 320, 1), dtype=np.uint8),
            "middle_img": np.zeros((320, 320, 3), dtype=np.uint8),
        },
    ]
)
def images_dict(request: pytest.FixtureRequest) -> dict[str, np.ndarray]:
    return request.param


@pytest.fixture
def labels() -> Labels:
    return {
        "task/classification": np.array([1.0]),
        "task/boundingbox": np.array(
            [
                [0.0, 0.57, 0.30, 0.17, 0.25],
                [0.0, 0.39, 0.27, 0.20, 0.10],
            ]
        ),
        "task/keypoints": np.array(
            [
                [0.69, 0.37, 2.0, 0.0, 0.0, 0.0, 0.68, 0.36, 2.0],
                [0.51, 0.33, 2.0, 0.0, 0.0, 0.0, 0.50, 0.32, 2.0],
            ]
        ),
        "task/segmentation": np.zeros((1, 320, 320)),
    }


@pytest.fixture
def targets() -> dict[str, str]:
    return {
        "task/boundingbox": "boundingbox",
        "task/keypoints": "keypoints",
        "task/segmentation": "segmentation",
    }


@pytest.fixture
def n_classes() -> dict[str, int]:
    return {
        "task/boundingbox": 1,
        "task/keypoints": 1,
        "task/segmentation": 1,
    }


def test_mosaic4(
    images_dict: dict[str, np.ndarray],
    labels: Labels,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        }
    ]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    augmentations.apply([(images_dict, deepcopy(labels)) for _ in range(4)])


def test_mixup(
    images_dict: dict[str, np.ndarray],
    labels: Labels,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [{"name": "MixUp", "params": {"p": 1.0}}]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    augmentations.apply([(images_dict, deepcopy(labels)) for _ in range(2)])


def test_at_least_one_bbox_random_crop() -> None:
    """Test that AtLeastOneBBoxRandomCrop guarantees at least one bbox.

    This is a test for ensuring that the correct "bboxes" key is passed
    to Albumentations transforms that read data["bboxes"] directly.
    """
    image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    images_dict = {"image": image}
    labels: Labels = {
        "task/boundingbox": np.array(
            [
                [0.0, 0.5, 0.5, 0.1, 0.1],
            ]
        ),
    }
    targets = {"task/boundingbox": "boundingbox"}
    n_classes = {"task/boundingbox": 1}
    config = [
        {
            "name": "AtLeastOneBBoxRandomCrop",
            "params": {
                "height": 40,
                "width": 40,
                "erosion_factor": 0.0,
                "p": 1.0,
            },
        }
    ]
    engine = AlbumentationsEngine(
        256, 256, targets, n_classes, ["image"], config
    )
    for _ in range(10):
        _, out_labels = engine.apply([(images_dict, deepcopy(labels))])
        bboxes = out_labels.get("task/boundingbox")
        assert bboxes is not None, (
            "AtLeastOneBBoxRandomCrop should produce bounding box output"
        )
        assert len(bboxes) > 0, (
            "AtLeastOneBBoxRandomCrop should guarantee at least one "
            "bounding box per crop"
        )


def test_batched_p_0(
    images_dict: dict[str, np.ndarray],
    labels: Labels,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 0, "out_width": 640, "out_height": 640},
        },
        {"name": "MixUp", "params": {"p": 0}},
    ]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    augmentations.apply([(images_dict, deepcopy(labels)) for _ in range(8)])


def test_skipped_mosaic_before_mixup_reindexes_instance_masks() -> None:
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    # Two instances whose masks are centered inside their boxes, so a
    # box-to-mask mispairing is detectable geometrically.
    instance_mask = np.zeros((2, 320, 320), dtype=np.uint8)
    instance_mask[0, 48:112, 48:112] = 1  # box 0 spans [0.15, 0.35]
    instance_mask[1, 192:256, 192:256] = 1  # box 1 spans [0.60, 0.80]
    labels: Labels = {
        "task/instance_segmentation/boundingbox": np.array(
            [
                [0.0, 0.15, 0.15, 0.20, 0.20],
                [0.0, 0.60, 0.60, 0.20, 0.20],
            ]
        ),
        "task/instance_segmentation/segmentation": instance_mask,
    }
    targets = {
        "task/instance_segmentation/boundingbox": "boundingbox",
        "task/instance_segmentation/segmentation": "instance_segmentation",
    }
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 0, "out_width": 640, "out_height": 640},
        },
        {"name": "MixUp", "params": {"p": 1}},
    ]
    augmentations = AlbumentationsEngine(
        256,
        256,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        config,
    )

    _, out_labels = augmentations.apply(
        [({"image": image}, deepcopy(labels)) for _ in range(8)]
    )

    boxes = out_labels["task/instance_segmentation/boundingbox"]
    masks = out_labels["task/instance_segmentation/segmentation"]
    assert boxes.shape[0] == masks.shape[0] == 4
    assert_boxes_match_masks(boxes, masks)


def test_mosaic_drops_boxes_keeps_masks_aligned() -> None:
    """Boxes dropped by ``min_bbox_visibility`` must not scramble masks.

    When a batch transform (here Mosaic4) crops instances so some boxes fall
    below the visibility threshold, ``check_data_post_transform`` drops those
    boxes while every instance-mask channel remains. The surviving boxes must
    still be paired with their own masks; reindexing them to a contiguous
    range after the drop would silently pair them with the wrong masks.
    """
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    # Four instances in the four corners, each mask centered in its box.
    corners = [(0.05, 0.05), (0.80, 0.05), (0.05, 0.80), (0.80, 0.80)]
    size = 0.15
    instance_mask = np.zeros((4, 320, 320), dtype=np.uint8)
    boxes_in = []
    for i, (x, y) in enumerate(corners):
        r0, r1 = int(y * 320), int((y + size) * 320)
        c0, c1 = int(x * 320), int((x + size) * 320)
        instance_mask[i, r0:r1, c0:c1] = 1
        boxes_in.append([0.0, x, y, size, size])
    labels: Labels = {
        "task/instance_segmentation/boundingbox": np.array(boxes_in),
        "task/instance_segmentation/segmentation": instance_mask,
    }
    targets = {
        "task/instance_segmentation/boundingbox": "boundingbox",
        "task/instance_segmentation/segmentation": "instance_segmentation",
    }
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        }
    ]
    augmentations = AlbumentationsEngine(
        256,
        256,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        config,
        min_bbox_visibility=0.1,
        seed=42,
    )

    _, out_labels = augmentations.apply(
        [({"image": image}, deepcopy(labels)) for _ in range(4)]
    )

    boxes = out_labels["task/instance_segmentation/boundingbox"]
    masks = out_labels["task/instance_segmentation/segmentation"]
    # The mosaic crop must drop at least one of the 16 candidate instances,
    # otherwise the non-contiguous-index path is never exercised.
    assert 0 < boxes.shape[0] < 16
    assert_boxes_match_masks(boxes, masks)
