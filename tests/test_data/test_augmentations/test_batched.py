from copy import deepcopy

import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.typing import Labels


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
                [0.0, 0.57, 0.30, 0.17, 0.25, 0.0],
                [0.0, 0.39, 0.27, 0.20, 0.10, 0.0],
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
                [0.0, 0.5, 0.5, 0.1, 0.1, 0.0],
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


def test_affine_letterbox_preserves_obb_corners() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    labels: Labels = {
        "task/boundingbox": np.array(
            [
                [0.0, 0.0, 0.0, 0.4, 0.3, 0.0],
            ]
        ),
    }
    targets = {"task/boundingbox": "boundingbox"}
    n_classes = {"task/boundingbox": 1}
    config = [
        {
            "name": "Affine",
            "params": {
                "scale": 1.0,
                "translate_percent": 0.0,
                "rotate": 30,
                "shear": 0,
                "p": 1.0,
            },
        }
    ]
    engine = AlbumentationsEngine(
        512, 512, targets, n_classes, ["image"], config, bbox_area_threshold=0
    )

    _, out_labels = engine.apply([({"image": image}, labels)])
    bboxes = out_labels["task/boundingbox"]

    assert bboxes.shape == (1, 6)
    assert bboxes[0, 1] >= 0
    assert bboxes[0, 2] >= 0
    assert bboxes[0, 1] + bboxes[0, 3] <= 1
    assert bboxes[0, 2] + bboxes[0, 4] <= 1
    assert bboxes[0, 3] == pytest.approx(0.4)
    assert bboxes[0, 4] == pytest.approx(0.15)
    assert bboxes[0, 5] == pytest.approx(30, abs=1e-3)


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
