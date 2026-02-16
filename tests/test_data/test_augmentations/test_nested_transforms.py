from copy import deepcopy

import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.typing import Labels, LoaderMultiOutput, Params

H, W = 320, 320
SEED = 42


def _make_engine(config: list[Params]) -> AlbumentationsEngine:
    return AlbumentationsEngine(
        256,
        256,
        {"/classification": "classification"},
        {"/classification": 2},
        ["image"],
        config,
    )


def _make_sample() -> list[LoaderMultiOutput]:
    return [
        (
            {
                "image": np.random.randint(
                    0, 255, (3, 256, 256), dtype=np.uint8
                )
            },
            {"/classification": np.array([0])},
        )
    ]


def test_oneof_in_pipeline():
    # Nested OneOf config is parsed and runs without error
    config = [
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {"name": "GaussianBlur", "params": {"blur_limit": (3, 7)}},
                    {"name": "MotionBlur", "params": {"blur_limit": (3, 5)}},
                ],
                "p": 1.0,
            },
        },
    ]
    images, _ = _make_engine(config).apply(_make_sample())
    assert images["image"].shape == (256, 256, 256)


def test_someof_in_pipeline():
    # Nested SomeOf config is parsed and runs without error
    config = [
        {
            "name": "SomeOf",
            "params": {
                "n": 1,
                "transforms": [
                    {"name": "GaussianBlur", "params": {"blur_limit": (3, 7)}},
                    {"name": "MotionBlur", "params": {"blur_limit": (3, 5)}},
                ],
                "p": 1.0,
            },
        },
    ]
    images, _ = _make_engine(config).apply(_make_sample())
    assert images["image"].shape == (256, 256, 256)


def test_mixed_pipeline():
    # Composition and regular transforms coexist without error
    config = [
        {"name": "HorizontalFlip", "params": {"p": 1.0}},
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {"name": "GaussianBlur", "params": {"blur_limit": (3, 7)}},
                    {"name": "MotionBlur", "params": {"blur_limit": (3, 5)}},
                ],
                "p": 1.0,
            },
        },
    ]
    images, _ = _make_engine(config).apply(_make_sample())
    assert images["image"].shape == (256, 256, 256)


def test_batch_transform_inside_composition():
    """BatchTransforms nested inside compositions should raise an error."""
    with pytest.raises(ValueError, match="cannot be nested inside"):
        _make_engine(
            [
                {
                    "name": "OneOf",
                    "params": {
                        "transforms": [
                            {
                                "name": "Mosaic4",
                                "params": {
                                    "out_height": 64,
                                    "out_width": 64,
                                    "p": 1,
                                },
                            },
                        ],
                        "p": 1,
                    },
                }
            ]
        )

    with pytest.raises(ValueError, match="cannot be nested inside"):
        _make_engine(
            [
                {
                    "name": "SomeOf",
                    "params": {
                        "n": 1,
                        "transforms": [
                            {"name": "MixUp", "params": {"p": 1}},
                        ],
                        "p": 1,
                    },
                }
            ]
        )


# equivalence tests
@pytest.fixture
def coco_targets() -> dict[str, str]:
    return {
        "task/boundingbox": "boundingbox",
        "task/keypoints": "keypoints",
        "task/segmentation": "segmentation",
        "task/instance_segmentation/boundingbox": "boundingbox",
        "task/instance_segmentation/segmentation": "instance_segmentation",
    }


@pytest.fixture
def coco_n_classes() -> dict[str, int]:
    return {
        "task/boundingbox": 1,
        "task/keypoints": 1,
        "task/segmentation": 1,
        "task/instance_segmentation/boundingbox": 1,
        "task/instance_segmentation/segmentation": 1,
    }


@pytest.fixture
def coco_labels() -> Labels:
    mask = np.zeros((1, H, W), dtype=np.uint8)
    mask[0, 50:200, 50:200] = 1

    instance_mask = np.zeros((2, H, W), dtype=np.uint8)
    instance_mask[0, 30:100, 30:100] = 1
    instance_mask[1, 150:250, 150:250] = 1

    return {
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
        "task/segmentation": mask,
        "task/instance_segmentation/boundingbox": np.array(
            [
                [0.0, 0.09, 0.09, 0.22, 0.22],
                [0.0, 0.47, 0.47, 0.31, 0.31],
            ]
        ),
        "task/instance_segmentation/segmentation": instance_mask,
    }


@pytest.fixture
def coco_image() -> np.ndarray:
    return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)


def _apply_config(
    config: list[Params],
    targets: dict[str, str],
    n_classes: dict[str, int],
    image: np.ndarray,
    labels: Labels,
) -> tuple[dict[str, np.ndarray], Labels]:
    engine = AlbumentationsEngine(
        H, W, targets, n_classes, ["image"], config, seed=SEED
    )
    return engine.apply([({"image": image.copy()}, deepcopy(labels))])


def _assert_results_equal(
    result_a: tuple[dict[str, np.ndarray], Labels],
    result_b: tuple[dict[str, np.ndarray], Labels],
) -> None:
    images_a, labels_a = result_a
    images_b, labels_b = result_b

    np.testing.assert_array_equal(
        images_a["image"], images_b["image"], err_msg="Images differ"
    )

    assert labels_a.keys() == labels_b.keys(), (
        f"Label keys differ: {labels_a.keys()} vs {labels_b.keys()}"
    )
    for key in labels_a:
        np.testing.assert_array_almost_equal(
            labels_a[key],
            labels_b[key],
            err_msg=f"Labels differ for '{key}'",
        )


@pytest.mark.parametrize("wrapper", ["Sequential", "OneOf", "SomeOf"])
@pytest.mark.parametrize(
    "augmentation",
    [
        {"name": "HorizontalFlip", "params": {"p": 1.0}},
        {"name": "VerticalFlip", "params": {"p": 1.0}},
        {"name": "Transpose", "params": {"p": 1.0}},
    ],
    ids=["HorizontalFlip", "VerticalFlip", "Transpose"],
)
def test_nested_equivalent_to_direct(
    wrapper: str,
    augmentation: Params,
    coco_targets: dict[str, str],
    coco_n_classes: dict[str, int],
    coco_image: np.ndarray,
    coco_labels: Labels,
):
    """Wrapping a p=1 augmentation inside
    Sequential, OneOf, or SomeOf must produce exactly the same
    result as using the augmentation directly."""
    direct_config: list[Params] = [augmentation]

    nested_params: dict = {
        "transforms": [augmentation],
        "p": 1.0,
    }
    if wrapper == "SomeOf":
        nested_params["n"] = 1

    nested_config: list[Params] = [
        {"name": wrapper, "params": nested_params},
    ]

    direct_result = _apply_config(
        direct_config,
        coco_targets,
        coco_n_classes,
        coco_image,
        coco_labels,
    )
    nested_result = _apply_config(
        nested_config,
        coco_targets,
        coco_n_classes,
        coco_image,
        coco_labels,
    )

    _assert_results_equal(direct_result, nested_result)
