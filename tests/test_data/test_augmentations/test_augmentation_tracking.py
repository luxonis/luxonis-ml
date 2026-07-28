import json

import numpy as np

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.__main__ import _get_applied_augmentations
from luxonis_ml.typing import LoaderMultiOutput, Params


def _make_sample(size: int = 64) -> list[LoaderMultiOutput]:
    return [({"image": np.zeros((size, size, 3), dtype=np.uint8)}, {})]


def test_inspect_reads_augmentation_metadata():
    assert _get_applied_augmentations(
        {"augmentations": {"HorizontalFlip": {"shape": [64, 64, 3]}}}
    ) == ["HorizontalFlip"]
    assert _get_applied_augmentations({"augmentations": ["invalid"]}) == []


def test_tracks_only_configured_augmentations_that_are_applied():
    config: list[Params] = [
        {"name": "HorizontalFlip", "params": {"p": 1.0}},
        {"name": "Defocus", "params": {"p": 0.0}},
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {
                        "name": "RandomBrightnessContrast",
                        "params": {"p": 1.0},
                    },
                    {"name": "GaussianBlur", "params": {"p": 1.0}},
                ],
                "p": 1.0,
            },
        },
    ]
    engine = AlbumentationsEngine(64, 64, {}, {}, ["image"], config)
    one_of = engine._spatial_compose.transforms[1]
    one_of.transforms_ps = [1.0, 0.0]

    engine.apply(_make_sample())

    applied = engine.applied_augmentations
    assert list(applied) == [
        "HorizontalFlip",
        "OneOf/RandomBrightnessContrast",
    ]
    assert applied["HorizontalFlip"] == {"shape": [64, 64, 3]}
    assert isinstance(
        applied["OneOf/RandomBrightnessContrast"]["alpha"], float
    )
    assert isinstance(applied["OneOf/RandomBrightnessContrast"]["beta"], float)
    json.dumps(applied)

    applied["HorizontalFlip"]["shape"][0] = 0  # type: ignore[index]
    assert engine.applied_augmentations["HorizontalFlip"] == {
        "shape": [64, 64, 3]
    }


def test_clears_applied_augmentations_between_calls():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "HorizontalFlip", "params": {"p": 1.0}}],
    )

    engine.apply(_make_sample())
    assert engine.applied_augmentations == {
        "HorizontalFlip": {"shape": [64, 64, 3]}
    }

    engine._spatial_compose.transforms[0].p = 0.0
    engine.apply(_make_sample())

    assert engine.applied_augmentations == {}


def test_normalizes_numpy_augmentation_parameters_for_metadata():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "Rotate", "params": {"limit": 15, "p": 1.0}}],
        seed=42,
    )

    engine.apply(_make_sample())

    params = engine.applied_augmentations["Rotate"]
    assert isinstance(params["matrix"], list)
    assert isinstance(params["matrix"][0], list)  # type: ignore[index]
    json.dumps(params)


def test_tracks_applied_batch_augmentations():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "MixUp", "params": {"p": 1.0, "alpha": [0.3, 0.7]}}],
    )

    engine.apply(_make_sample() * 2)

    params = engine.applied_augmentations["MixUp"]
    assert 0.3 <= params["alpha"] <= 0.7  # type: ignore[operator]
    assert params["image_shapes"] == [[64, 64], [64, 64]]


def test_tracks_mosaic_runtime_parameters():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "Mosaic4",
                "params": {"p": 1.0, "out_width": 64, "out_height": 64},
            }
        ],
        seed=42,
    )

    engine.apply(_make_sample() * 4)

    params = engine.applied_augmentations["Mosaic4"]
    assert params["image_shapes"] == [[64, 64]] * 4
    assert isinstance(params["x_crop"], int)
    assert isinstance(params["y_crop"], int)


def test_tracks_every_selected_someof_transform():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "SomeOf",
                "params": {
                    "n": 2,
                    "replace": False,
                    "transforms": [
                        {
                            "name": "RandomBrightnessContrast",
                            "params": {"p": 1.0},
                        },
                        {"name": "GaussianBlur", "params": {"p": 1.0}},
                    ],
                    "p": 1.0,
                },
            }
        ],
        seed=42,
    )

    engine.apply(_make_sample())

    assert set(engine.applied_augmentations) == {
        "SomeOf/RandomBrightnessContrast",
        "SomeOf/GaussianBlur",
    }


def test_tracks_probabilistic_resize_under_its_oneof_path():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "Resize",
                "params": {
                    "height": 64,
                    "width": 64,
                    "p": 0.5,
                },
                "use_for_resizing": True,
            }
        ],
    )

    resize_oneof = engine._resize_compose.transforms[0]
    resize_oneof.transforms[0].p = 1.0
    resize_oneof.transforms[1].p = 0.0
    resize_oneof.transforms_ps = [1.0, 0.0]

    engine.apply(_make_sample(32))

    assert engine.applied_augmentations == {
        "OneOf/Resize": {"shape": [32, 32, 3], "interpolation": 1}
    }
