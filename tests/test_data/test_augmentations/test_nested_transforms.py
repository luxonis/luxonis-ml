import numpy as np

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.typing import LoaderMultiOutput, Params


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
