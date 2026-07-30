import json

import albumentations as A
import numpy as np

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.augmentations.custom import TRANSFORMATIONS
from luxonis_ml.data.utils.cli_utils import get_applied_augmentations
from luxonis_ml.typing import LoaderMultiOutput, Params


def _make_sample(size: int = 64) -> list[LoaderMultiOutput]:
    return [({"image": np.zeros((size, size, 3), dtype=np.uint8)}, {})]


def test_inspect_reads_augmentation_metadata():
    assert get_applied_augmentations(
        {"augmentations": {"HorizontalFlip": {}}}
    ) == ["HorizontalFlip"]
    assert get_applied_augmentations({"augmentations": ["invalid"]}) == []


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
    engine = AlbumentationsEngine(64, 64, {}, {}, ["image"], config, seed=2)

    engine.apply(_make_sample())

    applied = engine.applied_augmentations
    assert list(applied) == [
        "HorizontalFlip",
        "OneOf/RandomBrightnessContrast",
    ]
    assert applied["HorizontalFlip"] == {}
    assert isinstance(
        applied["OneOf/RandomBrightnessContrast"]["alpha"], float
    )
    assert isinstance(applied["OneOf/RandomBrightnessContrast"]["beta"], float)
    assert "OneOf/GaussianBlur" not in applied
    assert json.loads(json.dumps(applied)) == applied

    applied["OneOf/RandomBrightnessContrast"]["alpha"] = 0.0
    assert (
        engine.applied_augmentations["OneOf/RandomBrightnessContrast"]["alpha"]
        != 0.0
    )


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
    assert engine.applied_augmentations == {"HorizontalFlip": {}}

    engine._spatial_compose.transforms[0].p = 0.0
    engine.apply(_make_sample())

    assert engine.applied_augmentations == {}


def test_omits_array_and_known_non_random_parameters():
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
    # `matrix` and `bbox_matrix` are arrays, the rest is static.
    assert {"matrix", "bbox_matrix"}.isdisjoint(params)
    assert {"shape", "interpolation", "fill", "fill_mask"}.isdisjoint(params)
    json.dumps(params)


def test_reports_randomly_derived_crop_bounds():
    """The crop bounds of `Rotate` are sampled at runtime.

    They used to be suppressed as non-random parameters, which left the
    transformation reporting an empty parameter dictionary.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "Rotate",
                "params": {"limit": 45, "p": 1.0, "crop_border": True},
            }
        ],
        seed=7,
    )

    engine.apply(_make_sample())

    params = engine.applied_augmentations["Rotate"]
    assert {"x_min", "x_max", "y_min", "y_max"} <= set(params)
    assert params["x_min"] < params["x_max"]  # type: ignore[operator]
    assert params["y_min"] < params["y_max"]  # type: ignore[operator]


def test_omits_nested_arrays_and_known_non_random_parameters():
    params = AlbumentationsEngine._normalize_augmentation_params(
        {
            "large": np.zeros((32, 32)),
            "nested": {
                "shape": (64, 64, 3),
                "selected": np.float32(0.5),
            },
            "values": [1, np.zeros(1)],
        }
    )

    # ``values`` is dropped whole: reporting only the surviving items
    # would reindex the sequence and misrepresent its length.
    assert params == {"nested": {"selected": 0.5}}


def test_omits_sequences_of_arrays_instead_of_emptying_them():
    """`RandomShadow` reports its shadow polygons as a list of arrays.

    Filtering the arrays out of the list used to leave an empty list,
    which reads as "applied, but drew no shadows".
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "RandomShadow", "params": {"p": 1.0}}],
        seed=3,
    )

    engine.apply(_make_sample())

    params = engine.applied_augmentations["RandomShadow"]
    assert "vertices_list" not in params


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
    assert set(params) == {"alpha"}


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
    assert "image_shapes" not in params
    assert {"out_width", "out_height"}.isdisjoint(params)
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


def test_tracks_nested_oneof_selection_without_the_parent_paths():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "OneOf",
                "params": {
                    "transforms": [
                        {
                            "name": "OneOf",
                            "params": {
                                "transforms": [
                                    {
                                        "name": "RandomBrightnessContrast",
                                        "params": {"p": 1.0},
                                    },
                                ],
                                "p": 1.0,
                            },
                        }
                    ],
                    "p": 1.0,
                },
            }
        ],
    )

    engine.apply(_make_sample())

    assert list(engine.applied_augmentations) == [
        "OneOf/OneOf/RandomBrightnessContrast"
    ]


def test_disambiguates_repeated_configured_paths():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {"name": "HorizontalFlip", "params": {"p": 1.0}},
            {"name": "HorizontalFlip", "params": {"p": 1.0}},
        ],
    )

    engine.apply(_make_sample())

    assert engine.applied_augmentations == {
        "HorizontalFlip#1": {},
        "HorizontalFlip#2": {},
    }


def test_keeps_parameters_of_every_repeated_configured_path():
    """Repeated entries can be configured with different parameters.

    Collapsing them into a single entry reported the first one's runtime
    parameters for both, which falsifies the provenance of the second.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {"name": "Blur", "params": {"blur_limit": (3, 3), "p": 1.0}},
            {"name": "Blur", "params": {"blur_limit": (21, 21), "p": 1.0}},
        ],
        seed=1,
    )

    engine.apply(_make_sample())

    assert engine.applied_augmentations == {
        "Blur#1": {"kernel": 3},
        "Blur#2": {"kernel": 21},
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
        seed=2,
    )

    engine.apply(_make_sample(32))

    assert engine.applied_augmentations == {"OneOf/Resize": {}}


def test_tracks_the_injected_fallback_of_a_probabilistic_resize():
    """The engine injects a default resize next to a probabilistic one.

    When that branch is picked the sample really was resized, so reporting
    nothing would be indistinguishable from an un-augmented sample.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "Resize",
                "params": {"height": 64, "width": 64, "p": 0.01},
                "use_for_resizing": True,
            }
        ],
        seed=5,
    )

    engine.apply(_make_sample(128))

    assert list(engine.applied_augmentations) == [
        "OneOf/LetterboxResize (fallback)"
    ]


def test_tracks_nested_transforms_carrying_unknown_keys():
    """Nested items may carry keys the pipeline builder ignores.

    Re-validating them against the strict configuration model rejected
    configurations that the engine is otherwise happy to build.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "OneOf",
                "params": {
                    "transforms": [
                        {
                            "name": "RandomBrightnessContrast",
                            "params": {"p": 1.0},
                            "active": True,
                        }
                    ],
                    "p": 1.0,
                },
            }
        ],
        seed=1,
    )

    engine.apply(_make_sample())

    assert list(engine.applied_augmentations) == [
        "OneOf/RandomBrightnessContrast"
    ]


def test_tracks_a_configured_lambda_transform():
    """Only the `A.Lambda` instances the engine injects are internal.

    Skipping every transformation whose class is named ``Lambda`` also
    hid the ones the user configured.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "Lambda", "params": {"p": 1.0}}],
        seed=1,
    )

    engine.apply(_make_sample())

    assert list(engine.applied_augmentations) == ["Lambda"]


def test_tracks_transforms_registered_under_an_alias():
    """A transformation may be registered under a name of its own.

    Matching runtime transformations by class name never found those, so
    they could not be tracked at all.
    """

    class AliasedBlur(A.ImageOnlyTransform):
        def apply(self, img: np.ndarray, **_) -> np.ndarray:
            return img

        def get_params(self) -> Params:
            return {"strength": 7}

    TRANSFORMATIONS.register(module=AliasedBlur, name="AliasedName")
    try:
        engine = AlbumentationsEngine(
            64,
            64,
            {},
            {},
            ["image"],
            [{"name": "AliasedName", "params": {"p": 1.0}}],
            seed=1,
        )

        engine.apply(_make_sample())

        assert engine.applied_augmentations == {"AliasedName": {"strength": 7}}
    finally:
        TRANSFORMATIONS._module_dict.pop("AliasedName", None)


def test_tracks_batch_transforms_applied_in_any_sub_batch():
    """A batch transform is invoked once per sub-batch.

    Reading ``params`` once after the composition ran only described the
    last invocation, so an applied augmentation could go unreported.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "Mosaic4",
                "params": {"p": 0.5, "out_width": 64, "out_height": 64},
            },
            {"name": "MixUp", "params": {"p": 1.0}},
        ],
        seed=1,
    )

    engine.apply(_make_sample() * 8)

    # All eight inputs contributed, so both batch transforms applied.
    assert engine.batch_augmentation_indices == [0, 1, 2, 3, 4]
    assert set(engine.applied_augmentations) == {"Mosaic4", "MixUp"}
