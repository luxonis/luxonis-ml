import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import albumentations as A
import numpy as np

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.augmentations import BatchCompose, BatchTransform
from luxonis_ml.data.augmentations.albumentations_engine import (
    _normalize_params,
)
from luxonis_ml.data.augmentations.custom import TRANSFORMATIONS
from luxonis_ml.data.utils.cli_utils import (
    get_applied_augmentations,
    get_tracked_augmentations,
)
from luxonis_ml.typing import LoaderMultiOutput, Params, TrackedAugmentations


def _make_sample(size: int = 64) -> list[LoaderMultiOutput]:
    return [({"image": np.zeros((size, size, 3), dtype=np.uint8)}, {})]


@contextmanager
def _registered(
    transformation: type, name: str | None = None
) -> Iterator[str]:
    name = name or transformation.__name__
    TRANSFORMATIONS.register(module=transformation, name=name, force=True)
    try:
        yield name
    finally:
        TRANSFORMATIONS._module_dict.pop(name, None)


class PickSecond(BatchTransform):
    """Batch transform that applies without any runtime parameters."""

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(batch_size=2, p=p)

    def update_transform_params(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        return params

    def apply(self, image_batch: list[np.ndarray], **_: Any) -> np.ndarray:
        return image_batch[1]

    apply_to_mask = apply  # type: ignore[assignment]
    apply_to_bboxes = apply  # type: ignore[assignment]
    apply_to_keypoints = apply  # type: ignore[assignment]
    apply_to_instance_mask = apply  # type: ignore[assignment]


class MergeEverySecondSubBatch(BatchTransform):
    """Batch transform that applies to every second sub-batch."""

    def __init__(self) -> None:
        super().__init__(batch_size=2, p=1.0)
        self.invocations = 0

    def should_apply(self, force_apply: bool = False) -> bool:
        self.invocations += 1
        return self.invocations % 2 == 0

    def apply(self, image_batch: list[np.ndarray], **_: Any) -> np.ndarray:
        return image_batch[1]

    apply_to_mask = apply  # type: ignore[assignment]
    apply_to_bboxes = apply  # type: ignore[assignment]
    apply_to_keypoints = apply  # type: ignore[assignment]
    apply_to_instance_mask = apply  # type: ignore[assignment]


class CountedMerge(BatchTransform):
    """Batch transform reporting which invocation produced its output."""

    def __init__(self) -> None:
        super().__init__(batch_size=2, p=1.0)
        self.invocations = 0

    def get_params(self) -> Params:
        self.invocations += 1
        return {"invocation": self.invocations}

    def apply(self, image_batch: list[np.ndarray], **_: Any) -> np.ndarray:
        return image_batch[0]

    apply_to_mask = apply  # type: ignore[assignment]
    apply_to_bboxes = apply  # type: ignore[assignment]
    apply_to_keypoints = apply  # type: ignore[assignment]
    apply_to_instance_mask = apply  # type: ignore[assignment]


def _image_batch(count: int) -> list[dict[str, np.ndarray]]:
    return [
        {"image": np.full((2, 2, 1), i, dtype=np.uint8)} for i in range(count)
    ]


def test_batch_compose_forgets_a_discarded_sub_batch():
    """A sub-batch a transform applied to can still be thrown away.

    Whether a transform applied was recorded per sub-batch, but a later
    transform that does not apply keeps only its first input. Reporting
    such an invocation claims an augmentation that never touched the
    returned sample.
    """
    merge = MergeEverySecondSubBatch()
    compose = BatchCompose([merge, PickSecond(p=0.0)])

    batch = _image_batch(4)
    transformed = compose(batch)

    # Only the second sub-batch was merged, and the last transform kept the
    # first one, so the first input passed through untouched.
    assert np.array_equal(transformed["image"], batch[0]["image"])
    assert compose.batch_augmentation_indices == [0]
    assert compose.applied_params == {}


def test_batch_compose_reports_the_sub_batch_that_survived():
    """A transform is invoked once per sub-batch, with its own parameters.

    Keeping the last invocation's parameters described a sub-batch that is
    not the one the returned sample was made of.
    """
    merge = CountedMerge()
    compose = BatchCompose([merge, PickSecond(p=0.0)])

    compose(_image_batch(4))

    assert merge.invocations == 2
    assert compose.batch_augmentation_indices == [0, 1]
    assert compose.applied_params[id(merge)]["invocation"] == 1


def test_batch_compose_keeps_applied_transform_without_parameters():
    """An applied batch transform must not have its output discarded.

    Whether a batch transform ran was inferred from ``params`` being
    non-empty. A transform that applies deterministically samples no
    parameters, so its real output was thrown away and the first input
    passed through in its place.
    """
    first = np.zeros((2, 2, 1), dtype=np.uint8)
    second = np.ones((2, 2, 1), dtype=np.uint8)

    compose = BatchCompose([PickSecond()])
    transformed = compose([{"image": first}, {"image": second}])

    assert np.array_equal(transformed["image"], second)
    # Both inputs contributed, so both must be reported as contributors.
    assert compose.batch_augmentation_indices == [0, 1]


def test_skipped_batch_transform_still_passes_through_first_input():
    first = np.zeros((2, 2, 1), dtype=np.uint8)
    second = np.ones((2, 2, 1), dtype=np.uint8)

    compose = BatchCompose([PickSecond(p=0.0)])
    transformed = compose([{"image": first}, {"image": second}])

    assert np.array_equal(transformed["image"], first)
    assert compose.batch_augmentation_indices == [0]
    assert compose.applied_params == {}


def test_tracks_applied_batch_transform_without_runtime_parameters():
    with _registered(PickSecond):
        engine = AlbumentationsEngine(
            64,
            64,
            {},
            {},
            ["image"],
            [{"name": "PickSecond", "params": {"p": 1.0}}],
        )

        engine.apply(_make_sample() * 2)

        assert engine.applied_augmentations == {"PickSecond": {}}


def test_inspect_reads_augmentation_metadata():
    assert get_applied_augmentations(
        {"augmentations": TrackedAugmentations({"HorizontalFlip": {}})}
    ) == ["HorizontalFlip"]
    assert get_applied_augmentations({"augmentations": ["invalid"]}) == []


def test_inspect_does_not_read_stored_metadata_as_augmentations():
    """A record may store its own ``"augmentations"`` metadata.

    The loader keeps the stored value. Its shape must not cause it to be
    mistaken for augmentation provenance and listed as applied transforms.
    """
    stored: Params = {"augmentations": {"user_pipeline": {"run_id": 42}}}

    assert get_tracked_augmentations(stored) is None
    assert get_applied_augmentations(stored) == []
    tracked = {"augmentations": TrackedAugmentations({"Blur": {}})}
    assert get_tracked_augmentations(tracked) == {"Blur": {}}


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


def test_clears_applied_augmentations_between_calls():
    """The report of a call must not leak into the next one.

    The mapping is handed to the loader alongside the sample it describes,
    so the next call has to start from a fresh one rather than reuse it.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "HorizontalFlip", "params": {"p": 1.0}}],
    )

    engine.apply(_make_sample())
    flipped = engine.applied_augmentations
    assert flipped == {"HorizontalFlip": {}}

    engine._spatial_compose.transforms[0].p = 0.0
    engine.apply(_make_sample())

    assert engine.applied_augmentations == {}
    assert flipped == {"HorizontalFlip": {}}


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


def test_reports_a_parameter_sampled_under_a_static_name():
    """``fill`` is a constant for most transformations, but not for all.

    `A.CropAndPad` draws it from the range it was configured with, so
    suppressing the name outright hid the padding colour the sample was
    actually given.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [
            {
                "name": "CropAndPad",
                "params": {"px": 20, "fill": (0, 255), "p": 1.0},
            }
        ],
        seed=3,
    )

    engine.apply(_make_sample())

    params = engine.applied_augmentations["CropAndPad"]
    assert 0 <= params["fill"] <= 255  # type: ignore[operator]
    # Configured as a constant, and so still nothing to report.
    assert "fill_mask" not in params


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
    params = _normalize_params(
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


def test_pixel_transforms_follow_the_engine_seed():
    """Pixel transforms are composed and replayed apart from the rest.

    Their composition was built without the seed, and rebuilt on every
    call, so a seeded loader could not reproduce a run's pixel
    augmentations.
    """

    def sample_parameters() -> list[Params]:
        engine = AlbumentationsEngine(
            64,
            64,
            {},
            {},
            ["image"],
            [{"name": "RandomBrightnessContrast", "params": {"p": 1.0}}],
            seed=42,
        )
        parameters = []
        for _ in range(3):
            engine.apply(_make_sample())
            parameters.append(
                engine.applied_augmentations["RandomBrightnessContrast"]
            )
        return parameters

    parameters = sample_parameters()

    assert parameters == sample_parameters()
    # Reproducible across runs, but not the same sample after sample.
    assert parameters[0] != parameters[1] != parameters[2]


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


def test_tracks_leaf_transforms_exposing_a_transforms_attribute():
    """`A.ColorJitter` is a leaf that owns a ``transforms`` list.

    Treating every object with that attribute as a composition descended
    into its internal adjustment functions and dropped the transformation
    from the report entirely.
    """
    assert not isinstance(A.ColorJitter(), A.BaseCompose)
    assert hasattr(A.ColorJitter(), "transforms")

    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "ColorJitter", "params": {"p": 1.0}}],
        seed=1,
    )

    engine.apply(_make_sample())

    assert list(engine.applied_augmentations) == ["ColorJitter"]


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

    with _registered(AliasedBlur, "AliasedName"):
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

    # Mosaic4 applied to the first sub-batch only, so inputs 0-3 merged
    # into one image and input 4 passed through; MixUp then merged the two.
    assert engine.batch_augmentation_indices == [0, 1, 2, 3, 4]
    assert set(engine.applied_augmentations) == {"Mosaic4", "MixUp"}


def test_tracks_cutmix_runtime_parameters():
    """`CutMix` samples its mixing coefficient and its patch bounds.

    The patch bounds are derived from the coefficient and the first image
    shape, so they are runtime values rather than configuration echoed
    back, and describe what the augmentation did to the sample.
    """
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "CutMix", "params": {"p": 1.0, "alpha": 1.0}}],
        seed=42,
    )

    engine.apply(_make_sample() * 2)

    params = engine.applied_augmentations["CutMix"]
    assert set(params) == {"lambda_value", "x1", "y1", "x2", "y2"}
    assert 0.0 <= params["lambda_value"] <= 1.0  # type: ignore[operator]
    assert 0 <= params["x1"] <= params["x2"] <= 64  # type: ignore[operator]
    assert 0 <= params["y1"] <= params["y2"] <= 64  # type: ignore[operator]
    # The patch is what the loader reports, so it has to survive the trip
    # through the sample metadata unchanged.
    assert json.loads(json.dumps(params)) == params


def test_cutmix_reports_both_merged_inputs():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "CutMix", "params": {"p": 1.0}}],
        seed=42,
    )

    engine.apply(_make_sample() * 2)

    assert engine.batch_augmentation_indices == [0, 1]


def test_omits_cutmix_that_did_not_apply():
    engine = AlbumentationsEngine(
        64,
        64,
        {},
        {},
        ["image"],
        [{"name": "CutMix", "params": {"p": 0.0}}],
        seed=42,
    )

    engine.apply(_make_sample() * 2)

    assert engine.applied_augmentations == {}
    assert engine.batch_augmentation_indices == [0]


def test_tracks_cutmix_alongside_its_labels():
    """Provenance must survive the label bookkeeping `CutMix` performs.

    `CutMix` rewrites bounding boxes and keypoints as it patches, which is
    the part of the composition that drops and reindexes rows. Tracking
    reads the transformation itself, so it has to stay unaffected.
    """
    targets = {
        "task/boundingbox": "boundingbox",
        "task/keypoints": "keypoints",
    }
    engine = AlbumentationsEngine(
        320,
        320,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        [{"name": "CutMix", "params": {"p": 1.0}}],
        seed=3,
    )
    image = np.zeros((320, 320, 3), dtype=np.uint8)

    def labels(offset: float) -> dict[str, np.ndarray]:
        return {
            "task/boundingbox": np.array(
                [[0.0, 0.1 + offset, 0.1 + offset, 0.2, 0.2]]
            ),
            "task/keypoints": np.array([[0.15 + offset, 0.15 + offset, 2.0]]),
        }

    engine.apply(
        [({"image": image}, labels(0.0)), ({"image": image}, labels(0.5))]
    )

    params = engine.applied_augmentations["CutMix"]
    assert set(params) == {"lambda_value", "x1", "y1", "x2", "y2"}
    assert engine.batch_augmentation_indices == [0, 1]
