from collections.abc import Callable, Iterator

import albumentations as A
import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.augmentations import AugmentationEngine
from luxonis_ml.data.augmentations.custom import TRANSFORMATIONS
from luxonis_ml.typing import Labels, LoaderMultiOutput

Register = Callable[[type], type]


@pytest.fixture
def register_transform() -> Iterator[Register]:
    """Register transforms for one test and drop them again afterwards."""
    registered: list[str] = []

    def _register(cls: type) -> type:
        name = cls.__name__
        TRANSFORMATIONS[name] = cls  # type: ignore[assignment]
        registered.append(name)
        return cls

    yield _register

    for name in registered:
        # `Registry` has no public way to drop an entry again.
        TRANSFORMATIONS._module_dict.pop(name, None)


def build(
    targets: dict[str, str], config: list[dict], **kwargs
) -> AlbumentationsEngine:
    return AlbumentationsEngine(
        32,
        32,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        config,
        **kwargs,
    )


def test_engines_default_to_using_every_input_position() -> None:
    """An engine that does not track contributors is assumed to use all."""

    class UntrackedEngine(
        AugmentationEngine, register_name="untracked_engine"
    ):
        def __init__(self, *_, **__) -> None: ...

        def apply(
            self, input_batch: list[LoaderMultiOutput]
        ) -> LoaderMultiOutput:
            return input_batch[0]

        @property
        def batch_size(self) -> int:
            return 4

    assert UntrackedEngine().batch_augmentation_indices == [0, 1, 2, 3]


def test_unsupported_task_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported task type: 'depth'"):
        build({"task/depth": "depth"}, [])


def test_only_one_resizing_augmentation_is_allowed() -> None:
    config = [
        {
            "name": "Resize",
            "params": {"height": 32, "width": 32},
            "use_for_resizing": True,
        },
        {
            "name": "LetterboxResize",
            "params": {"height": 32, "width": 32},
            "use_for_resizing": True,
        },
    ]

    with pytest.raises(
        ValueError, match="Only one resizing augmentation can be provided"
    ):
        build({"task/boundingbox": "boundingbox"}, config)


@pytest.mark.parametrize("probability", ["1.0", True])
def test_resizing_probability_must_be_numeric(probability: object) -> None:
    """``p=None`` never reaches this check.

    Albumentations rejects it while the transform is being constructed.
    """
    config = [
        {
            "name": "Resize",
            "params": {"height": 32, "width": 32, "p": probability},
            "use_for_resizing": True,
        }
    ]

    with pytest.raises(TypeError, match="has invalid p="):
        build({"task/boundingbox": "boundingbox"}, config)


def test_certain_resize_is_used_directly() -> None:
    """``p >= 1`` needs no ``OneOf`` wrapper around the default resize."""
    config = [
        {
            "name": "Resize",
            "params": {"height": 32, "width": 32, "p": 1.0},
            "use_for_resizing": True,
        }
    ]

    engine = build({"task/boundingbox": "boundingbox"}, config)

    _, out_labels = engine.apply(
        [
            (
                {"image": np.zeros((64, 64, 3), dtype=np.uint8)},
                {"task/boundingbox": np.array([[0.0, 0.1, 0.1, 0.2, 0.2]])},
            )
        ]
    )
    assert out_labels["task/boundingbox"].shape == (1, 5)


def test_plain_basic_transforms_run_as_custom_transforms(
    register_transform: Register,
) -> None:
    """A bare ``BasicTransform`` is neither pixel, spatial, nor batch."""
    seen: list[int] = []

    @register_transform
    class CountingBasicTransform(A.BasicTransform):
        @property
        def targets(self) -> dict:
            return {"image": self.apply}

        def apply(self, img: np.ndarray, **_) -> np.ndarray:
            seen.append(1)
            return img

    engine = build(
        {"task/boundingbox": "boundingbox"},
        [{"name": CountingBasicTransform.__name__, "params": {"p": 1.0}}],
    )
    engine.apply(
        [
            (
                {"image": np.zeros((32, 32, 3), dtype=np.uint8)},
                {"task/boundingbox": np.array([[0.0, 0.1, 0.1, 0.2, 0.2]])},
            )
        ]
    )

    assert seen, "the custom transform stage never ran"


def test_non_transform_classes_are_rejected(
    register_transform: Register,
) -> None:
    @register_transform
    class NotATransform:
        def __init__(self, **_): ...

    with pytest.raises(
        ValueError, match="Unsupported transformation type: 'NotATransform'"
    ):
        build(
            {"task/boundingbox": "boundingbox"},
            [{"name": NotATransform.__name__, "params": {}}],
        )


def test_nested_transform_entries_must_name_a_transform() -> None:
    config = [
        {
            "name": "OneOf",
            "params": {"transforms": [{"params": {"p": 1.0}}], "p": 1.0},
        }
    ]

    with pytest.raises(
        ValueError, match="Invalid nested transform configuration"
    ):
        build({"task/boundingbox": "boundingbox"}, config)


def test_each_unnamed_task_is_its_own_group() -> None:
    """A task with no name is not in a group with every other such task.

    Collapsing them all to ``""`` would tie an unnamed keypoints task to an
    unnamed bbox task it has nothing to do with.
    """
    assert AlbumentationsEngine._get_task_group("boundingbox") == "boundingbox"
    assert AlbumentationsEngine._get_task_group("keypoints") == "keypoints"
    assert AlbumentationsEngine._get_task_group("metadata/id") == "metadata/id"
    assert AlbumentationsEngine._get_task_group("group/boundingbox") == "group"
    assert AlbumentationsEngine._get_task_group("a/b/keypoints") == "a/b"


def test_unnamed_tasks_are_usable_end_to_end() -> None:
    engine = build({"classification": "classification"}, [])

    _, out_labels = engine.apply(
        [
            (
                {"image": np.zeros((32, 32, 3), dtype=np.uint8)},
                {"classification": np.array([1.0])},
            )
        ]
    )

    assert out_labels["classification"].tolist() == [1.0]


def test_pixel_transforms_replay_across_every_image_source() -> None:
    sources = ["rgb", "depth", "ir"]
    engine = AlbumentationsEngine(
        32,
        32,
        {"task/boundingbox": "boundingbox"},
        {"task/boundingbox": 1},
        sources,
        [{"name": "InvertImg", "params": {"p": 1.0}}],
    )
    images = {name: np.zeros((32, 32, 3), dtype=np.uint8) for name in sources}
    labels: Labels = {
        "task/boundingbox": np.array([[0.0, 0.1, 0.1, 0.2, 0.2]])
    }

    out_images, _ = engine.apply([(images, labels)])

    assert set(out_images) == set(sources)
    for name in sources:
        assert np.all(out_images[name] == 255), (
            f"source '{name}' did not receive the pixel transform"
        )


def test_pixel_replay_skips_sources_without_a_channel_axis() -> None:
    engine = AlbumentationsEngine(
        16,
        16,
        {"task/boundingbox": "boundingbox"},
        {"task/boundingbox": 1},
        ["rgb", "flat"],
        [{"name": "InvertImg", "params": {"p": 1.0}}],
    )
    images = {
        "rgb": np.zeros((16, 16, 3), dtype=np.uint8),
        "flat": np.zeros((16, 16), dtype=np.uint8),
    }

    out_images, _ = engine.apply(
        [
            (
                images,
                {"task/boundingbox": np.array([[0.0, 0.1, 0.1, 0.2, 0.2]])},
            )
        ]
    )

    assert np.all(out_images["rgb"] == 255)
    assert np.all(out_images["flat"] == 0), (
        "a source with no channel axis must be left alone, not corrupted"
    )


def test_wrapped_transforms_tolerate_a_missing_image_key() -> None:
    wrapped = AlbumentationsEngine._wrap_transform(
        A.Compose([A.HorizontalFlip(p=1.0)])
    )

    out = wrapped(image=np.zeros((8, 8, 3), dtype=np.uint8))

    assert "_original_image_key" not in out


def test_pixel_transform_wrapper_needs_source_names() -> None:
    wrapped = AlbumentationsEngine._wrap_transform(
        A.Compose([A.InvertImg(p=1.0)]), is_pixel=True, source_names=None
    )

    with pytest.raises(ValueError, match="`source_names` must be provided"):
        wrapped(image=np.zeros((8, 8, 3), dtype=np.uint8))


def test_grayscale_sources_keep_a_channel_axis() -> None:
    engine = AlbumentationsEngine(
        16,
        16,
        {"task/boundingbox": "boundingbox"},
        {"task/boundingbox": 1},
        ["image"],
        [{"name": "ToGray", "params": {"p": 1.0, "num_output_channels": 1}}],
    )

    out_images, _ = engine.apply(
        [
            (
                {"image": np.zeros((16, 16, 3), dtype=np.uint8)},
                {"task/boundingbox": np.array([[0.0, 0.1, 0.1, 0.2, 0.2]])},
            )
        ]
    )

    assert out_images["image"].ndim == 3
    assert out_images["image"].shape[-1] == 1
