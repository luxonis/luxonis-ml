"""Tiles of `SampleComposer`, the per-sample layout of ``data inspect``."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.ldf import (
    ArrayAnnotation,
    BBoxAnnotation,
    DatasetRecord,
    Detection,
)
from luxonis_ml.vizlab import ArrayField, Image, RenderOptions, SampleComposer
from luxonis_ml.vizlab.options import ArrayView
from luxonis_ml.vizlab.viewer import LayerState

PHOTO = np.zeros((40, 60, 3), np.uint8)
DEPTH = np.linspace(0.0, 10.0, 40 * 60).reshape(40, 60)


def _record(**tasks: list[Detection]) -> DatasetRecord:
    return DatasetRecord.model_construct(
        files={"image": Path("frame.jpg")}, annotation=tasks
    )


def _box(class_name: str) -> Detection:
    return Detection(
        class_name=class_name,
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
    )


def _depth() -> Detection:
    """Build an array detection, as `LoaderOutput.to_ldf` makes for an array task."""
    return Detection(
        class_name="depth",
        array=ArrayAnnotation.model_validate({"data": DEPTH}),
    )


def _tiles(
    record: DatasetRecord, array_view: ArrayView
) -> tuple[list[Image], list[str]]:
    composer = SampleComposer(RenderOptions(array_view=array_view))
    tiles, titles = composer.tiles(
        {"image": PHOTO},
        {"depth": DEPTH[None, None]},
        record,
        LayerState(),
        "class",
    )
    assert all(isinstance(tile, Image) for tile in tiles)
    return tiles, titles  # type: ignore[return-value]


def _has_field(tile: Image) -> bool:
    return any(isinstance(a, ArrayField) for a in tile.annotations)


def test_an_array_task_does_not_split_the_sample_into_task_tiles() -> None:
    tiles, _ = _tiles(
        _record(objects=[_box("car")], depth=[_depth()]), "overlay"
    )

    assert len(tiles) == 1
    assert _has_field(tiles[0])


def test_overlay_paints_every_task_tile_of_a_multi_task_sample() -> None:
    record = _record(
        objects=[_box("car")], people=[_box("person")], depth=[_depth()]
    )

    tiles, titles = _tiles(record, "overlay")

    assert titles == ["image · objects", "image · people"]
    assert all(_has_field(tile) for tile in tiles)


def test_tile_view_adds_no_bare_tile_for_the_array_task() -> None:
    record = _record(
        objects=[_box("car")], people=[_box("person")], depth=[_depth()]
    )

    _, titles = _tiles(record, "tile")

    assert titles == ["image · objects", "image · people", "depth scalar"]


@pytest.mark.parametrize(
    "embedding",
    [
        np.ones((1, 2, 128)),  # one instance, class slot 1
        np.stack([np.ones((3, 128)), 2 * np.ones((3, 128))]),  # two, one slot
        np.ones((2, 3)),  # a scalar for each instance
        np.eye(3)[:2, :, None] * np.ones(128),  # two, distinct slots
    ],
)
def test_a_loader_array_that_is_no_picture_gets_no_tile(
    embedding: np.ndarray,
) -> None:
    # The loader wraps every array as (N, n_classes, *shape), so a vector or
    # a scalar arrives with two or three axes and reads like a tiny field.
    composer = SampleComposer(RenderOptions(array_view="tile"))
    _, titles = composer.tiles(
        {"image": PHOTO},
        {"embedding": embedding},
        _record(objects=[_box("car")]),
        LayerState(),
        "class",
    )

    assert titles == []


def test_a_sample_that_falls_back_to_class_colors_gets_the_class_legend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Instance coloring has nothing to color in a classification-only sample,
    # so the sample is drawn in class colors and needs the class legend.
    composer = SampleComposer(RenderOptions(), color_by="instance")
    record = _record(kind=[Detection(class_name="car")])
    color_by = composer.fallback_color_by(record)
    sidebars: list[dict[str, object]] = []
    sidebar = SampleComposer.sidebar

    def spy(self: SampleComposer, *args: object, **kwargs: object) -> object:
        sidebars.append(sidebar(self, *args, **kwargs))  # type: ignore[arg-type]
        return sidebars[-1]

    monkeypatch.setattr(SampleComposer, "sidebar", spy)
    composer.frame(
        {"image": PHOTO}, {}, record, {}, LayerState(classes=("car",)), color_by
    )

    assert color_by == "class"
    assert "classes" in sidebars[0]
