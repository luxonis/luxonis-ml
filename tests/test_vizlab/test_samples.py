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
from luxonis_ml.vizlab import (
    ArrayField,
    ColorBar,
    Image,
    Renderable,
    RenderOptions,
    SampleComposer,
)
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
) -> tuple[list[Renderable], list[str]]:
    composer = SampleComposer(RenderOptions(array_view=array_view))
    tiles, titles = composer.tiles(
        {"image": PHOTO},
        {"depth": DEPTH[None, None]},
        record,
        LayerState(),
        "class",
    )
    return tiles, titles  # type: ignore[return-value]


def _has_field(tile: Image) -> bool:
    return any(isinstance(a, ArrayField) for a in tile.annotations)


def test_an_array_task_does_not_split_the_sample_into_task_tiles() -> None:
    tiles, _ = _tiles(
        _record(objects=[_box("car")], depth=[_depth()]), "overlay"
    )

    assert len(tiles) == 1
    assert isinstance(tiles[0], Image)
    assert _has_field(tiles[0])


def test_overlay_paints_every_task_tile_of_a_multi_task_sample() -> None:
    record = _record(
        objects=[_box("car")], people=[_box("person")], depth=[_depth()]
    )

    tiles, titles = _tiles(record, "overlay")

    assert titles == ["image · objects", "image · people"]
    assert all(isinstance(tile, Image) and _has_field(tile) for tile in tiles)


def test_tile_view_adds_no_bare_tile_for_the_array_task() -> None:
    record = _record(
        objects=[_box("car")], people=[_box("person")], depth=[_depth()]
    )

    _, titles = _tiles(record, "tile")

    assert titles == ["image · objects", "image · people", "depth · scalar"]


@pytest.mark.parametrize(
    ("stable_grid", "titles"),
    [
        (False, ["image · objects", "image · people"]),
        (
            True,
            ["image · objects", "image · people", "depth · not in this frame"],
        ),
    ],
)
def test_a_sample_without_its_array_keeps_its_slot_only_in_a_stable_grid(
    stable_grid: bool, titles: list[str]
) -> None:
    # The loader lists every task of the dataset. A sample that lacks the
    # array gets an empty detection list and an empty array label. It never
    # gets a bare task tile; a stable grid keeps a slot for the array, so
    # the grid of a sequence does not reflow from frame to frame.
    record = _record(objects=[_box("car")], people=[_box("person")], depth=[])
    composer = SampleComposer(
        RenderOptions(array_view="tile"), stable_grid=stable_grid
    )

    tiles, got = composer.tiles(
        {"image": PHOTO},
        {"depth": np.zeros(0)},
        record,
        LayerState(),
        "class",
    )

    assert got == titles
    if stable_grid:
        assert (tiles[-1].width, tiles[-1].height) == (60, 40)  # photo size


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
        {"image": PHOTO},
        {},
        record,
        {},
        LayerState(classes=("car",)),
        color_by,
    )

    assert color_by == "class"
    assert "classes" in sidebars[0]


@pytest.mark.parametrize(
    ("shape", "size"),
    [((10, 15), (60, 40)), ((20, 20), (40, 40))],
)
def test_an_array_tile_is_as_tall_as_the_photo(
    shape: tuple[int, int], size: tuple[int, int]
) -> None:
    # Arrays are often stored at a fraction of the image size. A tile at the
    # stored size would be dwarfed by the photo beside it.
    field = np.linspace(0.0, 1.0, shape[0] * shape[1]).reshape(shape)
    composer = SampleComposer(
        RenderOptions(array_view="tile", array_colorbar=False)
    )

    tiles, _ = composer.tiles(
        {"image": PHOTO},
        {"depth": field[None, None]},
        _record(objects=[_box("car")], depth=[_depth()]),
        LayerState(),
        "class",
    )

    assert (tiles[-1].width, tiles[-1].height) == size


def test_array_keys_go_in_strips_of_one_height_under_every_tile() -> None:
    # A key over the field covers its data. Under it, every tile gets the
    # same strip, so the rows stay aligned.
    composer = SampleComposer(RenderOptions(array_view="tile"))

    tiles, _ = composer.tiles(
        {"image": PHOTO},
        {"depth": DEPTH[None, None]},
        _record(objects=[_box("car")], depth=[_depth()]),
        LayerState(),
        "class",
    )

    photo, field = tiles
    assert photo.height == field.height > PHOTO.shape[0]
    assert not any(isinstance(tile, Image) for tile in tiles)


@pytest.mark.parametrize(
    ("overlay_source", "labeled"), [(None, "left"), ("right", "right")]
)
def test_labels_go_on_the_reference_source_only(
    overlay_source: str | None, labeled: str
) -> None:
    # One record labels a stereo pair in the coordinates of one camera.
    composer = SampleComposer(
        RenderOptions(array_overlay_source=overlay_source)
    )

    tiles, titles = composer.tiles(
        {"right": PHOTO, "left": PHOTO},
        {},
        _record(objects=[_box("car")]),
        LayerState(),
        "class",
    )

    assert titles == ["left", "right"]
    drawn = {
        title: bool(tile.annotations)  # type: ignore[attr-defined]
        for title, tile in zip(titles, tiles, strict=True)
    }
    assert drawn == {"left": labeled == "left", "right": labeled == "right"}


def _depth_tile(field: np.ndarray) -> tuple[Image, str | None]:
    """Build the depth field tile and its key title, without the key strip."""
    composer = SampleComposer(RenderOptions(array_view="tile"))
    ((tile, key),), _ = composer._array_tiles(
        {"depth": field[None, None]}, LayerState(), image_shape=(400, 600)
    )
    assert isinstance(key, ColorBar)
    return tile, key.title


def test_an_array_with_gaps_shows_a_hatch_and_says_so_in_its_key() -> None:
    # The top half has no data; on the plain background it read as no tile.
    field = np.linspace(1.0, 50.0, 400 * 600).reshape(400, 600)
    field[:200] = np.nan
    tile, title = _depth_tile(field)

    assert title == "depth \u00b7 log  \u25a8 no data"
    rendered = tile.render()[..., :3]
    gap, data = rendered[2:190, 2:-2], rendered[210:300, 300:-2]
    # Two shades in the gap, the background and the stripe; the data draws
    # over the hatch.
    shades = np.unique(gap.reshape(-1, 3), axis=0)
    assert len(shades) == 2
    assert not (data[..., None, :] == shades).all(axis=-1).any()


def test_an_array_without_gaps_keeps_its_plain_key() -> None:
    field = np.linspace(1.0, 50.0, 400 * 600).reshape(400, 600)
    assert _depth_tile(field)[1] == "depth \u00b7 log"


@pytest.mark.parametrize("panel", [True, False])
def test_hover_less_metadata_goes_to_the_panel_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch, panel: bool
) -> None:
    from luxonis_ml.vizlab import InfoCard

    scene = Detection(class_name=None, metadata={"light": "day"})
    record = _record(objects=[_box("car")], scene=[scene])
    composer = SampleComposer(RenderOptions(), panel=panel)
    sidebars: list[dict[str, object]] = []
    sidebar = SampleComposer.sidebar

    def spy(self: SampleComposer, *args: object, **kwargs: object) -> object:
        sidebars.append(sidebar(self, *args, **kwargs))  # type: ignore[arg-type]
        return sidebars[-1]

    monkeypatch.setattr(SampleComposer, "sidebar", spy)
    render = composer.render(
        {"image": PHOTO}, {}, record, {}, LayerState(), "class"
    )
    tiles, _ = composer.tiles(
        {"image": PHOTO}, {}, record, LayerState(), "class"
    )

    cards = [
        a
        for tile in tiles
        if isinstance(tile, Image)
        for a in tile.annotations
        if isinstance(a, InfoCard)
    ]
    assert render is not None
    assert bool(cards) is not panel
    if panel:
        assert sidebars[0]["object metadata"] == {"light": "day"}


def test_the_legend_lists_classes_in_the_dataset_order() -> None:
    composer = SampleComposer(
        RenderOptions(), class_order=("person", "car", "bus")
    )
    sidebar = composer.sidebar(
        {}, LayerState(classes=("bus", "tree", "car")), color_by="class"
    )
    legend = sidebar["classes"]
    assert [name for _, name in legend.items] == ["car", "bus", "tree"]  # type: ignore[attr-defined]
