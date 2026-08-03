"""Tests for `RenderOptions.hover_metadata` attaching tooltips in convert."""

from luxonis_ml.ldf import Detection
from luxonis_ml.vizlab import (
    DARK_THEME,
    BBox,
    Palette,
    RenderOptions,
)
from luxonis_ml.vizlab.adapters.ldf import detection_to_annotations


def _pinned(
    palette: Palette, *, hover_metadata: bool = False
) -> RenderOptions:
    """Render options whose theme pins the given palette."""
    return RenderOptions(
        theme=DARK_THEME.with_palette(palette), hover_metadata=hover_metadata
    )


def _first_box(det: Detection, options: RenderOptions) -> BBox:
    annotations = detection_to_annotations(det, options)
    boxes = [a for a in annotations if isinstance(a, BBox)]
    assert boxes, "expected a BBox annotation"
    return boxes[0]


def _car(**metadata: object) -> Detection:
    data: dict[str, object] = {
        "class_name": "car",
        "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        "instance_id": 3,
    }
    if metadata:
        data["metadata"] = metadata
    return Detection.model_validate(data)


def test_hover_metadata_attaches_tooltip() -> None:
    palette = Palette(["car"])
    options = _pinned(palette, hover_metadata=True)
    box = _first_box(_car(text="AB123", track_id=7), options)
    tip = box.tooltip
    assert tip is not None
    assert tip.title == "car #3"  # class name + instance id
    # Every metadata entry becomes a row, "text" included like any other.
    assert tip.rows == (("text", "AB123"), ("track_id", "7"))
    assert tip.tint == palette.color_for("car")


def test_hover_metadata_off_by_default() -> None:
    box = _first_box(_car(track_id=7), _pinned(Palette(["car"])))
    assert box.tooltip is None


def test_text_metadata_is_just_another_row() -> None:
    # "text" is no longer special: it hovers like the rest of the metadata.
    box = _first_box(_car(text="AB123"), RenderOptions(hover_metadata=True))
    assert box.tooltip is not None
    assert box.tooltip.rows == (("text", "AB123"),)


def test_no_tooltip_without_metadata() -> None:
    box = _first_box(_car(), RenderOptions(hover_metadata=True))
    assert box.tooltip is None


def test_tooltip_data_renders_as_the_panels_json_tree() -> None:
    """JSON-like `Tooltip.data` flattens the way the metadata panel draws it."""
    from luxonis_ml.vizlab.layout.panel import format_tree
    from luxonis_ml.vizlab.tooltip import Tooltip

    data = {"scale": [0.9, 1.1], "rotate": -12.345678, "fit_output": False}
    tooltip = Tooltip(title="Affine", data=data)
    assert tooltip.resolved_rows == (
        ("scale:", ""),
        ("  • ", "0.9"),
        ("  • ", "1.1"),
        ("rotate: ", "-12.35"),  # the panel's float spelling, not the raw repr
        ("fit_output: ", "false"),  # ...and its JSON spelling for booleans
    )
    # It is the panel's own formatter, indented by depth -- not a second one.
    assert [line[3] for line in format_tree(data)] == [
        row[1] for row in tooltip.resolved_rows
    ]


def test_tooltip_data_appends_to_explicit_rows() -> None:
    from luxonis_ml.vizlab.tooltip import Tooltip

    tooltip = Tooltip(rows=(("id", "7"),), data={"p": 0.5})
    assert tooltip.resolved_rows == (("id", "7"), ("p: ", "0.5"))
    assert not tooltip.is_empty


def test_tooltip_with_only_data_is_not_empty_and_renders() -> None:
    from luxonis_ml.vizlab.tooltip import Tooltip
    from luxonis_ml.vizlab.viewer.tooltip_render import render_tooltip_card

    tooltip = Tooltip(title="Affine", data={"p": 0.5})
    assert not tooltip.is_empty
    # The card draws without a single explicit row.
    assert render_tooltip_card(tooltip, 16).shape[2] == 4
