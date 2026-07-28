"""Tests for interactive layer toggles (`LayerState` and `apply_layers`)."""

import numpy as np

from luxonis_ml.vizlab import BBox, Classification, Keypoints, Mask
from luxonis_ml.vizlab.annotations import Annotation
from luxonis_ml.vizlab.style import Palette, derive_child_color
from luxonis_ml.vizlab.viewer import LayerState

PALETTE = Palette(["car", "person"])


def _detection(label: str = "car") -> BBox:
    """Return a box carrying one keypoint child and one mask child."""
    box = BBox(x=0.1, y=0.1, w=0.5, h=0.5, label=label, score=0.9)
    box.add(Keypoints(keypoints=[(0.2, 0.2, 2)]))  # type: ignore[list-item]
    box.add(Mask(mask=np.ones((8, 8), np.uint8)))  # type: ignore[arg-type]
    return box


def _child_kinds(box: Annotation) -> list[str]:
    return [type(child).__name__ for child in box.children]


def test_default_state_passes_annotations_through_untouched() -> None:
    state = LayerState()
    anns = [_detection()]
    assert state.is_default()
    out = state.apply_layers(anns, PALETTE)
    assert out[0] is anns[0]  # not copied when nothing is toggled


def test_hiding_masks_drops_mask_children_only() -> None:
    (box,) = LayerState(masks=False).apply_layers([_detection()], PALETTE)
    kinds = _child_kinds(box)
    assert "Mask" not in kinds
    assert "Keypoints" in kinds


def test_hiding_keypoints_drops_keypoint_children_only() -> None:
    (box,) = LayerState(keypoints=False).apply_layers([_detection()], PALETTE)
    kinds = _child_kinds(box)
    assert "Keypoints" not in kinds
    assert "Mask" in kinds


def test_top_level_mask_is_removed_when_masks_off() -> None:
    mask = Mask(mask=np.ones((8, 8), np.uint8))  # type: ignore[arg-type]
    assert LayerState(masks=False).apply_layers([mask], PALETTE) == []


def test_hiding_boxes_promotes_children_and_drops_the_rectangle() -> None:
    out = LayerState(boxes=False).apply_layers([_detection()], PALETTE)
    kinds = [type(a).__name__ for a in out]
    assert "BBox" not in kinds  # the rectangle is gone
    assert kinds == ["Keypoints", "Mask"]  # its content stays, in place


def test_hiding_boxes_bakes_the_box_color_onto_promoted_children() -> None:
    (keypoints, _mask) = LayerState(boxes=False).apply_layers(
        [_detection()], PALETTE
    )
    # The keypoints derived their color from the box; now detached, that color is
    # baked on so they look exactly as they did nested.
    assert keypoints.color == derive_child_color(PALETTE.color_for("car"))


def test_hiding_boxes_removes_a_box_with_nothing_inside() -> None:
    lone = BBox(x=0.1, y=0.1, w=0.4, h=0.4, label="car")
    assert LayerState(boxes=False).apply_layers([lone], PALETTE) == []


def test_hiding_boxes_keeps_a_top_level_mask() -> None:
    mask = Mask(mask=np.ones((8, 8), np.uint8))  # type: ignore[arg-type]
    out = LayerState(boxes=False).apply_layers([mask], PALETTE)
    assert [type(a).__name__ for a in out] == ["Mask"]


def test_labels_off_strips_chip_text_but_bakes_the_color() -> None:
    (box,) = LayerState(labels=False).apply_layers([_detection()], PALETTE)
    assert box.label is None
    assert box.score is None
    assert box.color == PALETTE.color_for("car")  # color preserved


def test_labels_off_leaves_the_original_annotation_intact() -> None:
    box = _detection()
    LayerState(labels=False).apply_layers([box], PALETTE)
    assert (box.label, box.score) == ("car", 0.9)  # input untouched


def test_labels_off_empties_classification_tags() -> None:
    (tag,) = LayerState(labels=False).apply_layers(
        [Classification(tags=["indoor"])], PALETTE
    )
    assert isinstance(tag, Classification)
    assert tag.tags == []


def test_focus_isolates_a_single_class() -> None:
    car = BBox(x=0.0, y=0.0, w=0.4, h=0.4, label="car")
    person = BBox(x=0.5, y=0.5, w=0.4, h=0.4, label="person")
    out = LayerState(focus="car").apply_layers([car, person], PALETTE)
    assert [b.label for b in out] == ["car"]


def test_focus_keeps_unlabeled_scene_annotations() -> None:
    tag = Classification(tags=["scene"])  # no class of its own
    out = LayerState(focus="car").apply_layers([tag], PALETTE)
    assert len(out) == 1


def test_fill_alpha_override_layers_onto_shapes() -> None:
    (box,) = LayerState(fill_alpha=0.8).apply_layers([_detection()], PALETTE)
    assert box.style_overrides["fill_alpha"] == 0.8
    assert box.style_overrides["mask_alpha"] == 0.8


def _busy_scene() -> list[BBox]:
    """Build a crowded scene: 20 large boxes, a tiny cluster, one lone speck."""
    large = [
        BBox(
            x=0.4 + 0.01 * (k % 5),
            y=0.4 + 0.01 * (k // 5),
            w=0.08,
            h=0.08,
            label="car",
        )
        for k in range(20)
    ]
    cluster = [
        BBox(x=0.48 + 0.005 * k, y=0.5, w=0.02, h=0.02, label="tiny")
        for k in range(8)
    ]
    lonely = BBox(x=0.02, y=0.02, w=0.02, h=0.02, label="lonely")
    return [*large, *cluster, lonely]


def _labels(annotations: list) -> list[str]:
    return [a.label for a in annotations]


def test_declutter_drops_tiny_boxes_ringed_by_a_crowd() -> None:
    out = LayerState().apply_layers(_busy_scene(), PALETTE)
    labels = _labels(out)
    assert "tiny" not in labels  # the crowded specks are gone
    assert labels.count("car") == 20  # every large box stays
    assert labels.count("lonely") == 1  # the isolated speck stays


def test_declutter_keeps_an_isolated_tiny_box() -> None:
    # The lone speck in a corner has no neighbors, so it survives the busy scene.
    out = LayerState().apply_layers(_busy_scene(), PALETTE)
    assert "lonely" in _labels(out)


def test_declutter_off_keeps_every_detection() -> None:
    scene = _busy_scene()
    out = LayerState(declutter=False).apply_layers(scene, PALETTE)
    assert len(out) == len(scene)
    assert _labels(out).count("tiny") == 8


def test_declutter_leaves_sparse_scenes_untouched() -> None:
    # A tight cluster of tiny boxes, but too few to be a busy scene: all kept.
    cluster = [
        BBox(x=0.48 + 0.005 * k, y=0.5, w=0.02, h=0.02, label="tiny")
        for k in range(6)
    ]
    out = LayerState().apply_layers(cluster, PALETTE)
    assert _labels(out).count("tiny") == 6


def test_handle_toggles_layers_and_reports_control_keys() -> None:
    state = LayerState()
    assert state.handle("m")
    assert state.masks is False
    assert state.handle("k")
    assert state.keypoints is False
    assert state.handle("b")
    assert state.boxes is False
    assert state.handle("l")
    assert state.labels is False
    assert state.handle("d")
    assert state.declutter is False
    assert not state.handle("q")  # not a control key -> caller handles it
    assert not state.handle("x")


def test_update_classes_syncs_the_focus_set_and_drops_a_stale_focus() -> None:
    state = LayerState(focus="car")
    state.update_classes(["car", "person"])
    assert state.classes == ("car", "person")
    assert state.focus == "car"  # still present -> kept

    state.update_classes(["dog"])  # car no longer on screen
    assert state.classes == ("dog",)
    assert state.focus is None  # stale focus reset so the frame is not blank


def test_handle_cycles_class_focus() -> None:
    state = LayerState(classes=("car", "person"))
    assert state.focus is None
    state.handle("c")
    assert state.focus == "car"
    state.handle("c")
    assert state.focus == "person"
    state.handle("c")
    assert state.focus is None  # wraps back to "all"


def test_handle_cycle_is_a_noop_without_classes() -> None:
    state = LayerState()
    assert state.handle("c")  # still a recognized control key
    assert state.focus is None


def test_fill_nudge_moves_and_clamps_to_unit_range() -> None:
    state = LayerState()
    state.handle("]")
    assert state.fill_alpha == 0.4  # starts from 0.3, +0.1
    for _ in range(20):
        state.handle("]")
    assert state.fill_alpha == 1.0
    for _ in range(20):
        state.handle("[")
    assert state.fill_alpha == 0.0


def test_controls_report_current_state() -> None:
    controls = {
        c.name: c for c in LayerState(masks=False, boxes=False).controls()
    }
    assert (controls["masks"].value, controls["masks"].active) == (
        "off",
        False,
    )
    assert (controls["keypoints"].value, controls["keypoints"].active) == (
        "on",
        True,
    )
    assert controls["boxes"].value == "off"
    assert controls["declutter"].value == "on"  # on by default


def test_controls_reflect_class_focus_and_fill() -> None:
    controls = {
        c.name: c for c in LayerState(focus="car", fill_alpha=0.8).controls()
    }
    # A class focus and a fill override read as "active" (highlighted); an
    # unset one is neutral (active is None).
    assert (controls["class"].value, controls["class"].active) == ("car", True)
    assert controls["fill"].value == "0.80"
    assert LayerState().controls()[-2].active is None  # class "all" -> neutral
