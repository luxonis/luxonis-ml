"""Tests for interactive layer toggles (`LayerState` and `apply_layers`)."""

import numpy as np

from luxonis_ml.vizlab import BBox, Classification, Color, Keypoints, Mask
from luxonis_ml.vizlab.annotations import Annotation
from luxonis_ml.vizlab.style import Palette, derive_child_color
from luxonis_ml.vizlab.viewer import LayerState
from luxonis_ml.vizlab.viewer.layers import _resolved_color

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


def test_hidden_class_is_dropped_but_others_stay() -> None:
    car = BBox(x=0.0, y=0.0, w=0.4, h=0.4, label="car")
    person = BBox(x=0.5, y=0.5, w=0.4, h=0.4, label="person")
    out = LayerState(hidden={"person"}).apply_layers([car, person], PALETTE)
    assert [b.label for b in out] == ["car"]


def test_hidden_keeps_unlabeled_scene_annotations() -> None:
    tag = Classification(tags=["scene"])  # no class of its own
    out = LayerState(hidden={"car"}).apply_layers([tag], PALETTE)
    assert len(out) == 1


def test_toggle_class_flips_visibility() -> None:
    state = LayerState(classes=("car", "person"))
    state.toggle_class("car")
    assert state.hidden == {"car"}
    state.toggle_class("car")
    assert state.hidden == set()


def test_toggle_all_classes_hides_then_shows_everything() -> None:
    state = LayerState(classes=("car", "person", "dog"))
    state.toggle_all_classes()  # all shown -> hide the whole set
    assert state.hidden == {"car", "person", "dog"}
    state.toggle_all_classes()  # any hidden -> show all again
    assert state.hidden == set()


def test_toggle_all_classes_shows_all_from_a_partial_selection() -> None:
    state = LayerState(classes=("car", "person"))
    state.toggle_class("car")  # one already hidden
    state.toggle_all_classes()  # a partial selection clears to all shown
    assert state.hidden == set()
    state.handle("c")  # the isolate cursor was reset, so c restarts cleanly
    assert state.hidden == {"person"}


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


def _labels(annotations: list[Annotation]) -> list[str | None]:
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


def test_update_classes_keeps_hidden_and_drops_a_stale_cursor() -> None:
    state = LayerState(classes=("car", "person"))
    state.toggle_class("car")  # hide car (persists across samples)
    state.update_classes(["car", "person", "dog"])
    assert state.classes == ("car", "person", "dog")
    assert state.hidden == {"car"}  # the hidden set persists


def test_update_classes_resets_an_out_of_range_focus() -> None:
    state = LayerState(classes=("car", "person", "dog"), _focus=2)
    state.update_classes(["car"])
    assert state._focus is None


def test_handle_c_cycles_isolate_then_wraps() -> None:
    state = LayerState(classes=("car", "person"))
    state.handle("c")  # isolate car -> hide person
    assert state.hidden == {"person"}
    state.handle("c")  # isolate person -> hide car
    assert state.hidden == {"car"}
    state.handle("c")  # wrap back to all shown
    assert state.hidden == set()


def test_handle_c_resets_after_manual_toggles() -> None:
    state = LayerState(classes=("car", "person", "dog"))
    state.toggle_class("car")
    state.toggle_class("person")
    assert state.hidden == {"car", "person"}
    state.handle(
        "c"
    )  # a manual set -> the first c resets to "all", restarting
    assert state.hidden == set()
    state.handle("c")  # then the cycle isolates the first class
    assert state.hidden == {"person", "dog"}


def test_handle_c_is_a_noop_without_classes() -> None:
    state = LayerState()
    assert state.handle("c")  # still a recognized control key
    assert state.hidden == set()


def test_controls_report_current_state() -> None:
    controls = {
        c.name: c for c in LayerState(masks=False, boxes=False).controls()
    }
    assert (controls["masks"].value, controls["masks"].active) == (
        "off",
        False,
    )
    assert controls["boxes"].value == "off"
    assert controls["declutter"].value == "on"  # on by default
    # No fill control any more.
    assert "fill" not in controls


def test_controls_class_reflects_visibility() -> None:
    # All shown -> neutral "all".
    allc = {c.name: c for c in LayerState(classes=("car",)).controls()}[
        "class"
    ]
    assert (allc.value, allc.active) == ("all", None)
    # A manual toggle -> "N off".
    state = LayerState(classes=("car", "person"))
    state.toggle_class("car")
    off = {c.name: c for c in state.controls()}["class"]
    assert (off.value, off.active) == ("1 off", True)
    # An isolate cycle counts the hidden classes too (it never names the kept
    # one — the legend already shows that, and a name would jump the width).
    state.handle("c")  # resets, then...
    state.handle("c")  # isolate car -> hide the other
    iso = {c.name: c for c in state.controls()}["class"]
    assert (iso.value, iso.active) == ("1 off", True)


def test_declutter_returns_original_when_busy_scene_has_no_tiny_boxes() -> (
    None
):
    scene = [
        BBox(
            x=0.1 + 0.01 * (index % 5),
            y=0.1 + 0.01 * (index // 5),
            w=0.1,
            h=0.1,
            label="car",
        )
        for index in range(25)
    ]
    out = LayerState().apply_layers(scene, PALETTE)
    assert all(
        actual is expected for actual, expected in zip(out, scene, strict=True)
    )


def test_explicit_annotation_color_is_preserved_when_boxes_are_hidden() -> (
    None
):
    box = _detection()
    box.color = "#123456"
    promoted = LayerState(boxes=False).apply_layers([box], PALETTE)
    assert promoted
    assert all(
        child.color == derive_child_color(Color.parse("#123456"))
        for child in promoted
    )


def test_unlabeled_annotations_keep_one_stable_palette_slot() -> None:
    """Rebuilding an unlabeled annotation must not change its color.

    `luxonis_ml data inspect` rebuilds its annotations on every navigation, and
    keypoints-only tasks have no class name to color by. Keying the palette on
    object identity gave each rebuild a fresh color and grew the shared palette
    without bound — which also shifts every later class's color, since a palette
    assigns colors in order of first use.
    """
    palette = Palette()
    colors = {
        _resolved_color(Keypoints(keypoints=[(0.5, 0.5, 2)]), palette, None)
        for _ in range(5)
    }
    assert len(colors) == 1
    assert len(palette) == 1


def test_unlabeled_palette_key_is_per_type_not_per_instance() -> None:
    first, second = (
        Keypoints(keypoints=[(0.5, 0.5, 2)]),
        Keypoints(keypoints=[(0.1, 0.1, 2)]),
    )
    assert first.unlabeled_color_key() == second.unlabeled_color_key()
    # Distinct from any class name, so it cannot collide with a real label.
    assert first.unlabeled_color_key() != type(first).__name__
    assert (
        Mask(mask=np.ones((4, 4), np.uint8)).unlabeled_color_key()  # type: ignore[arg-type]
        != first.unlabeled_color_key()
    )


# --- array fields -----------------------------------------------------------


def test_hiding_arrays_drops_the_field_and_its_key() -> None:
    from luxonis_ml.vizlab import ColorBar, Heatmap

    annotations = [Heatmap(values=np.ones((4, 4))), ColorBar()]
    state = LayerState()
    assert len(state.apply_layers(annotations, PALETTE)) == 2
    state.arrays = False
    assert state.apply_layers(annotations, PALETTE) == []


def test_a_key_toggles_arrays_even_without_any() -> None:
    # Consumed unconditionally, so the key never changes meaning per sample.
    state = LayerState()
    assert state.handle("a") is True
    assert state.arrays is False
    assert state.handle("A") is True
    assert state.arrays is True


def test_array_control_row_appears_only_when_the_sample_has_fields() -> None:
    assert "a" not in {c.key for c in LayerState().controls()}
    keyed = {c.key: c for c in LayerState(has_arrays=True).controls()}
    assert keyed["a"].name == "arrays"
    assert keyed["a"].value == "on"


def test_copy_carries_array_state() -> None:
    # copy() enumerates fields by hand; a missed one only shows up under
    # --prefetch, as a snapshot that silently disagrees with the live state.
    state = LayerState(arrays=False, has_arrays=True)
    clone = state.copy()
    assert (clone.arrays, clone.has_arrays) == (False, True)
    assert clone == state


def test_is_default_is_false_when_arrays_are_hidden() -> None:
    assert LayerState().is_default()
    assert not LayerState(arrays=False).is_default()
