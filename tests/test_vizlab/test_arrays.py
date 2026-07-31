"""Coverage for the array-label bridge: which arrays become which pictures."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from luxonis_ml.vizlab import (
    ArrayImage,
    ColorBar,
    FlowField,
    FlowWheel,
    Image,
    Legend,
    NormalMap,
    RenderOptions,
    ScalarField,
    SegmentationScores,
)
from luxonis_ml.vizlab.adapters.arrays import (
    OVERLAY_ALPHA,
    ArrayPayload,
    array_annotation,
    array_annotations,
    array_field,
    array_payload,
    infer_array_kind,
    is_image_compatible,
    reserved_array_kind,
    resolve_array_kind,
)
from luxonis_ml.vizlab.options import ArrayKinds


def _loader_shaped(
    values: np.ndarray, *, n_classes: int = 3, slot: int = 1
) -> np.ndarray:
    """Wrap a field the way `ArrayAnnotation.combine_to_numpy` stores it."""
    out = np.zeros((1, n_classes, *values.shape), dtype=np.float64)
    out[0, slot] = values
    return out


# --- array_field ------------------------------------------------------------


def test_one_hot_class_axis_recovers_the_populated_slice() -> None:
    field = np.arange(20, dtype=np.float64).reshape(4, 5)
    recovered = array_field(_loader_shaped(field))
    assert recovered is not None
    assert np.array_equal(recovered, field)


def test_negative_values_survive_the_class_collapse() -> None:
    # The reason the class axis is summed rather than maxed: every unselected
    # slot is exactly 0, so a max would floor negative measurements to zero.
    field = np.full((4, 5), -7.0)
    recovered = array_field(_loader_shaped(field))
    assert recovered is not None
    assert float(recovered.min()) == -7.0


def test_leading_singleton_dims_are_squeezed() -> None:
    reduced = array_field(np.ones((1, 1, 6, 8)))
    assert reduced is not None
    assert reduced.shape == (6, 8)


def test_multiple_instances_are_unioned() -> None:
    # Two annotations describing one frame: take the union, not the sum, so
    # overlapping support is not double-counted.
    values = np.zeros((2, 1, 2, 2))
    values[0, 0, 0, 0] = 5.0
    values[1, 0, 1, 1] = 3.0
    recovered = array_field(values)
    assert recovered is not None
    assert np.array_equal(recovered, [[5.0, 0.0], [0.0, 3.0]])


def test_raw_two_dimensional_array_passes_through() -> None:
    # The shape `ArrayAnnotation.to_numpy()` returns directly.
    field = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    recovered = array_field(field)
    assert recovered is not None
    assert np.allclose(recovered, field)


def test_array_field_is_float32() -> None:
    # Guards the render cache, which sha256s every array on each render.
    field = array_field(np.ones((4, 5), dtype=np.float64))
    assert field is not None
    assert field.dtype == np.float32


def test_unrenderable_shapes_are_declined() -> None:
    assert array_field(np.zeros(512)) is None  # an embedding
    assert array_field(np.array(3.0)) is None  # a scalar
    assert array_field(np.zeros((3, 1))) is None  # a one-pixel strip
    assert array_field(np.zeros(0)) is None  # empty


def test_all_non_finite_field_is_declined() -> None:
    assert array_field(np.full((4, 5), np.nan)) is None


# --- image compatibility ----------------------------------------------------


def test_exact_and_proportional_shapes_are_compatible() -> None:
    assert is_image_compatible(np.zeros((540, 960)), (540, 960))
    assert is_image_compatible(np.zeros((270, 480)), (540, 960))  # half res


def test_differently_proportioned_field_is_not_compatible() -> None:
    # 4:3 values stretched over a 16:9 photo would describe pixels they never
    # measured.
    assert not is_image_compatible(np.zeros((480, 640)), (540, 960))


# --- array_payload: the two storage encodings -------------------------------


def _per_channel(*planes: np.ndarray) -> np.ndarray:
    """Store one plane per class slot, as one array annotation each.

    The "named" encoding: `combine_to_numpy` builds ``(N, n_classes, H, W)``
    populated down the diagonal, so the class axis *is* the channel axis.
    """
    count = len(planes)
    out = np.zeros((count, count, *planes[0].shape), dtype=np.float64)
    for index, plane in enumerate(planes):
        out[index, index] = plane
    return out


def test_channels_inside_one_file_survive_unnamed() -> None:
    # Encoding A: a single (2, H, W) .npy under one class. The channel axis is
    # intact but nothing names it.
    stored = np.zeros((1, 1, 2, 4, 5))
    stored[0, 0, 0] = 1.0
    stored[0, 0, 1] = 2.0
    payload = array_payload(stored)
    assert payload is not None
    assert payload.data.shape == (2, 4, 5)
    assert payload.channel_names is None
    assert payload.channels == 2


def test_channels_on_the_class_axis_come_back_named() -> None:
    # Encoding B: one .npy per channel, so the LDF class names carry over.
    payload = array_payload(
        _per_channel(np.full((4, 5), 1.0), np.full((4, 5), 2.0)),
        class_names=["u", "v"],
    )
    assert payload is not None
    assert payload.data.shape == (2, 4, 5)
    assert payload.channel_names == ["u", "v"]
    assert float(payload.data[0].max()) == 1.0
    assert float(payload.data[1].max()) == 2.0


def test_class_names_that_do_not_line_up_are_dropped() -> None:
    payload = array_payload(
        _per_channel(np.ones((4, 5)), np.ones((4, 5))),
        class_names=["only-one"],
    )
    assert payload is not None
    assert payload.channel_names is None


def test_two_annotations_sharing_a_class_fall_back_to_unnamed() -> None:
    # Summing the instance axis would double-count where they overlap, so the
    # named branch is only taken when every annotation owns a distinct slot.
    stored = np.zeros((2, 1, 2, 2))
    stored[0, 0, 0, 0] = 5.0
    stored[1, 0, 1, 1] = 3.0
    payload = array_payload(stored, class_names=["stereo"])
    assert payload is not None
    assert payload.channel_names is None
    assert np.array_equal(payload.data, [[5.0, 0.0], [0.0, 3.0]])


def test_an_all_zero_annotation_falls_back_to_unnamed() -> None:
    # Nothing is populated, so no slot can be identified. Documented behaviour:
    # the field still draws, it just loses its channel names.
    payload = array_payload(np.zeros((2, 2, 4, 5)), class_names=["u", "v"])
    assert payload is not None
    assert payload.channel_names is None


def test_single_annotation_never_takes_the_named_branch() -> None:
    # One array in a many-class task is a plain field, not a 3-channel stack;
    # reading it as one would turn a disparity map into nonsense.
    payload = array_payload(_loader_shaped(np.ones((4, 5)), n_classes=3))
    assert payload is not None
    assert payload.data.shape == (4, 5)
    assert payload.channels == 0


def test_a_trailing_channel_axis_is_moved_to_the_front() -> None:
    payload = array_payload(np.zeros((6, 8, 3)))
    assert payload is not None
    assert payload.data.shape == (3, 6, 8)


def test_an_ambiguous_small_shape_stays_channels_first() -> None:
    # (3, 8, 3) could be read either way; LDF and torch write channels first.
    payload = array_payload(np.zeros((3, 8, 3)))
    assert payload is not None
    assert payload.data.shape == (3, 8, 3)


# --- kind resolution --------------------------------------------------------


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("disparity", "scalar"),
        ("depth", "scalar"),
        ("error", "signed"),
        ("flow", "flow"),
        ("flow-field", "flow"),  # normalized to flow_field
        ("Optical Flow", "flow"),
        ("normals", "normals"),
        ("rgb", "image"),
        ("segmentation", "scores"),
    ],
)
def test_reserved_task_names_declare_their_kind(
    task_name: str, expected: str
) -> None:
    assert reserved_array_kind(task_name) == expected


def test_reserved_names_match_exactly_and_nothing_else() -> None:
    # Deliberate: a forgiving rule would have to guess which of several
    # matching words wins in a name like `depth_error`.
    assert reserved_array_kind("left_disparity") is None
    assert reserved_array_kind("flow_fwd") is None
    assert reserved_array_kind("stereo") is None


def test_shape_inference_reads_a_signed_field() -> None:
    straddles = np.linspace(-3.0, 3.0, 20).reshape(4, 5)
    assert infer_array_kind(ArrayPayload(straddles)) == "signed"
    assert infer_array_kind(ArrayPayload(np.abs(straddles))) == "scalar"


def test_shape_inference_tells_normals_from_a_picture() -> None:
    normals = np.zeros((3, 4, 5), np.float32)
    normals[2] = 1.0  # unit vectors facing the camera
    assert infer_array_kind(ArrayPayload(normals)) == "normals"
    assert infer_array_kind(ArrayPayload(np.full((3, 4, 5), 200.0))) == "image"


def test_a_deep_unnamed_stack_is_declined() -> None:
    # 21 channels with no names could be anything; guessing would produce a
    # confident wrong picture.
    assert infer_array_kind(ArrayPayload(np.zeros((21, 4, 5)))) is None


def test_a_deep_named_stack_is_class_scores() -> None:
    payload = ArrayPayload(np.zeros((21, 4, 5)), [f"c{i}" for i in range(21)])
    assert infer_array_kind(payload) == "scores"


def test_resolution_precedence_runs_pin_then_name_then_shape() -> None:
    payload = ArrayPayload(np.zeros((2, 4, 5)))
    assert resolve_array_kind("stereo", payload) == "flow"  # shape only
    assert resolve_array_kind("depth", payload) == "scalar"  # name beats shape
    pins: ArrayKinds = (("depth", "image"),)
    assert resolve_array_kind("depth", payload, kinds=pins) == "image"


def test_an_explicit_pin_matches_leniently() -> None:
    payload = ArrayPayload(np.zeros((4, 5)))
    kinds: ArrayKinds = (("Flow-Field", "image"),)
    assert resolve_array_kind("flow_field", payload, kinds=kinds) == "image"


# --- array_annotation / array_annotations -----------------------------------


def test_each_kind_builds_its_own_annotation() -> None:
    facing = np.zeros((3, 4, 5))
    facing[2] = 1.0
    sources = {
        "disparity": (np.linspace(0.0, 9.0, 20).reshape(4, 5), ScalarField),
        "error": (np.linspace(-4.0, 4.0, 20).reshape(4, 5), ScalarField),
        "flow": (np.ones((2, 4, 5)), FlowField),
        "normals": (facing, NormalMap),
        "rgb": (np.full((3, 4, 5), 200.0), ArrayImage),
    }
    for name, (values, expected) in sources.items():
        drawing = array_annotation(
            values, task_name=name, options=RenderOptions()
        )
        assert drawing is not None, name
        assert isinstance(drawing.field, expected), name


def test_a_signed_field_is_centred_on_zero_with_a_diverging_gradient() -> None:
    drawing = array_annotation(
        np.linspace(-4.0, 12.0, 20).reshape(4, 5),
        task_name="error",
        options=RenderOptions(),
    )
    assert drawing is not None
    assert isinstance(drawing.field, ScalarField)
    assert drawing.field.center == 0.0
    # Symmetric about zero, so the neutral colour lands there rather than at
    # whatever min-max stretching happened to put in the middle.
    assert drawing.field.value_range() == (-12.0, 12.0)


def test_the_explicit_centre_beats_the_signed_default() -> None:
    drawing = array_annotation(
        np.linspace(-4.0, 4.0, 20).reshape(4, 5),
        task_name="error",
        options=RenderOptions(array_center=2.0),
    )
    assert drawing is not None
    assert isinstance(drawing.field, ScalarField)
    assert drawing.field.center == 2.0


def test_scores_carry_their_class_names_into_the_mask() -> None:
    scores = np.zeros((3, 2, 2), np.float64)
    scores[1, 0] = 5.0
    scores[2, 1] = 5.0
    drawing = array_annotation(
        _per_channel(*scores),
        task_name="segmentation",
        options=RenderOptions(),
        class_names=["bg", "road", "sky"],
    )
    assert drawing is not None
    assert isinstance(drawing.field, SegmentationScores)
    assert drawing.field.names == ["bg", "road", "sky"]
    assert drawing.field.labels().tolist() == [[1, 1], [2, 2]]
    # The key names only the classes that actually won a pixel, and drops the
    # ignored background.
    assert isinstance(drawing.key, Legend)
    assert drawing.key.entries == ["road", "sky"]


def test_each_kind_gets_the_key_that_reads_it() -> None:
    scalar = array_annotation(
        np.ones((4, 5)), task_name="depth", options=RenderOptions()
    )
    flow = array_annotation(
        np.ones((2, 4, 5)), task_name="flow", options=RenderOptions()
    )
    assert scalar is not None
    assert flow is not None
    assert isinstance(scalar.key, ColorBar)
    assert isinstance(flow.key, FlowWheel)
    # A picture and a normal map need no key: nothing about them is a scale.
    picture = array_annotation(
        np.full((3, 4, 5), 200.0), task_name="rgb", options=RenderOptions()
    )
    assert picture is not None
    assert picture.key is None


def test_keys_can_be_switched_off() -> None:
    drawing = array_annotation(
        np.ones((4, 5)),
        task_name="depth",
        options=RenderOptions(array_colorbar=False),
    )
    assert drawing is not None
    assert drawing.key is None


def test_array_annotations_names_each_field_and_skips_the_rest() -> None:
    built = array_annotations(
        {
            "stereo": _loader_shaped(np.ones((4, 5))),
            "embedding": np.zeros(512),
        },
        options=RenderOptions(),
    )
    assert [drawing.task_name for drawing in built] == ["stereo"]


def test_array_annotations_applies_the_render_options() -> None:
    options = RenderOptions(
        array_vmin=0.0, array_vmax=291.0, array_ignore_value=0.0
    )
    (drawing,) = array_annotations(
        {"stereo": _loader_shaped(np.ones((4, 5)))}, options=options
    )
    field = drawing.field
    assert isinstance(field, ScalarField)
    assert (field.vmin, field.vmax, field.ignore_value) == (0.0, 291.0, 0.0)
    # Magnitude must not drive opacity: a small disparity is a real reading.
    assert field.weight_by_value is False


def test_array_annotations_requires_compatibility_only_when_given_a_shape() -> (
    None
):
    arrays = {"stereo": np.zeros((480, 640))}
    assert array_annotations(arrays, options=RenderOptions()) != []
    assert (
        array_annotations(
            arrays, options=RenderOptions(), image_shape=(540, 960)
        )
        == []
    )


def test_an_overlay_is_translucent_and_a_tile_is_opaque() -> None:
    arrays = {"stereo": np.zeros((540, 960))}
    (tile,) = array_annotations(arrays, options=RenderOptions())
    (overlay,) = array_annotations(
        arrays, options=RenderOptions(), image_shape=(540, 960)
    )
    assert tile.field.alpha == 1.0
    assert overlay.field.alpha == OVERLAY_ALPHA


# --- the annotations themselves ---------------------------------------------


def test_a_field_needs_exactly_one_source(tmp_path: Path) -> None:
    stored = tmp_path / "field.npy"
    np.save(stored, np.ones((4, 5), np.float32))
    with pytest.raises(ValidationError, match="exactly one"):
        ScalarField()
    with pytest.raises(ValidationError, match="exactly one"):
        ScalarField(values=np.ones((4, 5)), path=stored)


def test_a_field_reads_back_from_a_npy_path(tmp_path: Path) -> None:
    stored = tmp_path / "field.npy"
    np.save(stored, np.arange(20, dtype=np.float32).reshape(4, 5))
    field = ScalarField(path=stored)
    assert field.to_numpy().shape == (4, 5)
    assert field.value_range() == (0.0, 19.0)


def test_flow_reports_its_own_peak_unless_pinned() -> None:
    flow = np.zeros((2, 4, 5), np.float32)
    flow[0] = 3.0
    flow[1] = 4.0  # 3-4-5 triangle
    assert FlowField(values=flow).peak_magnitude() == 5.0
    assert FlowField(values=flow, max_magnitude=10.0).peak_magnitude() == 10.0


def test_flow_hue_follows_direction() -> None:
    """Opposite headings must not land on the same colour."""
    canvas = np.zeros((8, 10, 3), np.uint8)
    rights = np.zeros((2, 4, 5), np.float32)
    rights[0] = 1.0
    lefts = np.zeros((2, 4, 5), np.float32)
    lefts[0] = -1.0
    right = Image(canvas).add(FlowField(values=rights)).render()
    left = Image(canvas).add(FlowField(values=lefts)).render()
    assert not np.array_equal(right, left)


def test_scores_confidence_softmaxes_raw_logits() -> None:
    # A bare logit of 9.0 means nothing on its own -- only its size relative to
    # the other classes does -- so an un-normalized stack is softmaxed first.
    scores = np.zeros((3, 2, 2), np.float32)
    scores[1, 0] = 5.0
    scores[2, 1] = 9.0
    confidence = SegmentationScores(values=scores).confidence()
    low, high = confidence.value_range()
    # Pinned to the band a winning probability can occupy: a uniform guess over
    # three classes still awards the winner 1/3.
    assert (low, high) == pytest.approx((1 / 3, 1.0))
    values = confidence.field()
    assert values is not None
    # The 9.0 row won by more, so it is the more confident of the two.
    assert float(values[1, 0]) > float(values[0, 0]) > 1 / 3


def test_scores_confidence_keeps_an_existing_distribution() -> None:
    # Already probabilities: softmaxing again would flatten them towards
    # uniform and under-report every certainty.
    scores = np.zeros((2, 2, 2), np.float32)
    scores[0], scores[1] = 0.5, 0.5
    scores[0, 0, 0], scores[1, 0, 0] = 0.9, 0.1
    values = SegmentationScores(values=scores).confidence().field()
    assert values is not None
    assert float(values[0, 0]) == pytest.approx(0.9)
    assert float(values[0, 1]) == pytest.approx(0.5)


def test_scores_confidence_passes_a_flat_field_through() -> None:
    # One value per pixel is already a certainty map, not a stack to reduce.
    certainty = np.linspace(0.2, 0.8, 12).reshape(3, 4).astype(np.float32)
    confidence = SegmentationScores(values=certainty).confidence()
    assert confidence.value_range() == pytest.approx((0.2, 0.8))


def test_confidence_is_a_kind_of_its_own() -> None:
    # The same stack read two ways: which class won, and by how much.
    scores = np.zeros((3, 4, 5), np.float32)
    scores[1] = 4.0
    as_mask: ArrayKinds = (("pred", "scores"),)
    as_certainty: ArrayKinds = (("pred", "confidence"),)
    mask = array_annotation(
        scores, task_name="pred", options=RenderOptions(array_kinds=as_mask)
    )
    certainty = array_annotation(
        scores,
        task_name="pred",
        options=RenderOptions(array_kinds=as_certainty),
    )
    assert mask is not None
    assert certainty is not None
    assert isinstance(mask.field, SegmentationScores)
    assert isinstance(certainty.field, ScalarField)
    assert isinstance(certainty.key, ColorBar)


def test_a_task_named_confidence_reads_as_one() -> None:
    assert reserved_array_kind("confidence") == "confidence"


def test_class_confidence_fades_the_pixels_the_model_hesitated_on() -> None:
    # Opacity tracks how decisively the winner won. Only a pixel split evenly
    # across *every* class vanishes; a two-way tie among three still says
    # something, and stays partly visible.
    scores = np.zeros((3, 2, 6), np.float32)
    scores[1, :, :2] = 9.0  # class 1 wins outright
    scores[1, :, 2:4] = 4.0  # ...ties class 2, with class 0 out of it
    scores[2, :, 2:4] = 4.0
    scores[:, :, 4:] = 4.0  # ...and all three tie
    field = SegmentationScores(values=scores, weight_by_confidence=True)
    data = field.field()
    assert data is not None
    weights = field._certainty_weights(data)
    decisive, two_way, uniform = (
        float(weights[0, 0]),
        float(weights[0, 3]),
        float(weights[0, 5]),
    )
    assert decisive == pytest.approx(1.0, abs=1e-3)
    assert uniform == pytest.approx(0.0, abs=1e-3)
    assert 0.0 < two_way < decisive


def test_class_confidence_keeps_the_class_colours() -> None:
    """The weighted fill must still be per class, not one gradient."""
    scores = np.zeros((3, 6, 8), np.float32)
    scores[1, :3] = 9.0
    scores[2, 3:] = 9.0
    canvas = np.zeros((6, 8, 3), np.uint8)
    weighted = (
        Image(canvas)
        .add(
            SegmentationScores(
                values=scores,
                names=["bg", "road", "sky"],
                weight_by_confidence=True,
                contour=False,
            )
        )
        .render()
    )
    # Two confident regions, each its own colour -- not two ends of one ramp.
    top, bottom = weighted[0, 0, :3], weighted[5, 0, :3]
    assert not np.array_equal(top, bottom)
    assert weighted[0, 0, 3] > 0
    assert weighted[5, 0, 3] > 0


def test_class_confidence_is_reachable_as_a_kind() -> None:
    scores = np.zeros((3, 4, 5), np.float32)
    scores[1] = 9.0
    pins: ArrayKinds = (("pred", "class_confidence"),)
    drawing = array_annotation(
        scores, task_name="pred", options=RenderOptions(array_kinds=pins)
    )
    assert drawing is not None
    assert isinstance(drawing.field, SegmentationScores)
    assert drawing.field.weight_by_confidence is True
