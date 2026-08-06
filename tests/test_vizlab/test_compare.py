"""Tests for `compare` — detection matching, metrics, and rendering."""

import numpy as np
import pytest

from luxonis_ml.ldf import Detection
from luxonis_ml.vizlab import (
    BBox,
    ComparisonReport,
    ComparisonResult,
    Image,
    Keypoints,
    Verdict,
    compare,
    confusion_matrix_figure,
    match_detections,
)
from luxonis_ml.vizlab.annotations import Annotation
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.comparison.match import (
    CLASS_ERROR_COLOR,
    FN_COLOR,
    FP_COLOR,
    NONE_LABEL,
    TP_COLOR,
    Match,
)
from luxonis_ml.vizlab.comparison.render import (
    _ghost,
    _grade_keypoints,
    _keypoints_of,
    _recolor,
    _render_match,
    _with_keypoint_verdicts,
)
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.options import RenderOptions


def _box(
    x: float,
    y: float,
    w: float = 0.2,
    h: float = 0.2,
    *,
    label: str | None = None,
    score: float | None = None,
) -> BBox:
    return BBox(x=x, y=y, w=w, h=h, label=label, score=score)


def _verdicts(result: ComparisonResult) -> list[Verdict]:
    return [m.verdict for m in result.matches]


def test_verdict_colors_preserve_failure_semantics() -> None:
    assert Color.parse("#35d6a6") == TP_COLOR
    assert Color.parse("#ff6b6b") == FP_COLOR
    assert Color.parse("#ffc24b") == FN_COLOR
    assert Color.parse("#ff9142") == CLASS_ERROR_COLOR


# --------------------------------------------------------------------------- #
# Rect geometry
# --------------------------------------------------------------------------- #
def test_rect_iou_identical_is_one() -> None:
    r = Rect(0.0, 0.0, 2.0, 2.0)
    assert r.iou(r) == 1.0


def test_rect_iou_disjoint_is_zero() -> None:
    assert Rect(0.0, 0.0, 1.0, 1.0).iou(Rect(5.0, 5.0, 6.0, 6.0)) == 0.0


def test_rect_iou_half_overlap() -> None:
    # Two unit squares sharing exactly half their area -> 1/3.
    iou = Rect(0.0, 0.0, 2.0, 2.0).iou(Rect(1.0, 0.0, 3.0, 2.0))
    assert abs(iou - 1.0 / 3.0) < 1e-9


def test_rect_intersection_none_when_touching() -> None:
    assert (
        Rect(0.0, 0.0, 1.0, 1.0).intersection(Rect(1.0, 0.0, 2.0, 1.0)) is None
    )


# --------------------------------------------------------------------------- #
# Matching semantics
# --------------------------------------------------------------------------- #
def test_perfect_match_is_true_positive() -> None:
    gt = [_box(0.1, 0.1, label="car")]
    pred = [_box(0.1, 0.1, label="car", score=0.9)]
    result = match_detections(gt, pred)
    assert _verdicts(result) == [Verdict.TP]
    assert result.matches[0].iou == 1.0


def test_unmatched_prediction_is_false_positive() -> None:
    gt = [_box(0.1, 0.1, label="car")]
    pred = [_box(0.8, 0.8, label="car", score=0.9)]
    result = match_detections(gt, pred)
    assert set(_verdicts(result)) == {Verdict.FP, Verdict.FN}


def test_unmatched_gt_is_false_negative() -> None:
    result = match_detections([_box(0.1, 0.1, label="car")], [])
    assert _verdicts(result) == [Verdict.FN]


def test_wrong_label_is_class_error_when_class_aware() -> None:
    gt = [_box(0.1, 0.1, label="bus")]
    pred = [_box(0.1, 0.1, label="truck", score=0.5)]
    result = match_detections(gt, pred, class_aware=True)
    assert _verdicts(result) == [Verdict.CLASS_ERROR]
    # Counted as FP + FN even though it is drawn as one orange box.
    assert result.n_fp == 1
    assert result.n_fn == 1
    assert result.n_tp == 0


def test_wrong_label_is_true_positive_when_class_agnostic() -> None:
    gt = [_box(0.1, 0.1, label="bus")]
    pred = [_box(0.1, 0.1, label="truck", score=0.5)]
    result = match_detections(gt, pred, class_aware=False)
    assert _verdicts(result) == [Verdict.TP]
    assert result.n_class_errors == 0


def test_below_iou_threshold_does_not_match() -> None:
    gt = [_box(0.0, 0.0, 0.2, 0.2, label="car")]
    # Overlap well under 0.5.
    pred = [_box(0.15, 0.15, 0.2, 0.2, label="car", score=0.9)]
    result = match_detections(gt, pred, iou_threshold=0.5)
    assert set(_verdicts(result)) == {Verdict.FP, Verdict.FN}


def test_score_threshold_drops_low_confidence_predictions() -> None:
    gt = [_box(0.1, 0.1, label="car")]
    pred = [_box(0.1, 0.1, label="car", score=0.1)]
    result = match_detections(gt, pred, score_threshold=0.25)
    # The prediction is discarded before matching, leaving the GT missed.
    assert _verdicts(result) == [Verdict.FN]


def test_greedy_takes_higher_score_first() -> None:
    # Two preds overlap one GT; the higher-scoring one should win the match.
    gt = [_box(0.1, 0.1, 0.2, 0.2, label="car")]
    weak = _box(0.1, 0.1, 0.2, 0.2, label="car", score=0.4)
    strong = _box(0.12, 0.12, 0.2, 0.2, label="car", score=0.95)
    result = match_detections(gt, [weak, strong])
    tp = [m for m in result.matches if m.verdict is Verdict.TP]
    assert len(tp) == 1
    assert tp[0].pred is strong  # the confident box claimed the GT
    assert result.n_fp == 1  # the weaker one is a false positive


def test_scoreless_predictions_rank_behind_scored_ones() -> None:
    # A prediction with no score carries no confidence, so it must not outrank
    # (and steal the GT from) one that is known to be confident.
    gt = [_box(0.1, 0.1, 0.2, 0.2, label="car")]
    scoreless = _box(0.1, 0.1, 0.2, 0.2, label="car")
    strong = _box(0.12, 0.12, 0.2, 0.2, label="car", score=0.99)
    result = match_detections(gt, [scoreless, strong])
    tp = [m for m in result.matches if m.verdict is Verdict.TP]
    assert len(tp) == 1
    assert tp[0].pred is strong
    assert result.n_fp == 1


def test_scoreless_predictions_survive_the_score_threshold() -> None:
    # There is nothing to threshold on, so they are kept rather than dropped.
    gt = [_box(0.1, 0.1, label="car")]
    result = match_detections(
        gt, [_box(0.1, 0.1, label="car")], score_threshold=0.9
    )
    assert _verdicts(result) == [Verdict.TP]


def test_scoreless_predictions_keep_their_given_order() -> None:
    # Among themselves they have no ranking signal, so the first one listed
    # claims the ground truth.
    gt = [_box(0.1, 0.1, 0.2, 0.2, label="car")]
    first = _box(0.1, 0.1, 0.2, 0.2, label="car")
    second = _box(0.11, 0.11, 0.2, 0.2, label="car")
    result = match_detections(gt, [first, second])
    tp = [m for m in result.matches if m.verdict is Verdict.TP]
    assert [m.pred for m in tp] == [first]


# --------------------------------------------------------------------------- #
# Aggregate metrics — the mock scene, hand-checked
# --------------------------------------------------------------------------- #
def _mock_scene() -> tuple[list[BBox], list[BBox]]:
    gt = [
        _box(0.05, 0.55, 0.25, 0.25, label="car"),
        _box(0.42, 0.35, 0.08, 0.45, label="person"),
        _box(0.52, 0.28, 0.28, 0.36, label="bus"),
        _box(0.85, 0.34, 0.08, 0.46, label="person"),  # missed
    ]
    pred = [
        _box(0.06, 0.56, 0.25, 0.25, label="car", score=0.91),
        _box(0.42, 0.35, 0.08, 0.45, label="person", score=0.88),
        _box(0.52, 0.28, 0.28, 0.36, label="truck", score=0.44),  # class error
        _box(0.36, 0.75, 0.13, 0.13, label="car", score=0.38),  # hallucination
    ]
    return gt, pred


def test_mock_scene_counts_and_metrics() -> None:
    gt, pred = _mock_scene()
    result = match_detections(gt, pred)
    assert result.n_tp == 2
    assert result.n_class_errors == 1
    assert result.n_fp == 2  # 1 hallucination + 1 class error
    assert result.n_fn == 2  # 1 missed person + 1 class error
    assert abs(result.precision - 0.5) < 1e-9
    assert abs(result.recall - 0.5) < 1e-9
    assert abs(result.f1 - 0.5) < 1e-9
    assert (
        result.mean_iou > 0.9
    )  # near-perfect boxes for the localized matches


def test_summary_is_panel_ready() -> None:
    gt, pred = _mock_scene()
    summary = match_detections(gt, pred).summary()
    assert summary["precision"] == "0.50"
    assert summary["TP"] == 2
    assert summary["FP"] == 2
    assert summary["class errors"] == 1


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _blank() -> np.ndarray:
    return np.zeros((80, 120, 3), np.uint8)


def test_compare_returns_image_with_panel() -> None:
    from luxonis_ml.vizlab.scene.image import Renderable

    gt, pred = _mock_scene()
    img = compare(_blank(), gt=gt, pred=pred)
    assert isinstance(
        img, Renderable
    )  # a panel composite, renderable to any target
    # A metrics panel widens the composite beyond the source image.
    assert img.render().shape[1] > 120


def test_compare_colors_boxes_by_verdict() -> None:
    gt = [_box(0.1, 0.1, label="car")]
    pred = [_box(0.8, 0.1, label="car", score=0.9)]  # FP + FN
    img = compare(_blank(), gt=gt, pred=pred, panel=False)
    colors = {b.color for b in img.annotations if isinstance(b, BBox)}
    assert FP_COLOR in colors
    assert FN_COLOR in colors


def test_true_positive_draws_a_ghost() -> None:
    gt = [_box(0.1, 0.1, label="car")]
    pred = [_box(0.1, 0.1, label="car", score=0.9)]
    img = compare(_blank(), gt=gt, pred=pred, panel=False)
    boxes = [b for b in img.annotations if isinstance(b, BBox)]
    assert len(boxes) == 2  # the ghost GT plus the green prediction
    assert any(b.color == TP_COLOR for b in boxes)


def test_tooltip_explains_the_verdict() -> None:
    gt = [_box(0.1, 0.1, label="bus")]
    pred = [_box(0.1, 0.1, label="truck", score=0.44)]
    img = compare(_blank(), gt=gt, pred=pred, panel=False)
    box = next(
        b
        for b in img.annotations
        if isinstance(b, BBox) and b.color == CLASS_ERROR_COLOR
    )
    assert box.tooltip is not None
    assert ("verdict", "class error") in box.tooltip.rows
    assert box.tooltip.title == "bus → truck"


def test_compare_accepts_precomputed_result() -> None:
    gt, pred = _mock_scene()
    result = match_detections(gt, pred)
    img = compare(_blank(), result=result, panel=False)
    boxes = [b for b in img.annotations if isinstance(b, BBox)]
    assert boxes  # rendered from the result without re-matching


def test_render_hits_reports_hover_regions() -> None:
    gt, pred = _mock_scene()
    img = compare(_blank(), gt=gt, pred=pred, panel=False)
    _, hitmap = img.render_hits()
    assert len(hitmap.items) > 0


# --------------------------------------------------------------------------- #
# Per-class metrics
# --------------------------------------------------------------------------- #
def test_per_class_breakdown() -> None:
    gt, pred = _mock_scene()
    per_class = match_detections(gt, pred).per_class
    # car: one hit plus one hallucination -> P 0.5, R 1.0.
    assert per_class["car"] == {
        "precision": 0.5,
        "recall": 1.0,
        "tp": 1,
        "fp": 1,
        "fn": 0,
    }
    # The class error splits across the two labels it touches.
    assert per_class["bus"]["fn"] == 1  # true class missed
    assert per_class["truck"]["fp"] == 1  # predicted class hallucinated
    assert per_class["bus"]["tp"] == 0


def test_per_class_panel_rows_are_compact_strings() -> None:
    gt, pred = _mock_scene()
    rows = match_detections(gt, pred).per_class_panel()
    assert set(rows) == {"car", "person", "bus", "truck"}
    assert rows["car"].startswith("P 0.50  R 1.00")


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #
def test_invalid_show_raises() -> None:
    gt, pred = _mock_scene()
    with pytest.raises(ValueError, match="mosaic"):
        compare(_blank(), gt=gt, pred=pred, show="mosaic")  # type: ignore[arg-type]


def test_side_by_side_is_two_panels_wide() -> None:
    gt, pred = _mock_scene()
    overlay = compare(_blank(), gt=gt, pred=pred, panel=False)
    sbs = compare(_blank(), gt=gt, pred=pred, show="side_by_side", panel=False)
    # Ground truth beside prediction spans well over a single frame.
    assert sbs.render().shape[1] > overlay.render().shape[1] * 1.8


def test_triptych_is_three_panels_wide() -> None:
    gt, pred = _mock_scene()
    sbs = compare(_blank(), gt=gt, pred=pred, show="side_by_side", panel=False)
    trip = compare(_blank(), gt=gt, pred=pred, show="triptych", panel=False)
    assert trip.render().shape[1] > sbs.render().shape[1]


def test_layouts_render_without_error() -> None:
    gt, pred = _mock_scene()
    for show in ("overlay", "side_by_side", "triptych"):
        img = compare(_blank(), gt=gt, pred=pred, show=show, per_class=True)
        assert img.render().ndim == 3


def test_side_by_side_preserves_tooltips() -> None:
    # Composing the two panels must not drop the hover regions.
    gt = [_box(0.1, 0.3, 0.3, 0.4, label="car")]
    pred = [_box(0.1, 0.3, 0.3, 0.4, label="car", score=0.9)]
    img = compare(_blank(), gt=gt, pred=pred, show="side_by_side", panel=False)
    _, hitmap = img.render_hits()
    assert hitmap.items  # tooltips survived the composition
    assert any(("verdict", "true positive") in t.rows for _, t in hitmap.items)


def test_triptych_diff_panel_is_hoverable() -> None:
    gt = [_box(0.1, 0.3, 0.3, 0.4, label="bus")]
    pred = [_box(0.1, 0.3, 0.3, 0.4, label="truck", score=0.5)]  # class error
    img = compare(_blank(), gt=gt, pred=pred, show="triptych", panel=False)
    _, hitmap = img.render_hits()
    assert any(("verdict", "class error") in t.rows for _, t in hitmap.items)


# --------------------------------------------------------------------------- #
# Dataset-level report, confusion matrix, verdict filter
# --------------------------------------------------------------------------- #
def test_report_aggregates_over_images() -> None:
    gt, pred = _mock_scene()
    report = ComparisonReport()
    for i in range(4):
        report.add(match_detections(gt, pred), name=str(i))
    assert report.n_images == 4
    assert report.n_tp == 8  # 2 true positives per image
    assert abs(report.precision - 0.5) < 1e-9
    assert report.summary()["images"] == 4
    # Each image has 3 errors (1 FP + 1 class error + 1 FN).
    assert report.worst(1) == [(3, "3")]
    assert len(report.worst(10)) == 4


def test_report_confusion_matrix() -> None:
    gt = [
        _box(0.1, 0.1, label="car"),
        _box(0.5, 0.5, label="bus"),
        _box(0.8, 0.1, label="car"),  # will be missed
    ]
    pred = [
        _box(0.1, 0.1, label="car", score=0.9),  # TP: car -> car
        _box(0.5, 0.5, label="truck", score=0.5),  # class error: bus -> truck
        _box(0.3, 0.8, label="dog", score=0.4),  # FP: nothing -> dog
    ]
    report = ComparisonReport()
    report.add(match_detections(gt, pred))
    labels, matrix = report.confusion_matrix()
    at = {label: i for i, label in enumerate(labels)}
    assert matrix[at["car"]][at["car"]] == 1  # true positive on the diagonal
    assert matrix[at["bus"]][at["truck"]] == 1  # class confusion, off-diagonal
    assert matrix[at[NONE_LABEL]][at["dog"]] == 1  # false positive (∅ -> dog)
    assert matrix[at["car"]][at[NONE_LABEL]] == 1  # false negative (car -> ∅)


def test_confusion_matrix_figure_renders() -> None:
    gt, pred = _mock_scene()
    report = ComparisonReport()
    report.add(match_detections(gt, pred))
    out = confusion_matrix_figure(report).render()
    assert out.ndim == 3
    assert out.shape[0] > 0
    assert out.shape[1] > 0


def test_verdicts_filter_draws_only_selected() -> None:
    gt = [_box(0.1, 0.1, label="car")]
    pred = [
        _box(0.1, 0.1, label="car", score=0.9),  # TP
        _box(0.7, 0.7, label="dog", score=0.5),  # FP
    ]
    img = compare(
        _blank(),
        gt=gt,
        pred=pred,
        panel=False,
        verdicts={Verdict.FP},
    )
    colors = {b.color for b in img.annotations if isinstance(b, BBox)}
    assert FP_COLOR in colors  # the false positive is drawn
    assert (
        TP_COLOR not in colors
    )  # the true positive (and its ghost) are hidden


def test_verdicts_filter_keeps_full_metrics_in_panel() -> None:
    gt = [_box(0.1, 0.1, label="car")]
    pred = [_box(0.1, 0.1, label="car", score=0.9)]  # a lone TP
    # Filtering to errors draws nothing, but the panel still reports the hit.
    img = compare(
        _blank(), gt=gt, pred=pred, verdicts={Verdict.FP}, per_class=False
    )
    assert img.render().shape[1] > 120  # the metrics panel is still attached


def test_carried_hitmap_scales_with_render_size() -> None:
    # A carried composite hit map scales with the render size, like an
    # annotation's captured region would.
    gt = [_box(0.1, 0.3, 0.3, 0.4, label="car")]
    pred = [_box(0.1, 0.3, 0.3, 0.4, label="car", score=0.9)]
    img = compare(_blank(), gt=gt, pred=pred, show="side_by_side", panel=False)
    _, native = img.render_hits()
    _, doubled = img.render_hits((img.width * 2, img.height * 2))
    assert native.items
    assert doubled.items
    near, far = native.items[0][0], doubled.items[0][0]
    assert abs(far.left - 2 * near.left) < 1.0
    assert abs(far.right - 2 * near.right) < 1.0


def test_compare_requires_inputs_or_a_result() -> None:
    with pytest.raises(ValueError, match="needs gt and pred"):
        compare(_blank(), panel=False)


# --------------------------------------------------------------------------- #
# LDF Detection support
# --------------------------------------------------------------------------- #
def _det(
    x: float,
    y: float,
    w: float = 0.2,
    h: float = 0.2,
    *,
    label: str,
    keypoints: bool = False,
) -> Detection:
    data: dict[str, object] = {
        "class_name": label,
        "boundingbox": {"x": x, "y": y, "w": w, "h": h},
    }
    if keypoints:
        data["keypoints"] = {"keypoints": [(x + 0.02, y + 0.02, 2)]}
    return Detection.model_validate(data)


def test_match_on_ldf_detections() -> None:
    gt = [_det(0.1, 0.1, label="car"), _det(0.6, 0.6, label="person")]
    pred = [_det(0.1, 0.1, label="car")]  # hits car, misses person
    result = match_detections(gt, pred)
    assert result.n_tp == 1
    assert result.n_fn == 1
    assert isinstance(result.matches[0].pred, Detection)


def test_ldf_metadata_controls_score_filtering_and_ranking() -> None:
    gt = [_det(0.1, 0.1, label="car")]
    dropped = _det(0.7, 0.7, label="car").model_copy(
        update={"metadata": {"score": 0.1}}
    )
    weak = _det(0.1, 0.1, label="car").model_copy(
        update={"metadata": {"score": 0.4}}
    )
    strong = _det(0.12, 0.12, label="car").model_copy(
        update={"metadata": {"confidence": 0.9}}
    )

    result = match_detections(
        gt, [dropped, weak, strong], score_threshold=0.25
    )

    assert result.n_tp == 1
    assert result.n_fp == 1
    tp = next(match for match in result.matches if match.verdict is Verdict.TP)
    assert tp.pred is strong
    assert tp.score == 0.9


def test_keypoint_only_detection_matched_by_extent() -> None:
    # Neither side has a box; bounds come from the keypoints' extent.
    kps = {"keypoints": [(0.2, 0.2, 2), (0.4, 0.4, 2)]}
    gt = [Detection.model_validate({"class_name": "pose", "keypoints": kps})]
    pred = [Detection.model_validate({"class_name": "pose", "keypoints": kps})]
    result = match_detections(gt, pred)
    assert result.n_tp == 1


def test_keypoint_only_match_renders_a_top_level_graded_pose() -> None:
    keypoints = {"keypoints": [(0.2, 0.2, 2), (0.4, 0.4, 2)]}
    gt = Detection.model_validate(
        {"class_name": "pose", "keypoints": keypoints}
    )
    pred = Detection.model_validate(
        {"class_name": "pose", "keypoints": keypoints}
    )

    image = compare(_blank(), gt=[gt], pred=[pred], panel=False)

    pose = next(
        annotation
        for annotation in image.annotations
        if isinstance(annotation, Keypoints)
    )
    assert pose.point_colors == [TP_COLOR, TP_COLOR]


def test_keypoint_bounds_exclude_invisible_placeholders() -> None:
    gt = [
        Detection.model_validate(
            {
                "class_name": "pose",
                "keypoints": {
                    "keypoints": [
                        (0.0, 0.0, 0),
                        (0.8, 0.8, 2),
                        (0.9, 0.9, 2),
                    ]
                },
            }
        )
    ]
    pred = [
        Detection.model_validate(
            {
                "class_name": "pose",
                "keypoints": {
                    "keypoints": [
                        (0.0, 0.0, 0),
                        (0.6, 0.8, 2),
                        (0.7, 0.9, 2),
                    ]
                },
            }
        )
    ]

    result = match_detections(gt, pred)

    assert set(_verdicts(result)) == {Verdict.FP, Verdict.FN}


def test_mask_only_detection_has_no_bounds_and_is_skipped() -> None:
    from luxonis_ml.ldf import Detection as Det

    boxed = _det(0.1, 0.1, label="car")
    # A record-level segmentation detection has no box or keypoints to match on.
    maskish = Det.model_validate({"class_name": "road"})
    result = match_detections([boxed, maskish], [boxed])
    # Only the boxed pair matches; the boundless detection is ignored entirely.
    assert result.n_tp == 1
    assert result.n_fn == 0


def test_compare_renders_detection_tree() -> None:
    gt = [_det(0.1, 0.1, 0.4, 0.4, label="car", keypoints=True)]
    pred = [_det(0.1, 0.1, 0.4, 0.4, label="car", keypoints=True)]
    img = compare(_blank(), gt=gt, pred=pred, panel=False)
    roots = [a for a in img.annotations if isinstance(a, BBox)]
    tp_root = next(a for a in roots if a.color == TP_COLOR)
    # The keypoints ride along as children of the verdict-colored box.
    assert tp_root.children


# --------------------------------------------------------------------------- #
# Per-keypoint verdicts
# --------------------------------------------------------------------------- #
def _pose(
    box: dict[str, float], joints: list[tuple[float, float, int]]
) -> Detection:
    return Detection.model_validate(
        {
            "class_name": "person",
            "boundingbox": box,
            "keypoints": {"keypoints": joints},
        }
    )


def _joint_colors(img: Image) -> list[ColorLike | None]:
    root = next(
        a
        for a in img.annotations
        if isinstance(a, BBox) and a.color == TP_COLOR
    )
    pose = next(c for c in root.children if isinstance(c, Keypoints))
    assert pose.point_colors is not None  # graded per joint
    return list(pose.point_colors)


def test_keypoints_full_match_are_all_green() -> None:
    box = {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.6}
    joints = [(0.3, 0.3, 2), (0.5, 0.3, 2), (0.4, 0.7, 2)]
    img = compare(
        _blank(),
        gt=[_pose(box, joints)],
        pred=[_pose(box, joints)],
        panel=False,
    )
    colors = _joint_colors(img)
    assert len(colors) == 3
    assert all(c == TP_COLOR for c in colors)


def test_partly_misplaced_keypoints_go_red() -> None:
    box = {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.6}
    gt_j = [(0.3, 0.3, 2), (0.5, 0.3, 2), (0.4, 0.7, 2)]
    pred_j = [(0.3, 0.3, 2), (0.5, 0.3, 2), (0.9, 0.1, 2)]  # 3rd far off
    img = compare(
        _blank(), gt=[_pose(box, gt_j)], pred=[_pose(box, pred_j)], panel=False
    )
    colors = _joint_colors(img)
    assert TP_COLOR in colors  # the two on-target joints
    assert FP_COLOR in colors  # the displaced one


def test_missed_keypoint_is_amber() -> None:
    box = {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.6}
    gt_j = [(0.3, 0.3, 2), (0.5, 0.3, 2), (0.4, 0.7, 2)]
    pred_j = [(0.3, 0.3, 2), (0.5, 0.3, 2), (0.4, 0.7, 0)]  # 3rd not predicted
    img = compare(
        _blank(), gt=[_pose(box, gt_j)], pred=[_pose(box, pred_j)], panel=False
    )
    assert FN_COLOR in _joint_colors(img)


def test_unmatched_pose_keeps_one_verdict_color() -> None:
    # No pair -> no per-joint grading; the whole miss stays one amber pose.
    box = {"x": 0.6, "y": 0.6, "w": 0.3, "h": 0.3}
    joints = [(0.7, 0.7, 2), (0.8, 0.8, 2)]
    img = compare(_blank(), gt=[_pose(box, joints)], pred=[], panel=False)
    roots = [a for a in img.annotations if isinstance(a, BBox)]
    assert any(a.color == FN_COLOR for a in roots)


def test_comparison_helpers_handle_unrenderable_and_partial_inputs() -> None:
    options = RenderOptions()
    empty = Detection(class_name=None)
    assert _recolor(empty, options, TP_COLOR) == []
    assert _ghost(empty) == []
    assert _keypoints_of(empty) is None

    pose = Detection.model_validate(
        {
            "class_name": "pose",
            "keypoints": {"keypoints": [(0.2, 0.2, 2)]},
        }
    )
    match = Match(Verdict.TP, pose, pose, 1.0)
    annotations: list[Annotation] = [BBox(x=0.1, y=0.1, w=0.2, h=0.2)]
    assert _with_keypoint_verdicts(annotations, match) is annotations


def test_grade_keypoints_marks_an_invented_joint_false_positive() -> None:
    points, colors = _grade_keypoints(
        [(0.3, 0.4, 2)],
        [(0.0, 0.0, 0)],
        Rect(0.0, 0.0, 1.0, 1.0),
    )
    assert points == [(0.3, 0.4, 2)]
    assert colors == [FP_COLOR]


def test_grade_keypoints_grades_joints_the_prediction_never_produced() -> None:
    """A shorter prediction (13-joint model vs 17-joint GT) still shows misses."""
    points, colors = _grade_keypoints(
        [(0.3, 0.4, 2)],
        [(0.3, 0.4, 2), (0.6, 0.7, 2), (0.8, 0.9, 1)],
        Rect(0.0, 0.0, 1.0, 1.0),
    )
    # The matched joint, plus the two the prediction lacks entirely, drawn as
    # missed at their ground-truth positions.
    assert points == [(0.3, 0.4, 2), (0.6, 0.7, 2), (0.8, 0.9, 2)]
    assert colors == [TP_COLOR, FN_COLOR, FN_COLOR]


def test_grade_keypoints_skips_joints_absent_on_both_sides() -> None:
    # Index kept (the skeleton's edges are index-based) but invisible, so
    # nothing is drawn for a joint neither side labels.
    points, colors = _grade_keypoints(
        [(0.3, 0.4, 2)],
        [(0.3, 0.4, 2), (0.0, 0.0, 0)],
        Rect(0.0, 0.0, 1.0, 1.0),
    )
    assert len(points) == len(colors) == 2
    assert points[1][2] == 0


def test_render_match_ignores_incomplete_match() -> None:
    incomplete = Match(Verdict.TP, None, None)
    assert _render_match(incomplete, RenderOptions()) == []


def test_empty_comparison_report_per_class_is_empty() -> None:
    assert ComparisonReport().per_class() == {}
