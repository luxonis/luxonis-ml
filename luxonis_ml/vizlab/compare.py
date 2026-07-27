"""Ground-truth vs prediction comparison: match, score, and render.

`compare` takes an image plus two sets of detections — ground truth and a model's
predictions — matches them, and returns an `Image` where color *is* the verdict:
green for a hit, red for a false alarm, dashed amber for a miss, and orange for a
box that landed on the right object but with the wrong label.

Detections may be vizlab `BBox` annotations or LDF
`Detection` objects; a ``Detection``'s whole annotation
tree (box, keypoints, instance mask, sub-detections) is drawn in its verdict
color, so masks and keypoints inherit the outcome. Matching is by bounding box —
a detection's own box, or one derived from its keypoints when it has none. When a
matched pair carries keypoints, they are graded per joint (green within
`KEYPOINT_TOLERANCE` of the ground-truth joint, red when off, amber when missed)
rather than all one color, so a partly-correct pose reads at a glance; a skeleton
limb between two differently graded joints fades from one color to the other.

Matching mirrors COCO: predictions are taken in descending confidence order and
each greedily claims the best-overlapping unclaimed ground-truth box of the
*same* class (`match_detections`). Metrics are counted the COCO way — a
wrong-label prediction is a false positive and the ground-truth box it covered is
a false negative. The orange "class error" verdict is a second, cosmetic pass
over those leftovers: it recolors the pair so the mistake is visible, but it still
scores as one false positive plus one false negative, so the numbers never move.

The matcher (`match_detections`) is pure — no image, no rendering — so it is
cheap to test. `compare` layers the drawing on top and can also take a
precomputed `ComparisonResult` (``result=``) from any caller that already matched.
"""

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np

from .annotations import Annotation, BBox, Keypoints
from .color import Color
from .geometry import XY, Rect, bounding_rect
from .options import RenderOptions
from .tooltip import Tooltip

if TYPE_CHECKING:
    from luxonis_ml.ldf import Detection

    from .canvas import Canvas
    from .image import Image
    from .io import ImageSource
    from .panel import PanelData

    #: A matchable detection: a vizlab box or a full LDF detection tree.
    Detectionish = BBox | Detection

#: Verdict colors, tuned for the dark composite background. Themeable later.
TP_COLOR = Color.parse("#35d6a6")
FP_COLOR = Color.parse("#ff6b6b")
FN_COLOR = Color.parse("#ffc24b")
CLASS_ERROR_COLOR = Color.parse("#ff9142")
#: Faint tint for the ground-truth "ghost" drawn behind a true positive.
_GHOST_COLOR = Color(234, 241, 248, 110)
#: Identity hues for the side-by-side view — a matched pair shares one, and they
#: stay clear of the verdict colors so the two color languages never collide.
_IDENTITY_COLORS = (
    Color.parse("#38bdf8"),  # cyan
    Color.parse("#a78bfa"),  # violet
    Color.parse("#2dd4bf"),  # teal
    Color.parse("#f472b6"),  # pink
    Color.parse("#a3e635"),  # lime
    Color.parse("#e879f9"),  # fuchsia
)


#: A predicted keypoint within this fraction of the match bounds' diagonal of its
#: ground-truth counterpart counts as correct (a PCK-style tolerance).
KEYPOINT_TOLERANCE = 0.1


def _faded(color: Color, alpha: int = 120) -> Color:
    """Return ``color`` at a lower opacity, for a "missing partner" ghost."""
    return Color(color.r, color.g, color.b, alpha)


class Verdict(Enum):
    """The outcome of matching one detection against the ground truth."""

    TP = "true_positive"
    FP = "false_positive"
    FN = "false_negative"
    CLASS_ERROR = "class_error"


@dataclass(frozen=True)
class Match:
    """One matched (or unmatched) detection and its verdict.

    Attributes:
        verdict: The outcome (`Verdict`).
        gt: The ground-truth detection, or ``None`` for a false positive.
        pred: The predicted detection, or ``None`` for a false negative.
        iou: Overlap between ``gt`` and ``pred`` for a localized match; ``0.0``
            for an unmatched false positive or false negative.

    """

    verdict: Verdict
    gt: "Detectionish | None"
    pred: "Detectionish | None"
    iou: float = 0.0

    @property
    def score(self) -> float | None:
        """The prediction's confidence, if this match has one."""
        return _score(self.pred) if self.pred is not None else None


@dataclass(frozen=True)
class ComparisonResult:
    """The matches between a prediction set and the ground truth, plus metrics.

    Counts follow COCO: a class error contributes one false positive (its
    prediction) and one false negative (the ground-truth box it covered), so
    `precision` and `recall` agree with an AP@``iou_threshold`` evaluation.
    `n_class_errors` is surfaced only so the visualization can flag it.

    Attributes:
        matches: Every match, in no particular order.
        iou_threshold: The overlap threshold a localized match had to meet.
        score_threshold: Predictions below this confidence were dropped first.

    """

    matches: tuple[Match, ...]
    iou_threshold: float
    score_threshold: float

    def _count(self, verdict: Verdict) -> int:
        return sum(1 for m in self.matches if m.verdict is verdict)

    @property
    def n_tp(self) -> int:
        """Number of true positives."""
        return self._count(Verdict.TP)

    @property
    def n_class_errors(self) -> int:
        """Number of localized-but-mislabeled detections (visual sugar only)."""
        return self._count(Verdict.CLASS_ERROR)

    @property
    def n_fp(self) -> int:
        """False positives, COCO-style (raw false alarms plus class errors)."""
        return self._count(Verdict.FP) + self.n_class_errors

    @property
    def n_fn(self) -> int:
        """False negatives, COCO-style (raw misses plus class errors)."""
        return self._count(Verdict.FN) + self.n_class_errors

    @property
    def precision(self) -> float:
        """``TP / (TP + FP)``; ``0.0`` when there are no predictions."""
        denom = self.n_tp + self.n_fp
        return self.n_tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """``TP / (TP + FN)``; ``0.0`` when there is no ground truth."""
        denom = self.n_tp + self.n_fn
        return self.n_tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of `precision` and `recall`."""
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_iou(self) -> float:
        """Mean overlap over localized matches (true positives and class errors)."""
        ious = [
            m.iou
            for m in self.matches
            if m.verdict in (Verdict.TP, Verdict.CLASS_ERROR)
        ]
        return sum(ious) / len(ious) if ious else 0.0

    @property
    def per_class(self) -> dict[str, dict[str, float | int]]:
        """Per-class ``{precision, recall, tp, fp, fn}``, keyed by class name.

        A class error contributes a false positive to its predicted class and a
        false negative to its true class, mirroring the aggregate counting.
        """
        tally: dict[str, list[int]] = {}

        def bump(label: str | None, slot: int) -> None:
            tally.setdefault(label or "object", [0, 0, 0])[slot] += 1

        for m in self.matches:
            if m.verdict is Verdict.TP and m.gt is not None:
                bump(_label(m.gt), 0)
            elif m.verdict is Verdict.FP and m.pred is not None:
                bump(_label(m.pred), 1)
            elif m.verdict is Verdict.FN and m.gt is not None:
                bump(_label(m.gt), 2)
            elif m.verdict is Verdict.CLASS_ERROR:
                if m.pred is not None:
                    bump(_label(m.pred), 1)
                if m.gt is not None:
                    bump(_label(m.gt), 2)

        out: dict[str, dict[str, float | int]] = {}
        for name, (tp, fp, fn) in sorted(tally.items()):
            out[name] = {
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return out

    def summary(self) -> dict[str, str | int]:
        """Aggregate metrics as a panel-ready mapping."""
        return {
            "precision": f"{self.precision:.2f}",
            "recall": f"{self.recall:.2f}",
            "F1": f"{self.f1:.2f}",
            "mean IoU": f"{self.mean_iou:.2f}",
            "TP": self.n_tp,
            "FP": self.n_fp,
            "FN": self.n_fn,
            "class errors": self.n_class_errors,
        }

    def per_class_panel(self) -> dict[str, str]:
        """Per-class metrics as compact ``{class: "P .. R .. (tp/fp/fn)"}`` rows."""
        return {
            name: f"P {v['precision']:.2f}  R {v['recall']:.2f}  "
            f"({v['tp']}/{v['fp']}/{v['fn']})"
            for name, v in self.per_class.items()
        }


# --------------------------------------------------------------------------- #
# Reading detections uniformly (a vizlab BBox or an LDF Detection)
# --------------------------------------------------------------------------- #
def _label(obj: "Detectionish") -> str | None:
    """Return the class label of a box or detection."""
    return obj.label if isinstance(obj, BBox) else obj.class_name


def _score(obj: "Detectionish") -> float | None:
    """Return the confidence of a box; LDF detections carry none."""
    return obj.score if isinstance(obj, BBox) else None


def _bounds(obj: "Detectionish") -> Rect | None:
    """Return normalized match bounds: the box, else the keypoints' extent.

    Rotation is ignored (axis-aligned bounds). A detection with neither a box nor
    keypoints (e.g. mask-only) has no matchable bounds and is skipped.
    """
    if isinstance(obj, BBox):
        return Rect(obj.x, obj.y, obj.x + obj.w, obj.y + obj.h)
    box = obj.boundingbox
    if box is not None:
        return Rect(box.x, box.y, box.x + box.w, box.y + box.h)
    keypoints = obj.keypoints
    if keypoints is not None and keypoints.keypoints:
        return bounding_rect([(p[0], p[1]) for p in keypoints.keypoints])
    return None


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _best(
    pred_rect: Rect,
    pred_label: str | None,
    gts: "Sequence[tuple[Detectionish, Rect]]",
    taken: set[int],
    iou_threshold: float,
    *,
    class_filter: bool,
) -> tuple[int, float]:
    """Return the ``(index, iou)`` of the best free GT for a prediction rect.

    With ``class_filter``, only ground truth whose label equals ``pred_label`` is
    eligible (the COCO-consistent pass); otherwise any label may match (the
    class-error pass).
    """
    best_idx, best_iou = -1, 0.0
    for i, (gt, gt_rect) in enumerate(gts):
        if i in taken:
            continue
        if class_filter and _label(gt) != pred_label:
            continue
        iou = pred_rect.iou(gt_rect)
        if iou >= iou_threshold and iou > best_iou:
            best_idx, best_iou = i, iou
    return best_idx, best_iou


def match_detections(
    gt: "Sequence[Detectionish]",
    pred: "Sequence[Detectionish]",
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
    class_aware: bool = True,
) -> ComparisonResult:
    """Match predictions to ground truth, COCO-style, and score the result.

    Predictions below ``score_threshold`` are dropped, then the rest are taken in
    descending confidence order (detections without a score keep their given
    order); each greedily claims the highest-IoU unclaimed ground-truth box (of
    the same class when ``class_aware``). A claim at or above ``iou_threshold`` is
    a true positive. When ``class_aware``, a second pass pairs each leftover
    prediction with an unclaimed ground-truth box of a *different* class that it
    still overlaps — a class error, drawn distinctly but counted as a false
    positive plus a false negative. Everything unclaimed is a false positive
    (prediction) or false negative (ground truth). Detections with no matchable
    bounds (neither a box nor keypoints) are ignored.

    Args:
        gt: Ground-truth detections (`BBox` or LDF `Detection`).
        pred: Predicted detections; ``score`` orders them when present.
        iou_threshold: Minimum overlap for a localized match.
        score_threshold: Minimum prediction confidence to consider.
        class_aware: When ``True`` (default), only same-class boxes are true
            positives and mismatched labels surface as class errors. When
            ``False``, any sufficiently overlapping pair is a true positive and no
            class errors arise.

    Returns:
        A `ComparisonResult` with every `Match` and the aggregate metrics.

    """
    gts: list[tuple[Detectionish, Rect]] = []
    for obj in gt:
        rect = _bounds(obj)
        if rect is not None:
            gts.append((obj, rect))

    # Predictions above the score cutoff, highest confidence first (unscored
    # detections sort as fully confident so their given order is preserved).
    ranked: list[tuple[Detectionish, Rect, float]] = []
    for obj in pred:
        rect = _bounds(obj)
        if rect is None:
            continue
        score = _score(obj)
        if score is not None and score < score_threshold:
            continue
        ranked.append((obj, rect, 1.0 if score is None else score))
    ranked.sort(key=lambda it: it[2], reverse=True)

    taken: set[int] = set()
    matches: list[Match] = []
    leftover: list[tuple[Detectionish, Rect]] = []

    # Pass 1 — COCO-consistent: same-class greedy claims define the true positives.
    for obj, rect, _score_val in ranked:
        idx, iou = _best(
            rect,
            _label(obj),
            gts,
            taken,
            iou_threshold,
            class_filter=class_aware,
        )
        if idx < 0 and not class_aware:
            idx, iou = _best(
                rect,
                _label(obj),
                gts,
                taken,
                iou_threshold,
                class_filter=False,
            )
        if idx >= 0:
            taken.add(idx)
            matches.append(Match(Verdict.TP, gts[idx][0], obj, iou))
        else:
            leftover.append((obj, rect))

    # Pass 2 — cosmetic: a leftover prediction over a different-class GT is a
    # class error (still one FP + one FN in the metrics).
    for obj, rect in leftover:
        idx, iou = -1, 0.0
        if class_aware:
            idx, iou = _best(
                rect,
                _label(obj),
                gts,
                taken,
                iou_threshold,
                class_filter=False,
            )
        if idx >= 0:
            taken.add(idx)
            matches.append(Match(Verdict.CLASS_ERROR, gts[idx][0], obj, iou))
        else:
            matches.append(Match(Verdict.FP, None, obj, 0.0))

    for i, (obj, _rect) in enumerate(gts):
        if i not in taken:
            matches.append(Match(Verdict.FN, obj, None, 0.0))

    return ComparisonResult(tuple(matches), iou_threshold, score_threshold)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_class(obj: "Detectionish | None") -> str:
    """Human-readable class name for a detection, defaulting to ``object``."""
    label = _label(obj) if obj is not None else None
    return label or "object"


def _match_tooltip(match: Match) -> Tooltip:
    """Build the hover explanation for a match from its verdict."""
    v = match.verdict
    if v is Verdict.TP:
        return Tooltip(
            title=_fmt_class(match.gt),
            tint=TP_COLOR,
            rows=(
                ("verdict", "true positive"),
                ("IoU", f"{match.iou:.2f}"),
                (
                    "score",
                    "—" if match.score is None else f"{match.score:.2f}",
                ),
            ),
        )
    if v is Verdict.FP:
        return Tooltip(
            title=_fmt_class(match.pred),
            tint=FP_COLOR,
            rows=(
                ("verdict", "false positive"),
                (
                    "score",
                    "—" if match.score is None else f"{match.score:.2f}",
                ),
                ("note", "no matching ground truth"),
            ),
        )
    if v is Verdict.FN:
        return Tooltip(
            title=_fmt_class(match.gt),
            tint=FN_COLOR,
            rows=(("verdict", "false negative"), ("note", "missed")),
        )
    return Tooltip(
        title=f"{_fmt_class(match.gt)} → {_fmt_class(match.pred)}",
        tint=CLASS_ERROR_COLOR,
        rows=(
            ("verdict", "class error"),
            ("IoU", f"{match.iou:.2f}"),
            ("score", "—" if match.score is None else f"{match.score:.2f}"),
            ("gt_class", _fmt_class(match.gt)),
            ("pred_class", _fmt_class(match.pred)),
        ),
    )


def _source_annotations(
    obj: "Detectionish", options: RenderOptions
) -> list[Annotation]:
    """Build the drawable annotations for one detection.

    A `BBox` becomes a single box; an LDF `Detection` becomes its full tree (box,
    keypoints, instance mask, sub-detections) via the LDF adapter, so the whole
    object can be recolored by its verdict.
    """
    if isinstance(obj, BBox):
        box = obj.model_copy(deep=True)
        box.children = []
        box.payload = None
        return [box]
    from .convert import detection_to_annotations

    return detection_to_annotations(obj, options.replace(hover_metadata=False))


def _recolor(
    obj: "Detectionish",
    options: RenderOptions,
    color: Color | None,
    *,
    tooltip: Tooltip | None = None,
    dash: tuple[float, float] | None = None,
    relabel: str | None = None,
    keep_score: bool = True,
) -> list[Annotation]:
    """Draw ``obj`` in ``color`` (``None`` keeps its palette color).

    The hover tooltip, an optional relabel, and a dash pattern are applied to the
    root; nested keypoints/masks derive the root's color and style.
    """
    annotations = _source_annotations(obj, options)
    if not annotations:
        return []
    root = annotations[0]
    root.tooltip = tooltip
    if relabel is not None:
        root.label = relabel
    if not keep_score:
        root.score = None
    overrides = {"dash": dash} if dash is not None else None
    for annotation in annotations:
        if color is not None:
            annotation.color = color
        annotation.style = None
        if overrides is not None:
            annotation.style_overrides = dict(overrides)
    return annotations


def _ghost(
    obj: "Detectionish", color: Color = _GHOST_COLOR
) -> list[Annotation]:
    """Return a faint dashed outline at ``obj``'s bounds (a missing-partner mark)."""
    rect = _bounds(obj)
    if rect is None:
        return []
    box = BBox(x=rect.left, y=rect.top, w=rect.width, h=rect.height)
    box.color = color
    box.style_overrides = {
        "dash": (4.0, 4.0),
        "stroke_width": 1.4,
        "fill_alpha": 0.0,
    }
    return [box]


def _keypoints_of(obj: "Detectionish | None") -> list | None:
    """Index-aligned ``(x, y, visibility)`` keypoints of a detection, or None."""
    if obj is None or isinstance(obj, BBox):
        return None
    keypoints = obj.keypoints
    if keypoints is None or not keypoints.keypoints:
        return None
    return list(keypoints.keypoints)


def _grade_keypoints(
    pred_kps: list, gt_kps: list, bounds: Rect
) -> tuple[list[tuple[float, float, float]], list[Color]]:
    """Grade predicted joints against the ground truth, index by index.

    Returns the points to draw and an index-aligned color per joint: a predicted
    joint is green within `KEYPOINT_TOLERANCE` of its ground-truth counterpart,
    red when off, amber (relocated to the ground-truth spot) when the prediction
    omits a joint the ground truth has, and red when it invents one the ground
    truth lacks.
    """
    tolerance = KEYPOINT_TOLERANCE * (
        math.hypot(bounds.width, bounds.height) or 1.0
    )
    points: list[tuple[float, float, float]] = []
    colors: list[Color] = []
    for i, (px, py, pv) in enumerate(pred_kps):
        gx, gy, gv = gt_kps[i] if i < len(gt_kps) else (px, py, 0)
        if not gv:  # ground truth has no joint here
            points.append((px, py, pv))
            colors.append(FP_COLOR)
        elif not pv:  # ground truth joint the prediction missed
            points.append((gx, gy, 2))
            colors.append(FN_COLOR)
        elif math.hypot(px - gx, py - gy) <= tolerance:
            points.append((px, py, pv))
            colors.append(TP_COLOR)
        else:
            points.append((px, py, pv))
            colors.append(FP_COLOR)
    return points, colors


def _with_keypoint_verdicts(
    annotations: list[Annotation], match: Match
) -> list[Annotation]:
    """Grade a matched detection's joints per keypoint instead of all one color.

    When both sides carry keypoints, the single `Keypoints` keeps its skeleton
    edges but gets per-joint colors (green correct / red wrong / amber missed);
    the box keeps its verdict color. A limb between two differently graded joints
    renders as a gradient. Detections without paired keypoints are unchanged.
    """
    pred_kps = _keypoints_of(match.pred)
    gt_kps = _keypoints_of(match.gt)
    bounds = _bounds(match.pred) if match.pred is not None else None
    if not pred_kps or not gt_kps or bounds is None or not annotations:
        return annotations
    root = annotations[0]
    pose = (
        root
        if isinstance(root, Keypoints)
        else next((c for c in root.children if isinstance(c, Keypoints)), None)
    )
    if pose is None:
        return annotations
    points, colors = _grade_keypoints(pred_kps, gt_kps, bounds)
    # Rebuild the pose with the graded points (keeping its edges/names), then
    # attach the per-joint colors; model_copy skips the invariant-list check.
    graded = pose.model_copy(update={"keypoints": points})
    graded.point_colors = list(colors)
    if pose is root:
        return [graded, *annotations[1:]]
    root.children = [graded if c is pose else c for c in root.children]
    return annotations


def _render_match(match: Match, options: RenderOptions) -> list[Annotation]:
    """Turn one match into the annotations that depict it (verdict-colored)."""
    tooltip = _match_tooltip(match)
    v = match.verdict
    if v is Verdict.TP and match.pred is not None and match.gt is not None:
        painted = _recolor(match.pred, options, TP_COLOR, tooltip=tooltip)
        return [*_ghost(match.gt), *_with_keypoint_verdicts(painted, match)]
    if v is Verdict.FP and match.pred is not None:
        return _recolor(match.pred, options, FP_COLOR, tooltip=tooltip)
    if v is Verdict.FN and match.gt is not None:
        return _recolor(
            match.gt,
            options,
            FN_COLOR,
            tooltip=tooltip,
            dash=(7.0, 5.0),
            keep_score=False,
        )
    if v is Verdict.CLASS_ERROR and match.pred is not None:
        painted = _recolor(
            match.pred,
            options,
            CLASS_ERROR_COLOR,
            tooltip=tooltip,
            relabel=f"{_fmt_class(match.gt)} → {_fmt_class(match.pred)}",
        )
        return _with_keypoint_verdicts(painted, match)
    return []


def _base(image: "ImageSource", options: RenderOptions) -> "Image":
    """Build a fresh `Image` over ``image`` for one comparison panel."""
    from .image import Image

    return Image(image, options=options)


def _overlay_image(
    image: "ImageSource", result: ComparisonResult, options: RenderOptions
) -> "Image":
    """One frame, every detection colored by its verdict (the default view)."""
    img = _base(image, options)
    for match in result.matches:
        for annotation in _render_match(match, options):
            img.add(annotation)
    return img


def _side_by_side_image(
    image: "ImageSource", result: ComparisonResult, options: RenderOptions
) -> "Image":
    """Ground truth beside prediction, color keyed to identity not verdict.

    A matched pair shares one hue across both panels, so the eye tracks a single
    object left to right; an unmatched detection has no twin — its partner shows
    as a faded ghost in the other panel.
    """
    from . import compose

    gt_img = _base(image, options)
    pred_img = _base(image, options)
    for i, match in enumerate(result.matches):
        color = _IDENTITY_COLORS[i % len(_IDENTITY_COLORS)]
        tooltip = _match_tooltip(match)
        if match.gt is not None and match.pred is not None:
            for a in _recolor(match.gt, options, color):
                gt_img.add(a)
            for a in _recolor(match.pred, options, color, tooltip=tooltip):
                pred_img.add(a)
        elif match.gt is not None:  # false negative: no predicted twin
            for a in _recolor(match.gt, options, color, tooltip=tooltip):
                gt_img.add(a)
            for a in _ghost(match.gt, _faded(color)):
                pred_img.add(a)
        elif match.pred is not None:  # false positive: no ground-truth twin
            for a in _recolor(match.pred, options, color, tooltip=tooltip):
                pred_img.add(a)
            for a in _ghost(match.pred, _faded(color)):
                gt_img.add(a)
    # grid_hits composes exactly like hstack but also captures each panel's hover
    # regions, offset into the composite — so tooltips survive the composition.
    frame = compose.grid_hits(
        [gt_img, pred_img], ncols=2, titles=["ground truth", "prediction"]
    )
    return frame.image.with_hitmap(frame.hitmap)


def _triptych_image(
    image: "ImageSource", result: ComparisonResult, options: RenderOptions
) -> "Image":
    """Ground truth, prediction, and the verdict-colored diff, side by side."""
    from . import compose

    gt_img = _base(image, options)
    pred_img = _base(image, options)
    for match in result.matches:
        if match.gt is not None:
            for a in _recolor(match.gt, options, None):
                gt_img.add(a)
        if match.pred is not None:
            for a in _recolor(match.pred, options, None):
                pred_img.add(a)
    diff_img = _overlay_image(image, result, options)
    # grid_hits keeps the diff panel's tooltips (the ground-truth and prediction
    # panels carry none), offset into the composite.
    frame = compose.grid_hits(
        [gt_img, pred_img, diff_img],
        ncols=3,
        titles=["ground truth", "prediction", "diff"],
    )
    return frame.image.with_hitmap(frame.hitmap)


_LAYOUTS = {
    "overlay": _overlay_image,
    "side_by_side": _side_by_side_image,
    "triptych": _triptych_image,
}


def compare(
    image: "ImageSource",
    *,
    gt: "Sequence[Detectionish] | None" = None,
    pred: "Sequence[Detectionish] | None" = None,
    result: ComparisonResult | None = None,
    options: RenderOptions | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
    class_aware: bool = True,
    show: Literal["overlay", "side_by_side", "triptych"] = "overlay",
    panel: bool = True,
    per_class: bool = False,
    verdicts: "Collection[Verdict] | None" = None,
) -> "Image":
    """Render a ground-truth vs prediction comparison over ``image``.

    Pass ``gt`` and ``pred`` to match here, or a precomputed ``result``. Each is a
    sequence of `BBox` or LDF `Detection` objects. The layout is chosen by
    ``show``:

    - ``"overlay"`` — one frame, each detection colored by its verdict (see
      `Verdict`), with a hover `Tooltip` explaining the match.
    - ``"side_by_side"`` — ground truth beside prediction, colored by identity so
      a matched pair shares a hue and errors are the detections with no twin.
    - ``"triptych"`` — ground truth, prediction, and the verdict-colored diff.

    Args:
        image: The base image (any source accepted by `Image`).
        gt: Ground-truth detections (required unless ``result`` is given).
        pred: Predicted detections (required unless ``result`` is given).
        result: A precomputed `ComparisonResult`; skips matching when provided.
        options: Render options (theme, palette); a default is used when ``None``.
        iou_threshold: Overlap threshold for matching (ignored with ``result``).
        score_threshold: Confidence cutoff for predictions (ignored with ``result``).
        class_aware: Whether matching is class-aware (ignored with ``result``).
        show: Layout — ``"overlay"``, ``"side_by_side"``, or ``"triptych"``.
        panel: Whether to append the metrics side panel.
        per_class: Whether to add a per-class breakdown to the panel (needs at
            least two classes to appear).
        verdicts: If given, only detections with one of these verdicts are
            *drawn* (e.g. ``{Verdict.FP, Verdict.FN, Verdict.CLASS_ERROR}`` to
            review only mistakes); the metrics panel still reflects every match.

    Returns:
        A new visualization `Image`, hoverable via `Image.render_hits` for every
        layout — the multi-panel layouts compose their panels' hover regions into
        the result rather than dropping them.

    Raises:
        ValueError: If ``show`` is unknown, or neither ``result`` nor both ``gt``
            and ``pred`` are given.

    """
    if show not in _LAYOUTS:
        raise ValueError(
            f"compare(show={show!r}) is invalid; expected one of "
            f"{', '.join(map(repr, _LAYOUTS))}"
        )
    if result is None:
        if gt is None or pred is None:
            raise ValueError("compare() needs gt and pred, or a result")
        result = match_detections(
            gt,
            pred,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            class_aware=class_aware,
        )

    # Filtering affects only what is *drawn*; the panel keeps the full metrics.
    drawn = result
    if verdicts is not None:
        kept = tuple(m for m in result.matches if m.verdict in verdicts)
        drawn = ComparisonResult(
            kept, result.iou_threshold, result.score_threshold
        )

    img = _LAYOUTS[show](image, drawn, options or RenderOptions())
    if panel:
        data: dict[str, PanelData] = {**result.summary()}
        if per_class and len(result.per_class) > 1:
            data["by class"] = result.per_class_panel()
        img = img.with_panel(data, title="Comparison")
    return img


# --------------------------------------------------------------------------- #
# Dataset-level aggregation
# --------------------------------------------------------------------------- #
#: Row/column label for "no detection" in a confusion matrix (a miss / a false
#: alarm has nothing on the other side).
NONE_LABEL = "∅"
_WHITE = Color(255, 255, 255)
_CM_TEXT = Color(202, 212, 226)


@dataclass
class ComparisonReport:
    """Dataset-wide accumulation of comparison results across many images.

    Feed each image's result to `add`; the report tracks aggregate counts, a
    per-class breakdown, a detection confusion matrix (with a `NONE_LABEL`
    row/column for misses and false alarms), running mean IoU, and the worst
    images by error count. Counts follow COCO: a class error is one false
    positive (its predicted class) plus one false negative (its true class).
    """

    n_images: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0
    _class: dict[str, list[int]] = field(default_factory=dict)
    _confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    _iou_sum: float = 0.0
    _n_localized: int = 0
    _worst: list[tuple[int, str]] = field(default_factory=list)

    def add(
        self, result: ComparisonResult, *, name: str | None = None
    ) -> None:
        """Accumulate one image's ``result`` (optionally tagged ``name``)."""
        self.n_images += 1
        self.n_tp += result.n_tp
        self.n_fp += result.n_fp
        self.n_fn += result.n_fn
        n_errors = 0
        for m in result.matches:
            gt = _fmt_class(m.gt) if m.gt is not None else NONE_LABEL
            pred = _fmt_class(m.pred) if m.pred is not None else NONE_LABEL
            if m.verdict is Verdict.TP:
                self._bump(gt, 0)
                self._confuse(gt, gt)
                self._iou_sum += m.iou
                self._n_localized += 1
            elif m.verdict is Verdict.FP:
                self._bump(pred, 1)
                self._confuse(NONE_LABEL, pred)
                n_errors += 1
            elif m.verdict is Verdict.FN:
                self._bump(gt, 2)
                self._confuse(gt, NONE_LABEL)
                n_errors += 1
            else:  # class error
                self._bump(pred, 1)
                self._bump(gt, 2)
                self._confuse(gt, pred)
                self._iou_sum += m.iou
                self._n_localized += 1
                n_errors += 1
        if name is not None and n_errors:
            self._worst.append((n_errors, name))

    def _bump(self, label: str, slot: int) -> None:
        self._class.setdefault(label, [0, 0, 0])[slot] += 1

    def _confuse(self, gt: str, pred: str) -> None:
        self._confusion[(gt, pred)] = self._confusion.get((gt, pred), 0) + 1

    @property
    def precision(self) -> float:
        """``TP / (TP + FP)`` over the whole dataset."""
        denom = self.n_tp + self.n_fp
        return self.n_tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """``TP / (TP + FN)`` over the whole dataset."""
        denom = self.n_tp + self.n_fn
        return self.n_tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of `precision` and `recall`."""
        p, r = self.precision, self.recall
        return 2.0 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        """Mean overlap over every localized match in the dataset."""
        return self._iou_sum / self._n_localized if self._n_localized else 0.0

    def classes(self) -> list[str]:
        """Sorted class names seen (excluding the `NONE_LABEL` background)."""
        names = {n for n in self._class if n != NONE_LABEL}
        for gt, pred in self._confusion:
            names.discard(NONE_LABEL)
            if gt != NONE_LABEL:
                names.add(gt)
            if pred != NONE_LABEL:
                names.add(pred)
        return sorted(names)

    def per_class(self) -> dict[str, dict[str, float | int]]:
        """Per-class ``{precision, recall, tp, fp, fn}`` across the dataset."""
        out: dict[str, dict[str, float | int]] = {}
        for name, (tp, fp, fn) in sorted(self._class.items()):
            if name == NONE_LABEL:
                continue
            out[name] = {
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return out

    def confusion_matrix(self) -> tuple[list[str], list[list[int]]]:
        """Return ``(labels, matrix)`` with ``matrix[gt][pred]`` counts.

        ``labels`` are the classes followed by `NONE_LABEL`; a row is the ground
        truth, a column the prediction. The ``(NONE, NONE)`` cell is always zero.
        """
        labels = [*self.classes(), NONE_LABEL]
        matrix = [
            [self._confusion.get((row, col), 0) for col in labels]
            for row in labels
        ]
        return labels, matrix

    def worst(self, n: int = 10) -> list[tuple[int, str]]:
        """Return the ``n`` images with the most errors as ``(count, name)``."""
        return sorted(self._worst, reverse=True)[:n]

    def summary(self) -> dict[str, str | int]:
        """Aggregate metrics as a display-ready mapping."""
        return {
            "images": self.n_images,
            "precision": f"{self.precision:.3f}",
            "recall": f"{self.recall:.3f}",
            "F1": f"{self.f1:.3f}",
            "mean IoU": f"{self.mean_iou:.3f}",
            "TP": self.n_tp,
            "FP": self.n_fp,
            "FN": self.n_fn,
        }


# --------------------------------------------------------------------------- #
# Confusion matrix figure
# --------------------------------------------------------------------------- #
def _draw_text(
    canvas: "Canvas",
    text: str,
    x: float,
    y: float,
    size: float,
    color: Color,
    *,
    anchor: str = "center",
    weight: int = 600,
) -> None:
    """Draw ``text`` at ``(x, y)`` anchored center/left/right and vertically mid."""
    metrics = canvas.measure_text(text, size, weight=weight)
    if anchor == "center":
        tx = x - metrics.width / 2
    elif anchor == "right":
        tx = x - metrics.width
    else:
        tx = x
    ty = y + (metrics.ascent - metrics.descent) / 2
    canvas.text((tx, ty), text, size=size, color=color, weight=weight)


def confusion_matrix_figure(
    report: ComparisonReport,
    *,
    options: RenderOptions | None = None,
    cell: int = 60,
) -> "Image":
    """Render ``report``'s confusion matrix as a labeled heatmap `Image`.

    Rows are the ground-truth class, columns the predicted class, with a trailing
    ``∅`` row/column for false negatives (missed) and false positives (invented).
    A cell on the diagonal is green (correct), off-diagonal red (a confusion),
    each shaded by how many detections land in it, with the count printed inside.

    Args:
        report: The accumulated `ComparisonReport`.
        options: Render options, for the background/theme; a default is used when
            ``None``.
        cell: Cell size in pixels.

    Returns:
        A new `Image` of the matrix.

    """
    from .canvas import Canvas
    from .image import Image

    options = options or RenderOptions()
    labels, matrix = report.confusion_matrix()
    n = len(labels)
    shown = [lbl if len(lbl) <= 9 else lbl[:8] + "…" for lbl in labels]
    left, top, pad, foot = 100, 70, 16, 28
    width = left + n * cell + pad
    height = top + n * cell + pad + foot

    bg = options.theme.background
    raster = np.empty((height, width, 4), np.uint8)
    raster[:] = (bg.r, bg.g, bg.b, 255)
    canvas = Canvas.from_rgba(raster)

    peak = max((c for row in matrix for c in row), default=0) or 1
    grid = Color(255, 255, 255, 26)
    for i in range(n):
        for j in range(n):
            count = matrix[i][j]
            x0, y0 = float(left + j * cell), float(top + i * cell)
            x1, y1 = x0 + cell, y0 + cell
            corners: list[XY] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            fill = None
            if count:
                base = (
                    TP_COLOR
                    if (i == j and labels[i] != NONE_LABEL)
                    else FP_COLOR
                )
                fill = base.with_alpha(0.18 + 0.72 * (count / peak))
            canvas.polygon(corners, fill=fill, stroke=grid, stroke_width=1.0)
            if count:
                _draw_text(
                    canvas,
                    str(count),
                    x0 + cell / 2,
                    y0 + cell / 2,
                    13.0,
                    _WHITE,
                    weight=700,
                )

    for j, name in enumerate(shown):
        _draw_text(
            canvas, name, left + j * cell + cell / 2, top - 14, 12.0, _CM_TEXT
        )
    for i, name in enumerate(shown):
        _draw_text(
            canvas,
            name,
            left - 10,
            top + i * cell + cell / 2,
            12.0,
            _CM_TEXT,
            anchor="right",
        )
    _draw_text(
        canvas,
        "truth ↓  /  prediction →",
        8,
        22,
        13.0,
        _CM_TEXT,
        anchor="left",
        weight=700,
    )
    return Image(canvas.to_rgba())
