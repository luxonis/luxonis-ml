"""Match predictions to ground truth, and score the result.

Pure logic: no image, no canvas, nothing drawn. Matching mirrors COCO —
predictions are taken in descending confidence order and each greedily claims the
best-overlapping unclaimed ground-truth box of the *same* class. Metrics are
counted the COCO way, so a wrong-label prediction is a false positive *and* the
ground-truth box it covered is a false negative.

The orange `Verdict.CLASS_ERROR` is a second, cosmetic pass over those
leftovers: it marks the pair so the mistake is visible, but it still scores as
one false positive plus one false negative, so the numbers never move.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from luxonis_ml.vizlab.annotations import BBox
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect, bounding_rect

if TYPE_CHECKING:
    from luxonis_ml.ldf import Detection

    #: A matchable detection: a vizlab box or a full LDF detection tree.
    Detectionish = BBox | Detection

# Verdict colors, tuned for the dark composite background. Themeable later.
TP_COLOR = Color.parse("#35d6a6")
# A miss outranks a false alarm: a missed object can never be recovered
# downstream, while a false alarm is at least visible. So red marks the miss.
FN_COLOR = Color.parse("#ff6b6b")
FP_COLOR = Color.parse("#ffc24b")
CLASS_ERROR_COLOR = Color.parse("#ff9142")

ComparisonKeypoint = tuple[float, float, int]


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


def _metric_row(counts: Sequence[int]) -> dict[str, float | int]:
    """Build precision/recall and raw counts from ``(tp, fp, fn)``."""
    tp, fp, fn = counts
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _class_tally_updates(
    match: Match,
) -> list[tuple[str | None, int]]:
    """Return the class/slot increments contributed by ``match``."""
    if match.verdict is Verdict.CLASS_ERROR:
        candidates = ((match.pred, 1), (match.gt, 2))
    else:
        field_name, slot = {
            Verdict.TP: ("gt", 0),
            Verdict.FP: ("pred", 1),
            Verdict.FN: ("gt", 2),
        }[match.verdict]
        candidates = ((getattr(match, field_name), slot),)
    return [
        (_label(detection), slot)
        for detection, slot in candidates
        if detection is not None
    ]


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
        for match in self.matches:
            for label, slot in _class_tally_updates(match):
                tally.setdefault(label or "object", [0, 0, 0])[slot] += 1
        return {
            name: _metric_row(counts) for name, counts in sorted(tally.items())
        }

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


def _label(obj: "Detectionish") -> str | None:
    """Return the class label of a box or detection."""
    return obj.label if isinstance(obj, BBox) else obj.class_name


def _score(obj: "Detectionish") -> float | None:
    """Return a native score or one preserved as LDF detection metadata."""
    if isinstance(obj, BBox):
        return obj.score
    for key in ("score", "confidence"):
        value = obj.metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


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
        visible = [(p[0], p[1]) for p in keypoints.keypoints if p[2] > 0]
        return bounding_rect(visible) if visible else None
    return None


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


def _matchable(
    detections: "Sequence[Detectionish]",
) -> "list[tuple[Detectionish, Rect]]":
    """Pair detections that have matchable bounds with those bounds."""
    bounded = []
    for detection in detections:
        rect = _bounds(detection)
        if rect is not None:
            bounded.append((detection, rect))
    return bounded


def _ranked_predictions(
    predictions: "Sequence[Detectionish]", score_threshold: float
) -> "list[tuple[Detectionish, Rect, float]]":
    """Return matchable predictions above the score cutoff, highest score first.

    A prediction without a score carries no confidence to threshold or rank on,
    so it is kept but queued behind every scored one (in its given order) rather
    than treated as a perfect-confidence detection that claims ground truth
    first.
    """
    ranked: list[tuple[Detectionish, Rect, float]] = []
    unscored: list[tuple[Detectionish, Rect, float]] = []
    for prediction, rect in _matchable(predictions):
        score = _score(prediction)
        if score is None:
            unscored.append((prediction, rect, 0.0))
        elif score >= score_threshold:
            ranked.append((prediction, rect, score))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked + unscored


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
    descending confidence order, with scoreless detections last in their given
    order; each greedily claims the highest-IoU unclaimed ground-truth box (of
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
    gts = _matchable(gt)
    ranked = _ranked_predictions(pred, score_threshold)

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


def _fmt_class(obj: "Detectionish | None") -> str:
    """Human-readable class name for a detection, defaulting to ``object``."""
    label = _label(obj) if obj is not None else None
    return label or "object"


NONE_LABEL = "∅"
