"""Draw a `ComparisonResult` — the view where colour *is* the verdict.

Green for a hit, red for a false alarm, dashed amber for a miss, and orange for a
box that landed on the right object with the wrong label. A `Detection`'s whole
annotation tree (box, keypoints, instance mask, sub-detections) is drawn in its
verdict colour, so masks and keypoints inherit the outcome.

When a matched pair carries keypoints they are graded per joint — green within
`KEYPOINT_TOLERANCE` of the ground-truth joint, red when off, amber when missed —
so a partly-correct pose reads at a glance, and a skeleton limb between two
differently graded joints fades from one colour to the other.
"""

import math
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Literal, overload

from luxonis_ml.vizlab.annotations import Annotation, BBox, Keypoints
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.layout.panel import with_panel
from luxonis_ml.vizlab.options import RenderOptions
from luxonis_ml.vizlab.style import StyleValue
from luxonis_ml.vizlab.tooltip import Tooltip

from .match import (
    CLASS_ERROR_COLOR,
    FN_COLOR,
    FP_COLOR,
    TP_COLOR,
    ComparisonKeypoint,
    ComparisonResult,
    Match,
    Verdict,
    _bounds,
    _fmt_class,
    match_detections,
)

if TYPE_CHECKING:
    from luxonis_ml.ldf import Detection
    from luxonis_ml.vizlab.io import ImageSource
    from luxonis_ml.vizlab.layout.panel import PanelData
    from luxonis_ml.vizlab.scene.image import Image, Renderable

    #: A matchable detection: a vizlab box or a full LDF detection tree.
    Detectionish = BBox | Detection

#: Verdict colors, tuned for the dark composite background. Themeable later.

_GHOST_COLOR = Color(234, 241, 248, 110)


_IDENTITY_COLORS = (
    Color.parse("#38bdf8"),  # cyan
    Color.parse("#a78bfa"),  # violet
    Color.parse("#2dd4bf"),  # teal
    Color.parse("#f472b6"),  # pink
    Color.parse("#a3e635"),  # lime
    Color.parse("#e879f9"),  # fuchsia
)


KEYPOINT_TOLERANCE = 0.1


def _faded(color: Color, alpha: int = 120) -> Color:
    """Return ``color`` at a lower opacity, for a "missing partner" ghost."""
    return Color(color.r, color.g, color.b, alpha)


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
    from luxonis_ml.vizlab.adapters.ldf import detection_to_annotations

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
    overrides: dict[str, StyleValue] | None = (
        {"dash": dash} if dash is not None else None
    )
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


def _keypoints_of(
    obj: "Detectionish | None",
) -> list[ComparisonKeypoint] | None:
    """Index-aligned ``(x, y, visibility)`` keypoints of a detection, or None."""
    if obj is None or isinstance(obj, BBox):
        return None
    keypoints = obj.keypoints
    if keypoints is None or not keypoints.keypoints:
        return None
    return [
        (float(x), float(y), int(visibility))
        for x, y, visibility in keypoints.keypoints
    ]


def _grade_keypoints(
    pred_kps: Sequence[ComparisonKeypoint],
    gt_kps: Sequence[ComparisonKeypoint],
    bounds: Rect,
) -> tuple[list[ComparisonKeypoint], list[Color]]:
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
    points: list[ComparisonKeypoint] = []
    colors: list[Color] = []
    for i in range(max(len(pred_kps), len(gt_kps))):
        gx, gy, gv = gt_kps[i] if i < len(gt_kps) else (0.0, 0.0, 0)
        if i < len(pred_kps):
            px, py, pv = pred_kps[i]
        else:
            # The prediction is shorter than the ground truth, so it has no joint
            # at this index at all. Stand in an unlabeled joint at the
            # ground-truth spot: a joint the ground truth does have then grades
            # as missed instead of going ungraded.
            px, py, pv = gx, gy, 0
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
    from luxonis_ml.vizlab.scene.image import Image

    return Image(image, options=options)


def _overlay_image(
    image: "ImageSource", result: ComparisonResult, options: RenderOptions
) -> "Image":
    """One frame, every detection colored by its verdict (the default view)."""
    img = _base(image, options)
    for match in result.matches:
        _add_annotations(img, _render_match(match, options))
    return img


def _add_annotations(
    image: "Image", annotations: Sequence[Annotation]
) -> None:
    """Add ``annotations`` to ``image`` in order."""
    for annotation in annotations:
        image.add(annotation)


def _add_identity_match(
    gt_image: "Image",
    pred_image: "Image",
    match: Match,
    color: Color,
    options: RenderOptions,
) -> None:
    """Paint one identity-colored match across the two comparison panels."""
    tooltip = _match_tooltip(match)
    if match.gt is not None:
        gt_tooltip = tooltip if match.pred is None else None
        _add_annotations(
            gt_image,
            _recolor(match.gt, options, color, tooltip=gt_tooltip),
        )
    if match.pred is not None:
        _add_annotations(
            pred_image,
            _recolor(match.pred, options, color, tooltip=tooltip),
        )
    if match.gt is not None and match.pred is None:
        _add_annotations(pred_image, _ghost(match.gt, _faded(color)))
    if match.pred is not None and match.gt is None:
        _add_annotations(gt_image, _ghost(match.pred, _faded(color)))


def _side_by_side_image(
    image: "ImageSource", result: ComparisonResult, options: RenderOptions
) -> "Renderable":
    """Ground truth beside prediction, color keyed to identity not verdict.

    A matched pair shares one hue across both panels, so the eye tracks a single
    object left to right; an unmatched detection has no twin — its partner shows
    as a faded ghost in the other panel.
    """
    from luxonis_ml.vizlab.layout import compose

    gt_img = _base(image, options)
    pred_img = _base(image, options)
    for i, match in enumerate(result.matches):
        color = _IDENTITY_COLORS[i % len(_IDENTITY_COLORS)]
        _add_identity_match(gt_img, pred_img, match, color, options)
    return compose.grid(
        [gt_img, pred_img], ncols=2, titles=["ground truth", "prediction"]
    )


def _triptych_image(
    image: "ImageSource", result: ComparisonResult, options: RenderOptions
) -> "Renderable":
    """Ground truth, prediction, and the verdict-colored diff, side by side."""
    from luxonis_ml.vizlab.layout import compose

    gt_img = _base(image, options)
    pred_img = _base(image, options)
    for match in result.matches:
        if match.gt is not None:
            _add_annotations(gt_img, _recolor(match.gt, options, None))
        if match.pred is not None:
            _add_annotations(pred_img, _recolor(match.pred, options, None))
    diff_img = _overlay_image(image, result, options)
    return compose.grid(
        [gt_img, pred_img, diff_img],
        ncols=3,
        titles=["ground truth", "prediction", "diff"],
    )


_LAYOUTS = {
    "overlay": _overlay_image,
    "side_by_side": _side_by_side_image,
    "triptych": _triptych_image,
}


@overload
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
    show: Literal["overlay"] = "overlay",
    panel: Literal[False],
    per_class: bool = False,
    verdicts: "Collection[Verdict] | None" = None,
) -> "Image": ...


@overload
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
) -> "Renderable": ...


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
) -> "Renderable":
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
        img = with_panel(img, data, title="Comparison")
    return img
