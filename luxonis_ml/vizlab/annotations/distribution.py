"""Classification predictions as a probability distribution over classes.

`Classification` renders dataset *labels* — the one (or few) correct classes as
corner chips. A model *prediction* is different: a probability distribution over
all classes. `ClassDistribution` renders that distribution, with four
interchangeable looks (``mode``) and an optional ground-truth marker so you can
see at a glance whether the top prediction is correct.

Like `Heatmap`, this is a caller-supplied construct: LDF datasets carry a single
class name, not a probability vector, so the scores come from the prediction side
(e.g. a softmax output paired with ``dataset.get_class_names()``), not from the
dataset path. It is an image-level corner overlay, built on the same
`CornerStack` machinery as `Legend` and `Classification`.
"""

from collections.abc import Mapping, Sequence
from typing import Literal

from luxonis_ml.vizlab.canvas import Canvas, Shadow, TextMetrics
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.style import Style

from .base import RenderContext
from .chip import chip_size, compose_label, draw_chip
from .overlay import CARD_BG, CARD_TEXT, Cell, CornerStack

DistributionMode = Literal["bars", "chips", "gauge", "stacked"]
"""How a `ClassDistribution` is drawn. See the class docstring."""

_WHITE = Color(255, 255, 255)
_OK = Color(80, 200, 120)
_BAD = Color(230, 90, 90)
_OTHER = Color(96, 96, 104)

_PAD = 10.0
_ROW_GAP = 6.0
_COL_GAP = 8.0


def _clamp01(value: float) -> float:
    """Clamp a probability into ``[0, 1]``."""
    return max(0.0, min(1.0, value))


def _pct(prob: float) -> str:
    """Format a probability as a whole-percent string, e.g. ``"92%"``."""
    return f"{round(_clamp01(prob) * 100)}%"


class ClassDistribution(CornerStack):
    """A predicted probability distribution over classes, in one of four looks.

    Feed it a distribution (a ``{name: probability}`` mapping, or ``(name, prob)``
    pairs) and it draws a corner panel. ``mode`` selects the look:

    - ``"bars"`` (default): a ranked horizontal bar chart — class name, a bar whose
      length is the probability, and the percentage. Best general-purpose view.
    - ``"chips"``: compact colored chips with percentages (the `Classification`
      tag look), stacked in the corner.
    - ``"gauge"``: just the winning class as a single large confidence bar.
    - ``"stacked"``: one segmented strip showing the classes as proportions, with
      an inline key.

    Each class is colored from the palette by name, so it matches its box color.
    Set ``ground_truth`` to mark the correct class (a highlighted row/chip/segment,
    or a ✓/✗ on the gauge); if the true class falls outside ``top_k`` it is still
    shown, so a missed prediction stays visible.

    Attributes:
        probabilities: The distribution as a ``{name: prob}`` mapping or a list of
            ``(name, prob)`` pairs. Probabilities are expected in ``[0, 1]``.
        mode: Which of the four looks to draw (see above).
        ground_truth: The correct class name to highlight, or ``None``.
        top_k: Show only the ``top_k`` most probable classes (plus the ground-truth
            row if it is set and would otherwise be hidden). ``None`` shows all.
        title: Optional heading drawn above the panel (ignored in ``"chips"``).

    See `CornerStack` for ``corner``/``margin``/``gap`` and
    `Annotation` for ``style``/``palette``.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import ClassDistribution, Image
        >>> dist = ClassDistribution(
        ...     probabilities={"cat": 0.82, "dog": 0.11, "fox": 0.07},
        ...     ground_truth="dog",
        ... )
        >>> Image(np.zeros((90, 160, 3), np.uint8)).add(dist).render().shape
        (90, 160, 4)

    """

    probabilities: Mapping[str, float] | list[tuple[str, float]] = {}
    mode: DistributionMode = "bars"
    ground_truth: str | None = None
    top_k: int | None = 5
    title: str | None = None

    @classmethod
    def from_scores(
        cls,
        class_names: Sequence[str],
        scores: Sequence[float],
        **kwargs: object,
    ) -> "ClassDistribution":
        """Build a distribution from an ordered score vector and its class names.

        This is the prediction-side constructor: pair a model's ``(C,)`` output
        with the class names in the same order (e.g. from
        ``dataset.get_class_names()``).

        Args:
            class_names: Class names, in the score vector's order.
            scores: One probability per class, aligned to ``class_names``.
            **kwargs: Forwarded to the constructor (``mode``, ``ground_truth``,
                ``top_k``, ``title``, ``corner``, ``palette``, ...).

        Returns:
            The `ClassDistribution`.

        """
        pairs = [
            (str(name), float(score))
            for name, score in zip(class_names, scores, strict=True)
        ]
        return cls(probabilities=pairs, **kwargs)  # type: ignore[arg-type]

    # -- distribution selection ---------------------------------------------

    def _pairs(self) -> list[tuple[str, float]]:
        """Return the distribution as ``(name, prob)`` pairs, mapping or list."""
        probs = self.probabilities
        if isinstance(probs, Mapping):
            return [(str(name), float(p)) for name, p in probs.items()]
        return [(str(name), float(p)) for name, p in probs]

    def _ranked(self) -> list[tuple[str, float]]:
        """Return the distribution sorted by probability, most likely first."""
        return sorted(self._pairs(), key=lambda kv: kv[1], reverse=True)

    def _selected(self) -> list[tuple[str, float]]:
        """Return the ``top_k`` rows, appending the ground-truth row if hidden."""
        ranked = self._ranked()
        if self.top_k is None:
            return ranked
        top = ranked[: self.top_k]
        if self.ground_truth is not None and all(
            name != self.ground_truth for name, _ in top
        ):
            hidden = next(
                (kv for kv in ranked if kv[0] == self.ground_truth), None
            )
            if hidden is not None:
                top = [*top, hidden]
        return top

    # -- dispatch ------------------------------------------------------------

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        if self.mode == "chips":
            return self._chips_cells(ctx, style)
        if self.mode == "gauge":
            return self._gauge_cells(ctx, style)
        if self.mode == "stacked":
            return self._stacked_cells(ctx, style)
        return self._bars_cells(ctx, style)

    def _card_bg(self, cv: Canvas, rect: Rect, style: Style) -> None:
        """Paint the shared rounded card background."""
        cv.rounded_rect(
            rect,
            radius=9.0,
            fill=CARD_BG,
            shadow=Shadow(blur=6.0, dy=2.0) if style.shadow else None,
        )

    def _title_metrics(
        self, canvas: Canvas, size: float
    ) -> TextMetrics | None:
        """Measure the optional title, or ``None`` when there is none."""
        if self.title is None:
            return None
        return canvas.measure_text(self.title, size * 1.05, weight=700)

    def _draw_title(self, cv: Canvas, x: float, y: float, size: float) -> None:
        """Draw the title at baseline; assumes there is one."""
        metrics = cv.measure_text(str(self.title), size * 1.05, weight=700)
        cv.text(
            (x, y + metrics.ascent),
            str(self.title),
            size=size * 1.05,
            color=CARD_TEXT,
            weight=700,
        )

    # -- bars ----------------------------------------------------------------

    def _bars_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        size, weight = style.font_size, style.font_weight
        rows = self._selected()
        if not rows:
            return []

        bar_w = size * 7.0
        bar_h = size * 0.72
        measured = [
            (name, prob, canvas.measure_text(name, size, weight=weight))
            for name, prob in rows
        ]
        pct_w = canvas.measure_text("100%", size, weight=weight).width
        name_w = max(m.width for _, _, m in measured)
        row_h = max(bar_h, *(m.height for _, _, m in measured))
        title_metrics = self._title_metrics(canvas, size)
        title_h = title_metrics.height + _ROW_GAP if title_metrics else 0.0

        content_w = name_w + _COL_GAP + bar_w + _COL_GAP + pct_w
        if title_metrics is not None:
            content_w = max(content_w, title_metrics.width)
        card_w = content_w + 2 * _PAD
        card_h = (
            2 * _PAD + title_h + len(rows) * row_h + _ROW_GAP * (len(rows) - 1)
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            self._card_bg(cv, rect, style)
            y = rect.top + _PAD
            if title_metrics is not None:
                self._draw_title(cv, rect.left + _PAD, y, size)
                y += title_h
            bar_x = rect.left + _PAD + name_w + _COL_GAP
            pct_x = bar_x + bar_w + _COL_GAP
            for name, prob, m in measured:
                is_gt = name == self.ground_truth
                color = palette.color_for(name)
                cv.text(
                    (rect.left + _PAD, y + (row_h - m.height) / 2 + m.ascent),
                    name,
                    size=size,
                    color=CARD_TEXT,
                    weight=700 if is_gt else weight,
                )
                track_top = y + (row_h - bar_h) / 2
                track = Rect(
                    bar_x, track_top, bar_x + bar_w, track_top + bar_h
                )
                cv.rounded_rect(
                    track, radius=bar_h / 2, fill=color.with_alpha(0.2)
                )
                fill_w = bar_w * _clamp01(prob)
                if fill_w > 0:
                    cv.rounded_rect(
                        Rect(
                            bar_x, track_top, bar_x + fill_w, track_top + bar_h
                        ),
                        radius=bar_h / 2,
                        fill=color,
                    )
                if is_gt:
                    cv.rounded_rect(
                        track,
                        radius=bar_h / 2,
                        stroke=_WHITE,
                        stroke_width=1.5,
                    )
                pct = _pct(prob)
                pm = cv.measure_text(pct, size, weight=weight)
                cv.text(
                    (pct_x, y + (row_h - pm.height) / 2 + pm.ascent),
                    pct,
                    size=size,
                    color=CARD_TEXT,
                    weight=weight,
                )
                y += row_h + _ROW_GAP

        return [Cell(card_w, card_h, _draw)]

    # -- chips ---------------------------------------------------------------

    def _chips_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        cells: list[Cell] = []
        for name, prob in self._selected():
            text = compose_label(name, prob, None)
            if not text:
                continue
            color = palette.color_for(name)
            is_gt = name == self.ground_truth
            cells.append(self._chip_cell(canvas, text, color, style, is_gt))
        return cells

    def _chip_cell(
        self, canvas: Canvas, text: str, color: Color, style: Style, gt: bool
    ) -> Cell:
        """Build one scored chip cell; the ground-truth chip gets an outline."""
        width, height, _ = chip_size(canvas, text, style)

        def _draw(cv: Canvas, rect: Rect) -> None:
            draw_chip(cv, (rect.left, rect.top), text, color, style)
            if gt:
                cv.rounded_rect(
                    rect,
                    radius=style.label_radius,
                    stroke=_WHITE,
                    stroke_width=2.0,
                )

        return Cell(width, height, _draw)

    # -- gauge ---------------------------------------------------------------

    def _gauge_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        ranked = self._ranked()
        if not ranked:
            return []
        name, prob = ranked[0]
        color = palette.color_for(name)
        size = style.font_size
        big = size * 1.5
        bar_h = size * 0.9

        name_m = canvas.measure_text(name, size, weight=700)
        pct_text = _pct(prob)
        pct_m = canvas.measure_text(pct_text, big, weight=700)
        gt = self.ground_truth
        correct = gt is not None and gt == name
        marker = size * 1.1 if gt is not None else 0.0
        header_h = max(name_m.height, pct_m.height, marker)
        content_w = max(
            size * 9.0,
            name_m.width + _COL_GAP + pct_m.width + (marker + 6 if gt else 0),
        )
        card_w = content_w + 2 * _PAD
        card_h = 2 * _PAD + header_h + 8.0 + bar_h

        def _draw(cv: Canvas, rect: Rect) -> None:
            self._card_bg(cv, rect, style)
            y = rect.top + _PAD
            cv.text(
                (
                    rect.left + _PAD,
                    y + (header_h - name_m.height) / 2 + name_m.ascent,
                ),
                name,
                size=size,
                color=CARD_TEXT,
                weight=700,
            )
            right = rect.right - _PAD
            if gt is not None:
                r = size * 0.55
                cx, cy = right - r, y + header_h / 2
                _draw_verdict(cv, cx, cy, r, correct)
                right = cx - r - 6.0
            cv.text(
                (
                    right - pct_m.width,
                    y + (header_h - pct_m.height) / 2 + pct_m.ascent,
                ),
                pct_text,
                size=big,
                color=CARD_TEXT,
                weight=700,
            )
            by = rect.top + _PAD + header_h + 8.0
            left, right_edge = rect.left + _PAD, rect.right - _PAD
            track = Rect(left, by, right_edge, by + bar_h)
            cv.rounded_rect(
                track, radius=bar_h / 2, fill=color.with_alpha(0.2)
            )
            fill_w = (right_edge - left) * _clamp01(prob)
            if fill_w > 0:
                cv.rounded_rect(
                    Rect(left, by, left + fill_w, by + bar_h),
                    radius=bar_h / 2,
                    fill=color,
                )

        return [Cell(card_w, card_h, _draw)]

    # -- stacked -------------------------------------------------------------

    def _stacked_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        size, weight = style.font_size, style.font_weight
        segs = self._selected()
        if not segs:
            return []

        other = max(0.0, 1.0 - sum(_clamp01(p) for _, p in segs))
        keys = [*segs, ("other", other)] if other > 0.005 else list(segs)

        strip_w = size * 14.0
        strip_h = size * 1.1
        swatch = size
        key_measured = [
            (
                name,
                prob,
                canvas.measure_text(
                    f"{name}  {_pct(prob)}", size, weight=weight
                ),
            )
            for name, prob in keys
        ]
        row_h = max(m.height for _, _, m in key_measured)
        key_w = max(swatch + _COL_GAP + m.width for _, _, m in key_measured)
        title_metrics = self._title_metrics(canvas, size)
        title_h = title_metrics.height + _ROW_GAP if title_metrics else 0.0

        content_w = max(strip_w, key_w)
        if title_metrics is not None:
            content_w = max(content_w, title_metrics.width)
        card_w = content_w + 2 * _PAD
        card_h = (
            2 * _PAD
            + title_h
            + strip_h
            + _ROW_GAP
            + len(key_measured) * row_h
            + _ROW_GAP * (len(key_measured) - 1)
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            self._card_bg(cv, rect, style)
            y = rect.top + _PAD
            if title_metrics is not None:
                self._draw_title(cv, rect.left + _PAD, y, size)
                y += title_h
            # Strip: a muted backdrop (the unshown "other" mass), then each class
            # segment laid left to right, proportional to its probability.
            left = rect.left + _PAD
            inner_w = content_w
            cv.rounded_rect(
                Rect(left, y, left + inner_w, y + strip_h),
                radius=4.0,
                fill=_OTHER,
            )
            seg_x = left
            for name, prob in segs:
                seg_w = inner_w * _clamp01(prob)
                if seg_w <= 0:
                    continue
                seg = Rect(seg_x, y, seg_x + seg_w, y + strip_h)
                cv.rounded_rect(seg, radius=0.0, fill=palette.color_for(name))
                if name == self.ground_truth:
                    cv.rounded_rect(
                        seg, radius=0.0, stroke=_WHITE, stroke_width=2.0
                    )
                seg_x += seg_w
            y += strip_h + _ROW_GAP
            # Inline key.
            for name, prob, m in key_measured:
                color = _OTHER if name == "other" else palette.color_for(name)
                sw_top = y + (row_h - swatch) / 2
                cv.rounded_rect(
                    Rect(
                        rect.left + _PAD,
                        sw_top,
                        rect.left + _PAD + swatch,
                        sw_top + swatch,
                    ),
                    radius=3.0,
                    fill=color,
                )
                cv.text(
                    (rect.left + _PAD + swatch + _COL_GAP, y + m.ascent),
                    f"{name}  {_pct(prob)}",
                    size=size,
                    color=CARD_TEXT,
                    weight=700 if name == self.ground_truth else weight,
                )
                y += row_h + _ROW_GAP

        return [Cell(card_w, card_h, _draw)]


def _draw_verdict(
    cv: Canvas, cx: float, cy: float, r: float, correct: bool
) -> None:
    """Draw a filled ✓ (correct) or ✗ (wrong) badge centered at ``(cx, cy)``."""
    cv.circle((cx, cy), r, fill=_OK if correct else _BAD)
    w = max(1.5, r * 0.28)
    if correct:
        cv.line(
            (cx - 0.45 * r, cy + 0.02 * r),
            (cx - 0.1 * r, cy + 0.38 * r),
            _WHITE,
            w,
        )
        cv.line(
            (cx - 0.1 * r, cy + 0.38 * r),
            (cx + 0.5 * r, cy - 0.4 * r),
            _WHITE,
            w,
        )
    else:
        cv.line(
            (cx - 0.38 * r, cy - 0.38 * r),
            (cx + 0.38 * r, cy + 0.38 * r),
            _WHITE,
            w,
        )
        cv.line(
            (cx - 0.38 * r, cy + 0.38 * r),
            (cx + 0.38 * r, cy - 0.38 * r),
            _WHITE,
            w,
        )
