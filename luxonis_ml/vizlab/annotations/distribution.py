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
from .chip import chip_size, draw_chip
from .overlay import CARD_BG, CARD_TEXT, Cell, CornerStack

DistributionMode = Literal["bars", "chips", "gauge", "stacked"]
"""How a `ClassDistribution` is drawn. See the class docstring."""

ValueFormat = Literal["percent", "count", "count+percent"]
"""How a `ClassDistribution` value is labeled: a percentage (the default, for
probabilities), a raw integer count, or ``"count · percent"`` — the last two for
frequency distributions such as dataset class counts, where bars scale to the
largest value instead of an absolute ``[0, 1]``."""

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
        value_format: How each value is labeled — ``"percent"`` (the default, for
            probabilities), ``"count"`` (a raw integer), or ``"count+percent"``.
            The count formats scale bars to the largest value rather than an
            absolute ``[0, 1]``, so they suit frequency distributions (e.g. dataset
            class counts).

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
    value_format: ValueFormat = "percent"

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

    # -- value formatting ----------------------------------------------------

    def _total(self) -> float:
        """Total of all values, used for the share in count formats."""
        return sum(max(0.0, v) for _, v in self._pairs())

    def _bar_scale(self, values: list[float]) -> float:
        """Value that maps to a full bar: ``1.0`` for percent, else the max value.

        Probabilities are drawn on an absolute ``[0, 1]`` scale; counts have no
        such ceiling, so their bars scale to the largest value.
        """
        if self.value_format == "percent":
            return 1.0
        top = max(values, default=1.0)
        return top if top > 0 else 1.0

    def _value_label(self, value: float, total: float) -> str:
        """Format a value per ``value_format`` (percent / count / count+percent)."""
        if self.value_format == "percent":
            return _pct(value)
        count = round(value)
        if self.value_format == "count":
            return str(count)
        share = _pct(value / total) if total > 0 else "0%"
        return f"{count} · {share}"

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

        total = self._total()
        scale = self._bar_scale([v for _, v in rows])
        bar_w = size * 7.0
        bar_h = size * 0.72
        measured = [
            (
                name,
                value,
                self._value_label(value, total),
                canvas.measure_text(name, size, weight=weight),
            )
            for name, value in rows
        ]
        val_w = max(
            canvas.measure_text(label, size, weight=weight).width
            for _, _, label, _ in measured
        )
        name_w = max(m.width for _, _, _, m in measured)
        row_h = max(bar_h, *(m.height for _, _, _, m in measured))
        title_metrics = self._title_metrics(canvas, size)
        title_h = title_metrics.height + _ROW_GAP if title_metrics else 0.0

        content_w = name_w + _COL_GAP + bar_w + _COL_GAP + val_w
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
            val_x = bar_x + bar_w + _COL_GAP
            for name, value, label, m in measured:
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
                fill_w = bar_w * _clamp01(value / scale)
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
                lm = cv.measure_text(label, size, weight=weight)
                cv.text(
                    (val_x, y + (row_h - lm.height) / 2 + lm.ascent),
                    label,
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
        total = self._total()
        cells: list[Cell] = []
        for name, value in self._selected():
            label = self._value_label(value, total)
            text = f"{name}  {label}" if name else label
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
        name, value = ranked[0]
        scale = self._bar_scale([v for _, v in ranked])
        color = palette.color_for(name)
        size = style.font_size
        big = size * 1.5
        bar_h = size * 0.9

        name_m = canvas.measure_text(name, size, weight=700)
        pct_text = self._value_label(value, self._total())
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
            fill_w = (right_edge - left) * _clamp01(value / scale)
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

        total = self._total()
        other_value = max(0.0, total - sum(v for _, v in segs))
        keys = (
            [*segs, ("other", other_value)]
            if total > 0 and other_value / total > 0.005
            else list(segs)
        )

        strip_w = size * 14.0
        strip_h = size * 1.1
        swatch = size
        key_measured = [
            (
                name,
                value,
                self._value_label(value, total),
                canvas.measure_text(
                    f"{name}  {self._value_label(value, total)}",
                    size,
                    weight=weight,
                ),
            )
            for name, value in keys
        ]
        row_h = max(m.height for _, _, _, m in key_measured)
        key_w = max(swatch + _COL_GAP + m.width for _, _, _, m in key_measured)
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
            # segment laid left to right, proportional to its share of the total.
            left = rect.left + _PAD
            inner_w = content_w
            cv.rounded_rect(
                Rect(left, y, left + inner_w, y + strip_h),
                radius=4.0,
                fill=_OTHER,
            )
            seg_x = left
            for name, value in segs:
                seg_w = inner_w * (value / total if total > 0 else 0.0)
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
            for name, _value, label, m in key_measured:
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
                    f"{name}  {label}",
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
