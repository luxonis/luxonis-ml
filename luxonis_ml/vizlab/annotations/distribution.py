"""Classification predictions as a probability distribution over classes.

`Classification` renders dataset *labels* — the one (or few) correct classes as
corner chips. A model *prediction* is different: a probability distribution over
all classes. `ClassDistribution` renders that distribution, with several
interchangeable looks (``mode``) and an optional ground-truth marker so you can
see at a glance whether the top prediction is correct.

Like `Heatmap`, this is a caller-supplied construct: LDF datasets carry a single
class name, not a probability vector, so the scores come from the prediction side
(e.g. a softmax output paired with ``dataset.get_class_names()``), not from the
dataset path. It is an image-level corner overlay, built on the same
`CornerStack` machinery as `Legend` and `Classification`.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.canvas import Canvas, Shadow, TextMetrics
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.style import Palette, Style

from .base import RenderContext
from .chip import chip_size, draw_chip
from .overlay import (
    Cell,
    CornerStack,
    resolve_chrome,
    shade_outline,
    swatch_outline,
)

DistributionMode = Literal["bars", "chips", "gauge", "stacked", "pie", "donut"]
"""How a `ClassDistribution` is drawn. See the class docstring."""

ValueFormat = Literal["percent", "count", "count+percent"]
"""How a `ClassDistribution` value is labeled: a percentage (the default, for
probabilities), a raw integer count, or ``"count · percent"`` — the last two for
frequency distributions such as dataset class counts, where each bar shows the
class's share of the total instead of an absolute ``[0, 1]`` probability."""

# Neutral highlight for the ground-truth marker and verdict ticks; the rest is
# on-brand chrome (green ✓ / red ✗ verdict, a muted slate "other" segment).
_WHITE = Color(255, 255, 255)
_OK = brand.SUCCESS
_BAD = brand.ERROR
_OTHER = brand.MUTED

_PAD = 10.0
_ROW_GAP = 6.0
_COL_GAP = 8.0
# Extra breathing room between a pie/donut's bottom edge and the key below it.
_PIE_KEY_GAP = 16.0


def _clamp01(value: float) -> float:
    """Clamp a probability into ``[0, 1]``."""
    return max(0.0, min(1.0, value))


def _pct(prob: float) -> str:
    """Format a probability as a whole-percent string, e.g. ``"92%"``."""
    return f"{round(_clamp01(prob) * 100)}%"


def _edge_w(style: Style) -> float:
    """Hairline width for chart-element outlines, tracking the font size."""
    return max(1.0, style.font_size * 0.09)


class ClassDistribution(CornerStack):
    """A predicted probability distribution over classes, in one of four looks.

    Feed it a distribution (a ``{name: probability}`` mapping, or ``(name, prob)``
    pairs) and it draws a corner panel. ``mode`` selects one of six looks:

    - ``"bars"`` (default): a ranked horizontal bar chart — class name, a bar whose
      length is the probability, and the percentage. Best general-purpose view.
    - ``"chips"``: compact colored chips with percentages (the `Classification`
      tag look), stacked in the corner.
    - ``"gauge"``: just the winning class as a single large confidence bar.
    - ``"stacked"``: one segmented strip showing the classes as proportions, with
      an inline key.
    - ``"pie"`` / ``"donut"``: a circular chart of the classes as proportional
      wedges, with an inline key; ``"donut"`` leaves a hole and prints the top
      class's share in its center.

    Each class is colored from the palette by name, so it matches its box color.
    Set ``ground_truth`` to mark the correct class (a highlighted row/chip/segment,
    or a ✓/✗ on the gauge); if the true class falls outside ``top_k`` it is still
    shown, so a missed prediction stays visible.

    .. image:: TODO-HOST/distributions.png
       :alt: One prediction shown in every class-distribution mode.

    Attributes:
        probabilities: The distribution as a ``{name: prob}`` mapping or a list of
            ``(name, prob)`` pairs. Probabilities are expected in ``[0, 1]``.
        mode: Which look to draw (see above).
        ground_truth: The correct class name to highlight, or ``None``.
        top_k: Show only the ``top_k`` most probable classes (plus the ground-truth
            row if it is set and would otherwise be hidden). ``None`` shows all.
        title: Optional heading drawn above the panel (ignored in ``"chips"``).
        value_format: How each value is labeled — ``"percent"`` (the default, for
            probabilities), ``"count"`` (a raw integer), or ``"count+percent"``.
            The count formats draw each bar as the class's share of the total
            rather than an absolute ``[0, 1]``, so they suit frequency
            distributions (e.g. dataset class counts).

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
        **kwargs,
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
        return cls(probabilities=pairs, **kwargs)

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

    def _bar_scale(self) -> float:
        """Value that maps to a full bar: ``1.0`` for percent, else the total.

        Probabilities are already fractions on an absolute ``[0, 1]`` scale, so a
        full bar is ``1.0``. Counts are drawn as a share of the whole, so a full
        bar is the total — every bar's length then matches the percentage shown in
        its label, and the bars together fill exactly one bar.
        """
        if self.value_format == "percent":
            return 1.0
        total = self._total()
        return total if total > 0 else 1.0

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
        if self.mode in ("pie", "donut"):
            return self._pie_cells(ctx, style)
        return self._bars_cells(ctx, style)

    def _card_bg(
        self, cv: Canvas, rect: Rect, style: Style, chrome: brand.Chrome
    ) -> None:
        """Paint the shared rounded card background (theme-aware fill/border)."""
        cv.rounded_rect(
            rect,
            radius=9.0,
            fill=chrome.card_bg,
            stroke=chrome.border,
            stroke_width=1.0 if chrome.border is not None else 0.0,
            shadow=Shadow(blur=6.0, dy=2.0) if style.shadow else None,
        )

    def _title_metrics(
        self, canvas: Canvas, size: float
    ) -> TextMetrics | None:
        """Measure the optional title, or ``None`` when there is none."""
        if self.title is None:
            return None
        return canvas.measure_text(self.title, size * 1.05, weight=700)

    def _title_band(
        self, canvas: Canvas, size: float
    ) -> tuple[TextMetrics | None, float]:
        """Measure the optional title and the vertical band it occupies."""
        metrics = self._title_metrics(canvas, size)
        return metrics, metrics.height + _ROW_GAP if metrics else 0.0

    def _draw_title(
        self,
        cv: Canvas,
        x: float,
        y: float,
        size: float,
        chrome: brand.Chrome,
    ) -> None:
        """Draw the title at baseline; assumes there is one."""
        metrics = cv.measure_text(str(self.title), size * 1.05, weight=700)
        cv.text(
            (x, y + metrics.ascent),
            str(self.title),
            size=size * 1.05,
            color=chrome.card_title,
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
        scale = self._bar_scale()
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
            canvas.measure_text(label, size, weight=weight, mono=True).width
            for _, _, label, _ in measured
        )
        name_w = max(m.width for _, _, _, m in measured)
        row_h = max(bar_h, *(m.height for _, _, _, m in measured))
        title_metrics, title_h = self._title_band(canvas, size)

        content_w = name_w + _COL_GAP + bar_w + _COL_GAP + val_w
        if title_metrics is not None:
            content_w = max(content_w, title_metrics.width)
        card_w = content_w + 2 * _PAD
        card_h = (
            2 * _PAD + title_h + len(rows) * row_h + _ROW_GAP * (len(rows) - 1)
        )
        layout = _BarsLayout(
            measured=measured,
            palette=palette,
            scale=scale,
            bar_w=bar_w,
            bar_h=bar_h,
            row_h=row_h,
            name_w=name_w,
            title_h=title_h,
            has_title=title_metrics is not None,
            chrome=resolve_chrome(ctx),
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            _draw_bars(cv, rect, self, style, layout)

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
        scale = self._bar_scale()
        color = palette.color_for(name)
        chrome = resolve_chrome(ctx)
        size = style.font_size
        big = size * 1.5
        bar_h = size * 0.9

        name_m = canvas.measure_text(name, size, weight=700)
        pct_text = self._value_label(value, self._total())
        pct_m = canvas.measure_text(pct_text, big, weight=700, mono=True)
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
            self._card_bg(cv, rect, style, chrome)
            y = rect.top + _PAD
            cv.text(
                (
                    rect.left + _PAD,
                    y + (header_h - name_m.height) / 2 + name_m.ascent,
                ),
                name,
                size=size,
                color=chrome.card_text,
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
                color=chrome.card_text,
                weight=700,
                mono=True,
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
                    stroke=shade_outline(color, chrome.card_bg),
                    stroke_width=_edge_w(style),
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
        keys = self._keyed_segments(segs, total)

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
        title_metrics, title_h = self._title_band(canvas, size)

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
        layout = _StackedLayout(
            segs=segs,
            key_measured=key_measured,
            palette=palette,
            total=total,
            strip_h=strip_h,
            swatch=swatch,
            row_h=row_h,
            inner_w=content_w,
            title_h=title_h,
            has_title=title_metrics is not None,
            chrome=resolve_chrome(ctx),
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            _draw_stacked(cv, rect, self, style, layout)

        return [Cell(card_w, card_h, _draw)]

    # -- pie / donut ---------------------------------------------------------

    def _keyed_segments(
        self, segs: list[tuple[str, float]], total: float
    ) -> list[tuple[str, float]]:
        """Append a rolled-up ``"other"`` slice when the rest is non-negligible."""
        other = max(0.0, total - sum(v for _, v in segs))
        if total > 0 and other / total > 0.005:
            return [*segs, ("other", other)]
        return list(segs)

    def _pie_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        size, weight = style.font_size, style.font_weight
        segs = self._selected()
        if not segs:
            return []

        total = self._total()
        keys = self._keyed_segments(segs, total)
        diameter = size * 8.0
        swatch = size
        key_measured = [
            (
                name,
                canvas.measure_text(
                    f"{name}  {self._value_label(value, total)}",
                    size,
                    weight=weight,
                ),
            )
            for name, value in keys
        ]
        row_h = max(m.height for _, m in key_measured)
        key_w = max(swatch + _COL_GAP + m.width for _, m in key_measured)
        title_metrics, title_h = self._title_band(canvas, size)

        content_w = max(diameter, key_w)
        if title_metrics is not None:
            content_w = max(content_w, title_metrics.width)
        card_w = content_w + 2 * _PAD
        card_h = (
            2 * _PAD
            + title_h
            + diameter
            + _PIE_KEY_GAP
            + len(key_measured) * row_h
            + _ROW_GAP * (len(key_measured) - 1)
        )
        layout = _PieLayout(
            segs=segs,
            keys=keys,
            key_measured=key_measured,
            palette=palette,
            total=total,
            diameter=diameter,
            swatch=swatch,
            row_h=row_h,
            donut=self.mode == "donut",
            title_h=title_h,
            has_title=title_metrics is not None,
            chrome=resolve_chrome(ctx),
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            _draw_pie(cv, rect, self, style, layout)

        return [Cell(card_w, card_h, _draw)]


@dataclass(frozen=True)
class _BarsLayout:
    """Precomputed geometry for a ``"bars"`` distribution card."""

    measured: list[tuple[str, float, str, TextMetrics]]
    palette: Palette
    scale: float
    bar_w: float
    bar_h: float
    row_h: float
    name_w: float
    title_h: float
    has_title: bool
    chrome: brand.Chrome


def _draw_bars(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: _BarsLayout,
) -> None:
    """Paint a ranked bar chart: per row a name, a value bar, and a label."""
    chrome = ll.chrome
    dist._card_bg(cv, rect, style, chrome)
    size, weight = style.font_size, style.font_weight
    y = rect.top + _PAD
    if ll.has_title:
        dist._draw_title(cv, rect.left + _PAD, y, size, chrome)
        y += ll.title_h
    bar_x = rect.left + _PAD + ll.name_w + _COL_GAP
    val_x = bar_x + ll.bar_w + _COL_GAP
    for name, value, label, m in ll.measured:
        is_gt = name == dist.ground_truth
        color = ll.palette.color_for(name)
        cv.text(
            (rect.left + _PAD, y + (ll.row_h - m.height) / 2 + m.ascent),
            name,
            size=size,
            color=chrome.card_text,
            weight=700 if is_gt else weight,
        )
        track_top = y + (ll.row_h - ll.bar_h) / 2
        track = Rect(bar_x, track_top, bar_x + ll.bar_w, track_top + ll.bar_h)
        cv.rounded_rect(track, radius=ll.bar_h / 2, fill=color.with_alpha(0.2))
        fill_w = ll.bar_w * _clamp01(value / ll.scale)
        if fill_w > 0:
            cv.rounded_rect(
                Rect(bar_x, track_top, bar_x + fill_w, track_top + ll.bar_h),
                radius=ll.bar_h / 2,
                fill=color,
                stroke=shade_outline(color, chrome.card_bg),
                stroke_width=_edge_w(style),
            )
        if is_gt:
            cv.rounded_rect(
                track, radius=ll.bar_h / 2, stroke=_WHITE, stroke_width=1.5
            )
        lm = cv.measure_text(label, size, weight=weight, mono=True)
        cv.text(
            (val_x, y + (ll.row_h - lm.height) / 2 + lm.ascent),
            label,
            size=size,
            color=chrome.card_text,
            weight=weight,
            mono=True,
        )
        y += ll.row_h + _ROW_GAP


@dataclass(frozen=True)
class _StackedLayout:
    """Precomputed geometry for a ``"stacked"`` distribution card."""

    segs: list[tuple[str, float]]
    key_measured: list[tuple[str, float, str, TextMetrics]]
    palette: Palette
    total: float
    strip_h: float
    swatch: float
    row_h: float
    inner_w: float
    title_h: float
    has_title: bool
    chrome: brand.Chrome


def _draw_stacked_strip(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    ll: _StackedLayout,
    y: float,
    edge_w: float,
) -> None:
    """Paint the proportional strip: a muted backdrop plus each class segment."""
    left = rect.left + _PAD
    cv.rounded_rect(
        Rect(left, y, left + ll.inner_w, y + ll.strip_h),
        radius=4.0,
        fill=_OTHER,
    )
    seg_x = left
    for name, value in ll.segs:
        seg_w = ll.inner_w * (value / ll.total if ll.total > 0 else 0.0)
        if seg_w <= 0:
            continue
        seg = Rect(seg_x, y, seg_x + seg_w, y + ll.strip_h)
        color = ll.palette.color_for(name)
        gt = name == dist.ground_truth
        cv.rounded_rect(
            seg,
            radius=0.0,
            fill=color,
            stroke=_WHITE if gt else shade_outline(color, ll.chrome.card_bg),
            stroke_width=2.0 if gt else edge_w,
        )
        seg_x += seg_w


def _draw_key_swatch(
    cv: Canvas,
    left: float,
    y: float,
    color: Color,
    ll: "_StackedLayout | _PieLayout",
) -> None:
    """Draw one key's color swatch, vertically centered in its row."""
    sw_top = y + (ll.row_h - ll.swatch) / 2
    cv.rounded_rect(
        Rect(left, sw_top, left + ll.swatch, sw_top + ll.swatch),
        radius=3.0,
        fill=color,
        stroke=swatch_outline(ll.chrome.card_bg),
        stroke_width=1.0,
    )


def _draw_stacked_key(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: _StackedLayout,
    y: float,
) -> None:
    """Paint the inline swatch + name key beneath the strip."""
    size, weight = style.font_size, style.font_weight
    for name, _value, label, m in ll.key_measured:
        color = _OTHER if name == "other" else ll.palette.color_for(name)
        _draw_key_swatch(cv, rect.left + _PAD, y, color, ll)
        cv.text(
            (rect.left + _PAD + ll.swatch + _COL_GAP, y + m.ascent),
            f"{name}  {label}",
            size=size,
            color=ll.chrome.card_text,
            weight=700 if name == dist.ground_truth else weight,
        )
        y += ll.row_h + _ROW_GAP


def _draw_stacked(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: _StackedLayout,
) -> None:
    """Paint a stacked proportion strip with an inline key below it."""
    dist._card_bg(cv, rect, style, ll.chrome)
    y = rect.top + _PAD
    if ll.has_title:
        dist._draw_title(cv, rect.left + _PAD, y, style.font_size, ll.chrome)
        y += ll.title_h
    _draw_stacked_strip(cv, rect, dist, ll, y, _edge_w(style))
    _draw_stacked_key(cv, rect, dist, style, ll, y + ll.strip_h + _ROW_GAP)


@dataclass(frozen=True)
class _PieLayout:
    """Precomputed geometry for a ``"pie"``/``"donut"`` distribution card."""

    segs: list[tuple[str, float]]
    keys: list[tuple[str, float]]
    key_measured: list[tuple[str, TextMetrics]]
    palette: Palette
    total: float
    diameter: float
    swatch: float
    row_h: float
    donut: bool
    title_h: float
    has_title: bool
    chrome: brand.Chrome


def _wedge_polygon(
    cx: float, cy: float, r_out: float, r_in: float, a0: float, a1: float
) -> list[XY]:
    """Points of an annular sector (a full pie wedge when ``r_in`` is ``0``)."""
    span = a1 - a0
    steps = max(2, int(span / (math.pi / 96)) + 1)
    outer: list[XY] = [
        (
            cx + r_out * math.cos(a0 + span * k / steps),
            cy + r_out * math.sin(a0 + span * k / steps),
        )
        for k in range(steps + 1)
    ]
    inner: list[XY] = [
        (
            cx + r_in * math.cos(a1 - span * k / steps),
            cy + r_in * math.sin(a1 - span * k / steps),
        )
        for k in range(steps + 1)
    ]
    return outer + inner


def _wedge_color(name: str, ll: _PieLayout) -> Color:
    """Resolve a slice's fill: the palette color, or muted slate for ``other``."""
    return _OTHER if name == "other" else ll.palette.color_for(name)


def _readable_center(color: Color, background: Color) -> Color:
    """Tint a slice color to a legible lightness for the donut's center number.

    Keeps the slice's hue (so the number still reads as "that class") but forces a
    lightness that contrasts with the hole background — bright on a dark card, dark
    on a light one — so a naturally dark or washed-out class color stays visible.
    """
    on_dark = background.readable_text_color().r > 200
    hue, lightness, saturation = color.hls
    target = max(lightness, 0.72) if on_dark else min(lightness, 0.32)
    return Color.from_hls(hue, target, min(saturation, 0.85))


def _draw_wedges(
    cv: Canvas,
    dist: "ClassDistribution",
    ll: _PieLayout,
    cx: float,
    cy: float,
    r_out: float,
    r_in: float,
    sep_w: float,
    pop: float,
) -> None:
    """Paint each class as a clean, solid wedge.

    Slices are separated by a seam in the card's own background color (white on a
    light card, navy on a dark one), for a crisp flat look that reads as a gap
    rather than a foreign line; the ground-truth slice is nudged outward (an
    "exploded" slice) so it stands out. A single class fills the whole circle/ring
    with no separators.
    """
    sep = ll.chrome.card_bg.with_alpha(255)
    drawn = [(name, value) for name, value in ll.keys if value > 0]
    if ll.total <= 0 or not drawn:
        cv.polygon(
            _wedge_polygon(cx, cy, r_out, r_in, -math.pi / 2, 1.5 * math.pi),
            fill=_OTHER,
        )
        return
    if len(drawn) == 1:  # one class -> a solid, seamless circle/ring
        cv.polygon(
            _wedge_polygon(cx, cy, r_out, r_in, -math.pi / 2, 1.5 * math.pi),
            fill=_wedge_color(drawn[0][0], ll),
        )
        return
    angle = -math.pi / 2  # start at 12 o'clock
    for name, value in drawn:
        a1 = angle + (value / ll.total) * 2 * math.pi
        ox, oy = cx, cy
        if pop > 0 and name == dist.ground_truth:
            mid = (angle + a1) / 2
            ox, oy = cx + pop * math.cos(mid), cy + pop * math.sin(mid)
        cv.polygon(
            _wedge_polygon(ox, oy, r_out, r_in, angle, a1),
            fill=_wedge_color(name, ll),
            stroke=sep,
            stroke_width=sep_w,
        )
        angle = a1


def _draw_pie_key(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: _PieLayout,
    y: float,
) -> None:
    """Paint the swatch + name + value key beneath the pie."""
    size, weight = style.font_size, style.font_weight
    for (name, value), (_name, m) in zip(
        ll.keys, ll.key_measured, strict=True
    ):
        color = _OTHER if name == "other" else ll.palette.color_for(name)
        _draw_key_swatch(cv, rect.left + _PAD, y, color, ll)
        cv.text(
            (rect.left + _PAD + ll.swatch + _COL_GAP, y + m.ascent),
            f"{name}  {dist._value_label(value, ll.total)}",
            size=size,
            color=ll.chrome.card_text,
            weight=700 if name == dist.ground_truth else weight,
        )
        y += ll.row_h + _ROW_GAP


def _draw_pie(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: _PieLayout,
) -> None:
    """Paint a pie/donut chart with an inline key below it."""
    dist._card_bg(cv, rect, style, ll.chrome)
    y = rect.top + _PAD
    if ll.has_title:
        dist._draw_title(cv, rect.left + _PAD, y, style.font_size, ll.chrome)
        y += ll.title_h
    # Reserve a little room inside the diameter box so an exploded ground-truth
    # slice never spills past it.
    pop = ll.diameter * 0.035 if dist.ground_truth is not None else 0.0
    r_out = ll.diameter / 2 - pop
    r_in = r_out * 0.6 if ll.donut else 0.0
    cx = (rect.left + rect.right) / 2
    cy = y + ll.diameter / 2
    sep_w = max(2.0, ll.diameter * 0.014)
    _draw_wedges(cv, dist, ll, cx, cy, r_out, r_in, sep_w, pop)
    if ll.donut and ll.total > 0 and ll.segs:
        top_name, top_value = max(ll.segs, key=lambda kv: kv[1])
        label = _pct(top_value / ll.total)
        big = style.font_size * 1.2
        m = cv.measure_text(label, big, weight=700, mono=True)
        cv.text(
            (cx - m.width / 2, cy - m.height / 2 + m.ascent),
            label,
            size=big,
            # The top slice's hue, but forced to a legible lightness on the hole.
            color=_readable_center(
                _wedge_color(top_name, ll), ll.chrome.card_bg
            ),
            weight=700,
            mono=True,
        )
    _draw_pie_key(cv, rect, dist, style, ll, y + ll.diameter + _PIE_KEY_GAP)


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
