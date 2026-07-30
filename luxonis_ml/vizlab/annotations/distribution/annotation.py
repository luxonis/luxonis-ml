"""The `ClassDistribution` overlay: prediction scores as a small chart."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.annotations.card import draw_card_background
from luxonis_ml.vizlab.annotations.chip import chip_size, draw_chip
from luxonis_ml.vizlab.annotations.overlay import (
    Cell,
    CornerStack,
    resolve_chrome,
    shade_outline,
)
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.canvas import Canvas, TextMetrics
from luxonis_ml.vizlab.style import Palette, Style

from .charts import (
    _BarsLayout,
    _draw_bars,
    _draw_pie,
    _draw_stacked,
    _draw_verdict,
    _PieLayout,
    _StackedLayout,
)
from .metrics import (
    _COL_GAP,
    _PAD,
    _PIE_KEY_GAP,
    _ROW_GAP,
    _WHITE,
    _clamp01,
    _edge_w,
    _pct,
)

DistributionMode = Literal["bars", "chips", "gauge", "stacked", "pie", "donut"]


ValueFormat = Literal["percent", "count", "count+percent"]


@dataclass(frozen=True)
class _SegmentData:
    """Shared selected-distribution data for proportional chart modes."""

    canvas: Canvas
    palette: Palette
    size: float
    weight: int
    segments: list[tuple[str, float]]
    total: float
    keys: list[tuple[str, float]]


@dataclass(frozen=True)
class _CardBox:
    """The measured box of a card stacking a chart above a legend key."""

    card_w: float
    card_h: float
    inner_w: float
    row_h: float
    title_h: float
    has_title: bool


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
            draw_card_background(cv, rect, style, chrome)
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
        data = self._segment_data(ctx, style)
        if data is None:
            return []
        strip_h = data.size * 1.1
        rows = self._key_rows(data)
        box = self._key_card_box(
            data, rows, data.size * 14.0, strip_h, _ROW_GAP
        )
        layout = _StackedLayout(
            segs=data.segments,
            key_measured=rows,
            palette=data.palette,
            total=data.total,
            strip_h=strip_h,
            swatch=data.size,
            row_h=box.row_h,
            inner_w=box.inner_w,
            title_h=box.title_h,
            has_title=box.has_title,
            chrome=resolve_chrome(ctx),
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            _draw_stacked(cv, rect, self, style, layout)

        return [Cell(box.card_w, box.card_h, _draw)]

    # -- pie / donut ---------------------------------------------------------

    def _keyed_segments(
        self, segs: list[tuple[str, float]], total: float
    ) -> list[tuple[str, float]]:
        """Append a rolled-up ``"other"`` slice when the rest is non-negligible."""
        other = max(0.0, total - sum(v for _, v in segs))
        if total > 0 and other / total > 0.005:
            return [*segs, ("other", other)]
        return list(segs)

    def _segment_data(
        self,
        ctx: RenderContext,
        style: Style,
    ) -> _SegmentData | None:
        """Resolve shared inputs for stacked, pie, and donut layouts."""
        segments = self._selected()
        if not segments:
            return None
        total = self._total()
        return _SegmentData(
            canvas=ctx.canvas,
            palette=self.resolved_palette(ctx),
            size=style.font_size,
            weight=style.font_weight,
            segments=segments,
            total=total,
            keys=self._keyed_segments(segments, total),
        )

    def _key_rows(
        self, data: _SegmentData
    ) -> list[tuple[str, str, TextMetrics]]:
        """Measure one ``"<name>  <value>"`` legend row per key segment."""
        rows = []
        for name, value in data.keys:
            label = self._value_label(value, data.total)
            rows.append(
                (
                    name,
                    label,
                    data.canvas.measure_text(
                        f"{name}  {label}", data.size, weight=data.weight
                    ),
                )
            )
        return rows

    def _key_card_box(
        self,
        data: _SegmentData,
        rows: Sequence[tuple[str, str, TextMetrics]],
        chart_w: float,
        chart_h: float,
        key_gap: float,
    ) -> _CardBox:
        """Size a card holding a chart above the legend key of ``rows``.

        The card is as wide as its widest part — chart, key, or title — and tall
        enough for the title band, the chart, and one ``row_h`` per key row.
        """
        row_h = max(m.height for _, _, m in rows)
        key_w = max(data.size + _COL_GAP + m.width for _, _, m in rows)
        title_metrics, title_h = self._title_band(data.canvas, data.size)
        inner_w = max(chart_w, key_w)
        if title_metrics is not None:
            inner_w = max(inner_w, title_metrics.width)
        return _CardBox(
            card_w=inner_w + 2 * _PAD,
            card_h=(
                2 * _PAD
                + title_h
                + chart_h
                + key_gap
                + len(rows) * row_h
                + _ROW_GAP * (len(rows) - 1)
            ),
            inner_w=inner_w,
            row_h=row_h,
            title_h=title_h,
            has_title=title_metrics is not None,
        )

    def _pie_cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        data = self._segment_data(ctx, style)
        if data is None:
            return []
        diameter = data.size * 8.0
        rows = self._key_rows(data)
        box = self._key_card_box(data, rows, diameter, diameter, _PIE_KEY_GAP)
        layout = _PieLayout(
            segs=data.segments,
            keys=data.keys,
            key_measured=rows,
            palette=data.palette,
            total=data.total,
            diameter=diameter,
            swatch=data.size,
            row_h=box.row_h,
            donut=self.mode == "donut",
            title_h=box.title_h,
            has_title=box.has_title,
            chrome=resolve_chrome(ctx),
        )

        def _draw(cv: Canvas, rect: Rect) -> None:
            _draw_pie(cv, rect, self, style, layout)

        return [Cell(box.card_w, box.card_h, _draw)]
