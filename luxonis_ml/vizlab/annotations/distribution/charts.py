"""Painters for each `"ClassDistribution"` mode: bars, stacked, pie, verdict.

Each takes a laid-out cell plus its resolved colours and draws it. They know
nothing about the annotation that chose them, so a mode can be read, changed
or added without touching the annotation's data handling.
"""

from typing import TYPE_CHECKING

from .metrics import (
    _BAD,
    _COL_GAP,
    _OK,
    _OTHER,
    _PAD,
    _PIE_KEY_GAP,
    _ROW_GAP,
    _WHITE,
    _clamp01,
    _edge_w,
    _pct,
)

if TYPE_CHECKING:
    from .annotation import ClassDistribution

import math
from collections.abc import Iterable
from dataclasses import dataclass

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.annotations.card import draw_card_background
from luxonis_ml.vizlab.annotations.overlay import (
    shade_outline,
    swatch_outline,
)
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY, Rect
from luxonis_ml.vizlab.render.canvas import Canvas, TextMetrics
from luxonis_ml.vizlab.style import Palette, Style


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


def _draw_chart_header(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    chrome: brand.Chrome,
    *,
    has_title: bool,
    title_h: float,
) -> float:
    """Draw the shared chart card/title and return its content y coordinate."""
    draw_card_background(cv, rect, style, chrome)
    y = rect.top + _PAD
    if has_title:
        dist._draw_title(cv, rect.left + _PAD, y, style.font_size, chrome)
        y += title_h
    return y


def _draw_bars(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: _BarsLayout,
) -> None:
    """Paint a ranked bar chart: per row a name, a value bar, and a label."""
    chrome = ll.chrome
    size, weight = style.font_size, style.font_weight
    y = _draw_chart_header(
        cv,
        rect,
        dist,
        style,
        chrome,
        has_title=ll.has_title,
        title_h=ll.title_h,
    )
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
    key_measured: list[tuple[str, str, TextMetrics]]
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


def _draw_distribution_key(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: "_StackedLayout | _PieLayout",
    rows: Iterable[tuple[str, str, TextMetrics]],
    y: float,
) -> None:
    """Paint shared inline swatch, name, and value rows."""
    size, weight = style.font_size, style.font_weight
    for name, label, metrics in rows:
        color = _OTHER if name == "other" else ll.palette.color_for(name)
        _draw_key_swatch(cv, rect.left + _PAD, y, color, ll)
        cv.text(
            (
                rect.left + _PAD + ll.swatch + _COL_GAP,
                y + metrics.ascent,
            ),
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
    y = _draw_chart_header(
        cv,
        rect,
        dist,
        style,
        ll.chrome,
        has_title=ll.has_title,
        title_h=ll.title_h,
    )
    _draw_stacked_strip(cv, rect, dist, ll, y, _edge_w(style))
    _draw_distribution_key(
        cv,
        rect,
        dist,
        style,
        ll,
        ll.key_measured,
        y + ll.strip_h + _ROW_GAP,
    )


@dataclass(frozen=True)
class _PieLayout:
    """Precomputed geometry for a ``"pie"``/``"donut"`` distribution card."""

    segs: list[tuple[str, float]]
    keys: list[tuple[str, float]]
    key_measured: list[tuple[str, str, TextMetrics]]
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


def _draw_pie(
    cv: Canvas,
    rect: Rect,
    dist: "ClassDistribution",
    style: Style,
    ll: _PieLayout,
) -> None:
    """Paint a pie/donut chart with an inline key below it."""
    y = _draw_chart_header(
        cv,
        rect,
        dist,
        style,
        ll.chrome,
        has_title=ll.has_title,
        title_h=ll.title_h,
    )
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
    _draw_distribution_key(
        cv,
        rect,
        dist,
        style,
        ll,
        ll.key_measured,
        y + ll.diameter + _PIE_KEY_GAP,
    )


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
