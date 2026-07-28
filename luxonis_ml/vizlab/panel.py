"""Append a metadata sidebar ("second window") to an image.

`with_panel` renders an image and appends a panel that shows arbitrary
JSON-like metadata (augmentations, tags, source, filenames, ...) as an indented
key/value tree. Like the `compose` functions it renders the image at
native resolution and returns a new `Image`, so the panel
never occludes pixels or labels and the original is untouched.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

from luxonis_ml.utils.color import brand

from ._util import is_sequence
from .canvas import Canvas
from .color import Color, ColorLike
from .geometry import Rect
from .image import Image, _style_scale
from .style import DEFAULT_STYLE, Style


@dataclass(frozen=True)
class Block:
    """A panel field rendered as a heading label with its value on its own line.

    Wrap a scalar in ``Block`` when its value is long (a file path, a URL) and an
    inline ``key: value`` row would cramp it: the panel gives it the group
    treatment instead — the key becomes an uppercase, letter-spaced section
    label, and the value sits on the line below it with the panel's full width.
    A value that overruns even that width is middle-ellipsized to one line,
    keeping its start and end (e.g. the filename and its extension) visible.

    Attributes:
        value: The scalar to show on its own line under the field's label.

    """

    value: str | int | float


@dataclass(frozen=True)
class Swatches:
    """A color-key field: ``(color, label)`` pairs drawn as an aligned legend.

    The panel renders each pair as a small filled square followed by its label,
    laid out as an aligned grid — a compact class legend that lives beside the
    metadata instead of over the image. Labels in ``disabled`` are drawn dimmed
    and struck through (their class is switched off) but stay in the legend.

    Attributes:
        items: The ``(color, label)`` pairs, in draw order.
        disabled: Labels currently switched off, drawn as disabled but kept.
        reserve: A label width to hold columns to even when it is absent (e.g. the
            dataset's longest class name), so the legend — and the panel — keep a
            stable width as the per-sample class set changes.

    """

    items: tuple[tuple[ColorLike, str], ...]
    disabled: frozenset[str] = frozenset()
    reserve: str = ""


@dataclass(frozen=True)
class Controls:
    """An interactive-controls field: ``(key, name, value, active)`` rows.

    The panel renders each as ``[key]  name … value``, the value tinted by
    ``active`` (on / off / neutral) — the interactive HUD, shown in the panel
    rather than floated over the image.

    Attributes:
        rows: One ``(key, name, value, active)`` per control. ``active`` is
            ``True`` (engaged), ``False`` (off), or ``None`` (neutral).

    """

    rows: tuple[tuple[str, str, str, bool | None], ...]


#: JSON-like metadata a panel can render: a scalar, a `Block` / `Swatches` /
#: `Controls` field, or a mapping/sequence nested arbitrarily. Anything else is
#: stringified as a leaf.
PanelData: TypeAlias = (
    Mapping[str, "PanelData"]
    | Sequence["PanelData"]
    | Block
    | Swatches
    | Controls
    | str
    | int
    | float
    | bool
    | None
)

# Nominal metrics at the style-reference resolution; scaled up on larger images
# so the panel's type tracks the picture size instead of shrinking against it.
_PAD = 16.0
_INDENT = 14.0
_LINE_GAP = 5.0
_PANEL_SIZE = 16.0
#: A comfortably wide default so small per-sample text changes (a shorter class
#: name, a different value) stay within it and the panel does not jump.
_MIN_WIDTH = 260.0
_MAX_WIDTH = 420.0
#: The class legend folds into at most this many columns.
_LEGEND_COLS = 2
#: The panel heading renders a step larger and bold, matching the grid/cell
#: titles (see `luxonis_ml.vizlab.compose`), so it reads as a heading. It is
#: uppercased and lightly letter-spaced to match the section labels below it.
_TITLE_SCALE = 1.3
_TITLE_WEIGHT = 700
_TITLE_TRACKING = 0.08  # letter-spacing as a fraction of the title size
#: Framing metrics: the image and the panel are drawn as separate rounded
#: surfaces floating on the composite background — a uniform outer margin, a gap
#: between them, rounded corners, a hairline border, and the breathing room
#: above/below each in-panel section rule.
_MARGIN = 14.0
_GAP = 12.0
_RADIUS = 10.0
_SECTION_GAP = 9.0
_BORDER_WIDTH = 1.0
#: Section headers render smaller than the body, uppercased, and letter-spaced,
#: so a group's label reads as a heading rather than another key/value row.
_HEADER_SCALE = 0.82
_HEADER_WEIGHT = 700
_HEADER_TRACKING = 0.16  # letter-spacing as a fraction of the header size

# Panel chrome is on-brand and background-aware: resolved per render from
# `brand.chrome_for` (see `_metrics`) so keys/values/title/divider adapt to a
# dark or light composite background.

_MEASURE = Canvas.blank(1, 1)

# One logical line: (depth, prefix, prefix_is_key, body).
Line = tuple[int, str, bool, str]


@dataclass(frozen=True)
class _Metrics:
    """Panel metrics (sizes + chrome colors) resolved for one panel."""

    size: float
    header_size: float
    pad: float
    indent: float
    line_gap: float
    min_width: float
    max_width: float
    margin: float
    gap: float
    radius: float
    section_gap: float
    border_width: float
    key: Color
    value: Color
    title: Color
    muted: Color
    active: Color
    divider: Color
    surface: Color
    border: Color
    page: Color


def _over(top: Color, bottom: Color) -> Color:
    """Composite a (possibly translucent) ``top`` over an opaque ``bottom``."""
    a = top.a / 255.0
    return Color(
        round(top.r * a + bottom.r * (1.0 - a)),
        round(top.g * a + bottom.g * (1.0 - a)),
        round(top.b * a + bottom.b * (1.0 - a)),
    )


def _metrics(scale: float, background: Color) -> _Metrics:
    """Resolve panel sizes (scaled by ``scale``) and background-aware chrome colors.

    Chrome adapts to the composite background via `brand.chrome_for`: near-white
    keys/values on a dark panel, brand-purple keys and deep-purple values/title on
    a light one (where the dark-theme chrome would be invisible). The image and
    panel float as rounded surfaces on an *inverted* page color (a light ground
    under a dark theme and vice versa) so the gap between them reads clearly; the
    panel keeps its themed dark/light card look by filling with an opaque version
    of ``card_bg`` (composited over the theme background) rather than letting the
    inverted page bleed through its translucency. Section headers get a muted
    label color.
    """
    chrome = brand.chrome_for(background)
    # A background that wants dark text is "light"; invert the page under it.
    is_light = background.readable_text_color().r < 128
    return _Metrics(
        size=_PANEL_SIZE * scale,
        header_size=_PANEL_SIZE * _HEADER_SCALE * scale,
        pad=_PAD * scale,
        indent=_INDENT * scale,
        line_gap=_LINE_GAP * scale,
        min_width=_MIN_WIDTH * scale,
        max_width=_MAX_WIDTH * scale,
        margin=_MARGIN * scale,
        gap=_GAP * scale,
        radius=_RADIUS * scale,
        section_gap=_SECTION_GAP * scale,
        border_width=max(1.0, _BORDER_WIDTH * scale),
        key=chrome.card_key,
        value=chrome.card_text,
        title=chrome.card_title,
        muted=brand.SLATE if is_light else brand.STEEL,
        active=brand.SUCCESS if is_light else brand.MINT,
        divider=chrome.divider,
        surface=_over(chrome.card_bg, background),
        border=chrome.border if chrome.border is not None else chrome.divider,
        page=brand.BACKGROUND if is_light else brand.LIGHT_BACKGROUND,
    )


def _is_container(value: object) -> bool:
    """Whether a value is a nested container (mapping or non-string sequence)."""
    return isinstance(value, Mapping) or is_sequence(value)


def _format_scalar(value: object) -> str:
    """Format a leaf value as a compact single-line string."""
    if isinstance(value, Block):
        return _format_scalar(value.value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= 500 else text[:499] + "…"


def _format_tree(data: "PanelData", depth: int = 0) -> list[Line]:
    """Flatten JSON-like data into indented key/value lines.

    Args:
        data: A mapping, sequence, or scalar (nested arbitrarily).
        depth: The current nesting depth (used for indentation).

    Returns:
        Logical lines ``(depth, prefix, prefix_is_key, body)``. Keys end in ``": "``
        (or ``":"`` for container headers); list items use a ``"• "`` bullet.

    Examples:
        >>> _format_tree({"a": 1, "b": [2, 3], "c": {"d": True}})
        [(0, 'a: ', True, '1'), (0, 'b:', True, ''), (1, '• ', False, '2'), \
(1, '• ', False, '3'), (0, 'c:', True, ''), (1, 'd: ', True, 'true')]

    """
    if isinstance(data, Mapping):
        return _mapping_lines(data, depth)
    if is_sequence(data):
        return _sequence_lines(data, depth)
    return [(depth, "", False, _format_scalar(data))]


def _mapping_lines(data: Mapping, depth: int) -> list[Line]:
    """Format a mapping's items as key/value (or key-header) lines."""
    lines: list[Line] = []
    for key, value in data.items():
        if isinstance(value, Block):
            # Fold a nested block (e.g. a filename inside a batched "sample N")
            # the same way the top level does: the key on its own line, the value
            # on the next, so a long value gets the full width instead of being
            # cramped after an inline "key: ".
            lines.append((depth, str(key), True, ""))
            lines.append((depth, "", False, _format_scalar(value.value)))
        elif _is_container(value):
            lines.append((depth, f"{key}:", True, ""))
            lines.extend(_format_tree(value, depth + 1))
        else:
            lines.append((depth, f"{key}: ", True, _format_scalar(value)))
    return lines


def _sequence_lines(data: Sequence, depth: int) -> list[Line]:
    """Format a sequence's items as bulleted lines."""
    lines: list[Line] = []
    for item in data:
        if _is_container(item):
            lines.append((depth, "•", False, ""))
            lines.extend(_format_tree(item, depth + 1))
        else:
            lines.append((depth, "• ", False, _format_scalar(item)))
    return lines


class Section(NamedTuple):
    """One panel section: a heading label plus its typed body.

    ``heading`` is ``None`` for a run of top-level scalars and the group key for
    any headed field. Exactly one body is populated: ``lines`` (text rows, or a
    single bare value when ``block``), ``swatches`` (a color legend), or
    ``controls`` (interactive-control rows).
    """

    heading: str | None
    lines: list[Line]
    block: bool = False
    #: (color, label, enabled) per legend swatch.
    swatches: tuple[tuple[Color, str, bool], ...] | None = None
    #: A label width the legend holds columns to even when absent (see `Swatches`).
    swatch_reserve: str = ""
    controls: tuple[tuple[str, str, str, bool | None], ...] | None = None


def _format_sections(data: "PanelData") -> list[Section]:
    """Split panel data into visually separated sections (a rule drawn between).

    Only the *top level* of a mapping is sectioned: each nested container (a
    grouped block, e.g. ``arrays`` or ``augmentations``) and each `Block` field
    becomes its own section headed by its key, while consecutive scalar entries
    are kept together as one heading-less section — so the rules and labels mark
    meaningful groups, not every single row. Sequences and bare scalars are a
    single, unheaded section.

    Args:
        data: The panel data (mapping/sequence/scalar/`Block`).

    Returns:
        The sections; ``heading`` is ``None`` for a scalar run and the group key
        for a container or a `Block` field (whose ``block`` flag is set).

    Examples:
        >>> [
        ...     (s.heading, len(s.lines), s.block)
        ...     for s in _format_sections(
        ...         {"a": 1, "nested": {"x": 1}, "path": Block("/a/b.jpg")}
        ...     )
        ... ]
        [(None, 1, False), ('nested', 1, False), ('path', 1, True)]

    """
    if not isinstance(data, Mapping):
        return [Section(None, _format_tree(data))]
    sections: list[Section] = []
    scalars: list[Line] = []

    def flush() -> None:
        nonlocal scalars
        if scalars:
            sections.append(Section(None, scalars))
            scalars = []

    for key, value in data.items():
        if isinstance(value, Block):
            flush()
            body = [(0, "", False, _format_scalar(value.value))]
            sections.append(Section(str(key), body, block=True))
        elif isinstance(value, Swatches):
            flush()
            items = tuple(
                (
                    Color.parse(color),
                    str(label),
                    str(label) not in value.disabled,
                )
                for color, label in value.items
            )
            sections.append(
                Section(
                    str(key),
                    [],
                    swatches=items,
                    swatch_reserve=str(value.reserve),
                )
            )
        elif isinstance(value, Controls):
            flush()
            sections.append(Section(str(key), [], controls=value.rows))
        elif _is_container(value):
            flush()
            sections.append(Section(str(key), _format_tree(value, 0)))
        else:
            scalars.append((0, f"{key}: ", True, _format_scalar(value)))
    flush()
    return sections or [Section(None, [])]


def _wrap(
    text: str,
    size: float,
    weight: int,
    max_width: float,
    mono: bool = False,
) -> list[str]:
    """Greedily wrap ``text`` to ``max_width`` using measured word widths."""
    if not text:
        return [""]
    wrapped: list[str] = []
    current = ""
    for word in text.split(" "):
        trial = f"{current} {word}".strip()
        if (
            not current
            or _MEASURE.measure_text(
                trial, size, weight=weight, mono=mono
            ).width
            <= max_width
        ):
            current = trial
        else:
            wrapped.append(current)
            current = word
    wrapped.append(current)
    return wrapped


# One draw op: (y, x, text, weight, color, mono). Values render monospace.
_Op = tuple[float, float, str, int, Color, bool]


def _line_ops(
    line: Line,
    content_w: float,
    m: _Metrics,
    row_h: float,
    ascent: float,
    y: float,
) -> tuple[list[_Op], float]:
    """Positioned text ops for one logical line; returns them and the next ``y``."""
    depth, prefix, is_key, body = line
    x = depth * m.indent
    weight = 600 if is_key else 400
    prefix_w = (
        _MEASURE.measure_text(prefix, m.size, weight=weight).width
        if prefix
        else 0.0
    )
    avail = max(24.0, content_w - x - prefix_w)
    # A value with no spaces to wrap on (e.g. a folded filename or path) can still
    # overrun the width, so trim any over-long line's middle to fit.
    body_lines = [
        _middle_ellipsize(part, avail, m)
        for part in _wrap(body, m.size, 400, avail, mono=True)
    ]
    ops: list[_Op] = []
    if prefix:
        color = m.key if is_key else m.value
        ops.append((y + ascent, x, prefix, weight, color, False))
    if body_lines[0]:
        ops.append(
            (y + ascent, x + prefix_w, body_lines[0], 400, m.value, True)
        )
    y += row_h
    for cont in body_lines[1:]:
        ops.append((y + ascent, x + prefix_w, cont, 400, m.value, True))
        y += row_h
    return ops, y


def _build_ops(
    lines: list[Line], content_w: float, m: _Metrics
) -> tuple[list[_Op], float]:
    """Lay out logical lines into positioned text ops; return them and total height."""
    metrics = _MEASURE.measure_text("Ag", m.size)
    row_h = metrics.height + m.line_gap
    ops: list[_Op] = []
    y = 0.0
    for line in lines:
        line_ops, y = _line_ops(line, content_w, m, row_h, metrics.ascent, y)
        ops.extend(line_ops)
    return ops, y


def _layout_body(
    canvas: Canvas | None,
    sections: list[Section],
    x0: float,
    y0: float,
    content_w: float,
    m: _Metrics,
    clicks: "list[tuple[Rect, str]] | None" = None,
) -> float:
    """Measure (``canvas`` ``None``) or draw the panel body; return its height.

    Walks the sections top-down: a rule between them, an uppercase label for each
    headed one, then its typed body — key/value text rows, a `Block` value on its
    own line, a `Swatches` color legend, or `Controls` rows. Positions are
    absolute (``x0``, ``y0 + local y``); the return value is the total local
    height so the caller can size the card before a second, drawing pass. When
    ``clicks`` is given, each control row and legend swatch appends its
    ``(region, action)`` to it (in the same absolute pixels) for a click map.
    """
    metrics = _MEASURE.measure_text("Ag", m.size)
    row_h = metrics.height + m.line_gap
    header = _MEASURE.measure_text("Ag", m.header_size, weight=_HEADER_WEIGHT)
    header_h = header.height + m.line_gap
    y = 0.0
    for i, section in enumerate(sections):
        if i > 0:
            y += m.section_gap
            if canvas is not None:
                canvas.line(
                    (x0, y0 + y),
                    (x0 + content_w, y0 + y),
                    m.divider,
                    width=1.0,
                )
            y += m.section_gap
        if section.heading is not None:
            if canvas is not None:
                _draw_tracked(
                    canvas,
                    x0,
                    y0 + y + header.ascent,
                    section.heading.upper(),
                    m.header_size,
                    m.muted,
                    _HEADER_WEIGHT,
                    _HEADER_TRACKING,
                )
            if section.swatches is not None:
                _layout_legend_toggle(
                    canvas,
                    section.swatches,
                    x0,
                    y0 + y,
                    content_w,
                    header.ascent,
                    header_h,
                    m,
                    clicks,
                )
            y += header_h
        if section.controls is not None:
            y = _layout_controls(
                canvas,
                section.controls,
                x0,
                y0,
                y,
                content_w,
                m,
                row_h,
                clicks,
            )
        elif section.swatches is not None:
            y = _layout_swatches(
                canvas,
                section.swatches,
                x0,
                y0,
                y,
                content_w,
                m,
                row_h,
                clicks,
                section.swatch_reserve,
            )
        elif section.block:
            if canvas is not None:
                # One full-width line, middle-ellipsized on overrun so a long
                # path keeps its start and end (e.g. the name and extension).
                value = _middle_ellipsize(section.lines[0][3], content_w, m)
                canvas.text(
                    (x0, y0 + y + metrics.ascent),
                    value,
                    size=m.size,
                    color=m.value,
                    weight=400,
                    mono=True,
                )
            y += row_h
        else:
            for line in section.lines:
                ops, y = _line_ops(
                    line, content_w, m, row_h, metrics.ascent, y
                )
                if canvas is not None:
                    for op_y, op_x, text, weight, color, mono in ops:
                        canvas.text(
                            (x0 + op_x, y0 + op_y),
                            text,
                            size=m.size,
                            color=color,
                            weight=weight,
                            mono=mono,
                        )
    return y


def _key_label(key: str) -> str:
    """Bracket a keycap label, unless it already carries its own brackets."""
    return key if key.startswith("[") else f"[{key}]"


def _control_key_width(
    rows: tuple[tuple[str, str, str, bool | None], ...], m: _Metrics
) -> float:
    """Width of the widest ``[key]`` cell, so the names align in a column."""
    return max(
        (
            _MEASURE.measure_text(
                _key_label(key), m.size, weight=600, mono=True
            ).width
            for key, _, _, _ in rows
        ),
        default=0.0,
    )


def _layout_controls(
    canvas: Canvas | None,
    rows: tuple[tuple[str, str, str, bool | None], ...],
    x0: float,
    y0: float,
    y: float,
    content_w: float,
    m: _Metrics,
    row_h: float,
    clicks: "list[tuple[Rect, str]] | None" = None,
) -> float:
    """Draw/measure control rows ``[key]  name … value`` (value tinted by state)."""
    ascent = _MEASURE.measure_text("Ag", m.size).ascent
    key_w = _control_key_width(rows, m)
    name_x = key_w + m.line_gap * 2
    for key, name, value, active in rows:
        if canvas is not None:
            baseline = y0 + y + ascent
            canvas.text(
                (x0, baseline),
                _key_label(key),
                size=m.size,
                color=m.key,
                weight=600,
                mono=True,
            )
            canvas.text(
                (x0 + name_x, baseline), name, size=m.size, color=m.value
            )
            tint = (
                m.active if active else m.muted if active is False else m.value
            )
            value_w = _MEASURE.measure_text(
                value, m.size, weight=600, mono=True
            ).width
            canvas.text(
                (x0 + content_w - value_w, baseline),
                value,
                size=m.size,
                color=tint,
                weight=600,
                mono=True,
            )
            if clicks is not None:
                clicks.append(
                    (
                        Rect(x0, y0 + y, x0 + content_w, y0 + y + row_h),
                        f"key:{key}",
                    )
                )
        y += row_h
    return y


#: The legend's master switch, beside the CLASSES heading: shows the action it
#: performs — "hide all" when every class is on, "show all" when any is off.
#: Both strings share a length, so the reserved width does not shift on toggle.
_LEGEND_HIDE_ALL = "hide all"
_LEGEND_SHOW_ALL = "show all"


def _legend_toggle_width(m: _Metrics) -> float:
    """Reserved width of the legend's master toggle (the wider of its two states)."""
    return max(
        _tracked_width(
            label.upper(), m.header_size, _HEADER_WEIGHT, _HEADER_TRACKING
        )
        for label in (_LEGEND_HIDE_ALL, _LEGEND_SHOW_ALL)
    )


def _layout_legend_toggle(
    canvas: Canvas | None,
    items: tuple[tuple[Color, str, bool], ...],
    x0: float,
    y: float,
    content_w: float,
    ascent: float,
    header_h: float,
    m: _Metrics,
    clicks: "list[tuple[Rect, str]] | None",
) -> None:
    """Draw the legend's master on/off switch, right-aligned on its heading row.

    Reads the current state from the swatches (any disabled → "show all", tinted
    to invite a click; all on → a muted "hide all") and registers a
    ``classes:toggle`` click over the switch so a click flips every class at once.
    """
    any_off = any(not enabled for _c, _l, enabled in items)
    label = (_LEGEND_SHOW_ALL if any_off else _LEGEND_HIDE_ALL).upper()
    width = _tracked_width(
        label, m.header_size, _HEADER_WEIGHT, _HEADER_TRACKING
    )
    x = x0 + content_w - width
    if canvas is not None:
        _draw_tracked(
            canvas,
            x,
            y + ascent,
            label,
            m.header_size,
            m.active if any_off else m.muted,
            _HEADER_WEIGHT,
            _HEADER_TRACKING,
        )
    if clicks is not None:
        clicks.append(
            (
                Rect(x - m.line_gap, y, x0 + content_w, y + header_h),
                "classes:toggle",
            )
        )


def _swatch_col_width(labels: list[str], m: _Metrics) -> float:
    """Width of one legend column: swatch + gap + widest label + gutter."""
    ascent = _MEASURE.measure_text("Ag", m.size).ascent
    square = round(ascent * 0.85)
    widest = max(
        (_MEASURE.measure_text(label, m.size).width for label in labels),
        default=0.0,
    )
    return square + m.line_gap + widest + m.indent


def _layout_swatches(
    canvas: Canvas | None,
    items: tuple[tuple[Color, str, bool], ...],
    x0: float,
    y0: float,
    y: float,
    content_w: float,
    m: _Metrics,
    row_h: float,
    clicks: "list[tuple[Rect, str]] | None" = None,
    reserve: str = "",
) -> float:
    """Draw/measure a color legend as an aligned grid of swatch+label cells.

    Every cell gets the same width (the widest chip, or the reserved label, plus a
    gutter), so the swatches line up in tidy columns; the legend folds into at
    most `_LEGEND_COLS` columns. A disabled class keeps its cell but is drawn
    dimmed and struck through. Each cell registers a ``class:<label>`` click.
    """
    ascent = _MEASURE.measure_text("Ag", m.size).ascent
    square = round(ascent * 0.85)
    label_gap = m.line_gap
    if not items:
        return y
    labels = [label for _, label, _ in items] + ([reserve] if reserve else [])
    col_w = _swatch_col_width(labels, m)
    gutter = m.indent
    cols = max(1, min(_LEGEND_COLS, int((content_w + gutter) // col_w)))
    for index, (color, label, enabled) in enumerate(items):
        row, col = divmod(index, cols)
        cx = x0 + col * col_w
        cy = y0 + y + row * row_h
        if canvas is not None:
            top = cy + (ascent - square)
            label_x = cx + square + label_gap
            baseline = cy + ascent
            if enabled:
                canvas.rounded_rect(
                    Rect(cx, top, cx + square, top + square),
                    radius=square * 0.28,
                    fill=color,
                )
                canvas.text(
                    (label_x, baseline), label, size=m.size, color=m.value
                )
            else:
                # Disabled: a hollow swatch and a muted, struck-through label.
                canvas.rounded_rect(
                    Rect(cx, top, cx + square, top + square),
                    radius=square * 0.28,
                    stroke=m.muted,
                    stroke_width=max(1.0, m.border_width),
                )
                canvas.text(
                    (label_x, baseline), label, size=m.size, color=m.muted
                )
                label_w = _MEASURE.measure_text(label, m.size).width
                strike = cy + ascent * 0.55
                canvas.line(
                    (label_x, strike),
                    (label_x + label_w, strike),
                    m.muted,
                    width=max(1.0, m.border_width),
                )
            if clicks is not None:
                clicks.append(
                    (
                        Rect(cx, cy, cx + col_w, cy + row_h),
                        f"class:{label}",
                    )
                )
    rows = (len(items) + cols - 1) // cols
    return y + rows * row_h


def _middle_ellipsize(text: str, max_width: float, m: _Metrics) -> str:
    """Trim ``text``'s middle with an ellipsis until it fits ``max_width`` (mono)."""

    def fits(candidate: str) -> bool:
        return (
            _MEASURE.measure_text(
                candidate, m.size, weight=400, mono=True
            ).width
            <= max_width
        )

    if fits(text):
        return text
    for keep in range(len(text) - 1, 0, -1):
        head, tail = (keep + 1) // 2, keep // 2
        candidate = text[:head] + "…" + (text[-tail:] if tail else "")
        if fits(candidate):
            return candidate
    return "…"


def _auto_width(
    sections: list[Section], title: str | None, m: _Metrics
) -> float:
    """Pick a panel width from the content, clamped to a sensible range."""
    widest = 0.0
    if title is not None:
        widest = _tracked_width(
            title.upper(),
            m.size * _TITLE_SCALE,
            _TITLE_WEIGHT,
            _TITLE_TRACKING,
        )
    for section in sections:
        if section.heading is not None:
            head_w = _tracked_width(
                section.heading.upper(),
                m.header_size,
                _HEADER_WEIGHT,
                _HEADER_TRACKING,
            )
            if section.swatches is not None:
                # The legend heading also carries a right-aligned master toggle.
                head_w += m.indent + _legend_toggle_width(m)
            widest = max(widest, head_w)
        if section.block:
            # A block value gets its own line and ellipsizes to fit, so it never
            # forces the panel wider — only its (already counted) label does.
            continue
        if section.controls is not None:
            key_w = _control_key_width(section.controls, m)
            for _key, name, value, _active in section.controls:
                name_w = _MEASURE.measure_text(name, m.size).width
                value_w = _MEASURE.measure_text(
                    value, m.size, weight=600, mono=True
                ).width
                row = key_w + m.line_gap * 2 + name_w + m.indent + value_w
                widest = max(widest, row)
            continue
        if section.swatches is not None:
            # Reserve the full two-column legend width up front (from the widest
            # label, or the reserved one), so the panel keeps a stable width and
            # the legend can fold into `_LEGEND_COLS` columns.
            labels = [label for _c, label, _e in section.swatches]
            if section.swatch_reserve:
                labels.append(section.swatch_reserve)
            col_w = _swatch_col_width(labels, m)
            widest = max(widest, _LEGEND_COLS * col_w - m.indent)
            continue
        for depth, prefix, is_key, body in section.lines:
            weight = 600 if is_key else 400
            prefix_w = _MEASURE.measure_text(
                prefix, m.size, weight=weight
            ).width
            body_w = _MEASURE.measure_text(
                body, m.size, weight=400, mono=True
            ).width
            widest = max(widest, depth * m.indent + prefix_w + body_w)
    return min(m.max_width, max(m.min_width, widest + 2 * m.pad))


def _tracked_width(
    text: str, size: float, weight: int, tracking_frac: float
) -> float:
    """Width of an uppercased, letter-spaced label at ``size``."""
    tracking = size * tracking_frac
    total = 0.0
    for char in text:
        total += (
            _MEASURE.measure_text(char, size, weight=weight).width + tracking
        )
    return total


def with_panel(
    image: Image,
    data: "PanelData",
    *,
    side: str = "right",
    width: float | None = None,
    title: str | None = None,
    style: Style | None = None,
    bg: ColorLike | None = None,
) -> Image:
    """Render an image and append a non-overlapping metadata panel.

    Mappings and sequences are flattened into an indented tree; long scalar
    values wrap to the available width. A right or left panel may increase the
    output height to fit its content. A bottom panel keeps the source image above
    the panel. The input image is rendered but not mutated.

    .. image:: TODO-HOST/panel.png
       :alt: A metadata side panel beside the annotated image.

    Args:
        image: The image to annotate. Rendered at native resolution.
        data: JSON-like metadata (mapping/sequence/scalar, nested arbitrarily).
        side: Which edge to attach the panel to: ``"right"`` (default), ``"left"``,
            or ``"bottom"``.
        width: Panel width in pixels for every side; ``None`` auto-sizes from
            the content. For a bottom panel this also sets the content width.
        title: Optional bold heading drawn above the tree.
        style: Style whose font is used (defaults to the library default).
        bg: Panel background color; defaults to the image's theme background.

    Returns:
        A new `Image` of the image plus the panel. The input is
        not mutated.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Image, with_panel
        >>> image = Image(np.zeros((20, 30, 3), np.uint8))
        >>> data = {"source": "frame.jpg", "augmentations": ["flip", "blur"]}
        >>> with_panel(image, data).render().shape[1] > image.width
        True

    """
    return _compose_panel(
        image, data, side=side, width=width, title=title, style=style, bg=bg
    )[0]


def _compose_panel(
    image: Image,
    data: "PanelData",
    *,
    side: str = "right",
    width: float | None = None,
    title: str | None = None,
    style: Style | None = None,
    bg: ColorLike | None = None,
) -> tuple[Image, tuple[float, float], list[tuple[Rect, str]]]:
    """Render the framed image-plus-panel and report where the image landed.

    The image and the panel are drawn as two separate rounded surfaces — each
    bordered, with a uniform outer margin and a gap between them — floating on the
    composite background, and the panel's sections are ruled apart. Returns the
    composed `Image`, the ``(dx, dy)`` the source image was translated by (so a
    caller carrying a hover `HitMap` can shift it to stay aligned), and the
    ``(region, action)`` click targets of the panel's controls and legend swatches
    in composed-image pixels (see `luxonis_ml.vizlab.frame.Frame.with_panel`).
    """
    style = style or DEFAULT_STYLE
    base = image.render()
    img_h, img_w = base.shape[:2]
    background = Color.parse(bg) if bg is not None else image.theme.background
    m = _metrics(_style_scale(img_w, img_h), background)

    sections = _format_sections(data)
    panel_w = width if width is not None else _auto_width(sections, title, m)
    content_w = panel_w - 2 * m.pad
    # A color legend (Swatches) is pinned to the card's bottom and grows upward,
    # so the controls and metadata above it never shift as the per-sample class
    # set changes — the rest of the panel stays put frame to frame.
    body = [s for s in sections if s.swatches is None]
    footer = [s for s in sections if s.swatches is not None]
    body_h = _layout_body(None, body, 0.0, 0.0, content_w, m)
    footer_h = (
        m.section_gap * 2 + _layout_body(None, footer, 0.0, 0.0, content_w, m)
        if footer
        else 0.0
    )

    title_metrics = (
        _MEASURE.measure_text(
            title, m.size * _TITLE_SCALE, weight=_TITLE_WEIGHT
        )
        if title is not None
        else None
    )
    title_h = title_metrics.height + m.line_gap * 2 if title_metrics else 0.0
    panel_h = 2 * m.pad + title_h + body_h + footer_h

    if side in ("right", "left"):
        surface_h = max(float(img_h), panel_h)  # stretch both to the taller
        out_w = int(2 * m.margin + m.gap + img_w + panel_w)
        out_h = int(2 * m.margin + surface_h)
        image_y = panel_y = m.margin
        panel_card_h = surface_h
        if side == "left":
            panel_x = m.margin
            image_x = m.margin + panel_w + m.gap
        else:
            image_x = m.margin
            panel_x = m.margin + img_w + m.gap
    else:  # bottom
        out_w = int(2 * m.margin + max(float(img_w), panel_w))
        out_h = int(2 * m.margin + m.gap + img_h + panel_h)
        image_x = panel_x = m.margin
        image_y = m.margin
        panel_y = m.margin + img_h + m.gap
        panel_card_h = panel_h

    canvas = Canvas.blank(out_w, out_h)
    # The "ultimate" background is inverted vs the theme, so the surfaces read as
    # distinct floating cards and the gap between them is clearly visible.
    canvas.rounded_rect(Rect(0, 0, out_w, out_h), 0.0, fill=m.page)
    # The image, as its own rounded, bordered surface.
    canvas.blit(base, image_x, image_y, radius=m.radius)
    canvas.rounded_rect(
        Rect(image_x, image_y, image_x + img_w, image_y + img_h),
        m.radius,
        stroke=m.border,
        stroke_width=m.border_width,
    )
    # The panel, as its own opaque rounded card (opaque so the inverted page does
    # not bleed through its translucency and wash the text out).
    canvas.rounded_rect(
        Rect(panel_x, panel_y, panel_x + panel_w, panel_y + panel_card_h),
        m.radius,
        fill=m.surface,
        stroke=m.border,
        stroke_width=m.border_width,
    )
    x0 = panel_x + m.pad
    y0 = panel_y + m.pad
    if title is not None and title_metrics is not None:
        _draw_tracked(
            canvas,
            x0,
            y0 + title_metrics.ascent,
            title.upper(),
            m.size * _TITLE_SCALE,
            m.title,
            _TITLE_WEIGHT,
            _TITLE_TRACKING,
        )
        y0 += title_metrics.height + m.line_gap * 2
    clicks: list[tuple[Rect, str]] = []
    _layout_body(canvas, body, x0, y0, content_w, m, clicks)
    if footer:
        # Pin the legend to the bottom: its own top sits so its bottom lands at
        # the card's inner edge, with a separating rule just above it. When the
        # card is taller than the content, the space opens up above the legend,
        # not below — so it appears to grow upward.
        inner_h = _layout_body(None, footer, 0.0, 0.0, content_w, m)
        footer_top = panel_y + panel_card_h - m.pad - inner_h
        rule_y = footer_top - m.section_gap
        canvas.line(
            (x0, rule_y), (x0 + content_w, rule_y), m.divider, width=1.0
        )
        _layout_body(canvas, footer, x0, footer_top, content_w, m, clicks)
    composed = Image(canvas.to_rgba(), options=image.options)
    return composed, (float(image_x), float(image_y)), clicks


def _draw_tracked(
    canvas: Canvas,
    x: float,
    baseline: float,
    text: str,
    size: float,
    color: Color,
    weight: int,
    tracking_frac: float,
) -> None:
    """Draw ``text`` letter-spaced (per-character advance) from ``x``, ``baseline``."""
    tracking = size * tracking_frac
    for char in text:
        canvas.text((x, baseline), char, size=size, color=color, weight=weight)
        x += _MEASURE.measure_text(char, size, weight=weight).width + tracking
