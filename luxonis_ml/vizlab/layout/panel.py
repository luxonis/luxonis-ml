"""Append a metadata sidebar ("second window") to an image.

`with_panel` renders an image and appends a panel that shows arbitrary
JSON-like metadata (augmentations, tags, source, filenames, ...) as an indented
key/value tree. Like the `compose` functions it renders the image at
native resolution and returns a new `Image`, so the panel
never occludes pixels or labels and the original is untouched.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias, TypeGuard

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.canvas import Canvas, TextMetrics
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.hitmap import InteractionCapture
from luxonis_ml.vizlab.image import Composite, Renderable, _style_scale
from luxonis_ml.vizlab.render import RenderEnvironment
from luxonis_ml.vizlab.style import DEFAULT_STYLE, Style


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
        interactive: Whether swatches register class-toggle click regions. Set
            this to ``False`` for informational color keys such as task colors.

    """

    items: tuple[tuple[ColorLike, str], ...]
    disabled: frozenset[str] = frozenset()
    reserve: str = ""
    interactive: bool = True


@dataclass(frozen=True)
class Controls:
    """Interactive control rows described by key, name, value, and state.

    The panel renders the value tinted by ``active`` (on, off, or neutral) in
    the interactive HUD beside the image.

    Attributes:
        rows: One ``(key, name, value, active)`` per control. ``active`` is
            ``True`` (engaged), ``False`` (off), or ``None`` (neutral).

    """

    rows: tuple[tuple[str, str, str, bool | None], ...]


#: JSON-like metadata a panel can render: a scalar, a `Block` / `Swatches` /
#: `Controls` field, or a mapping/sequence nested arbitrarily. Anything else is
#: stringified as a leaf.
PanelKey: TypeAlias = str
"""A JSON-compatible mapping key accepted by panel data."""

PanelLeaf: TypeAlias = (
    Block | Swatches | Controls | str | int | float | bool | None
)
"""A non-container value accepted by a panel."""

PanelContainer: TypeAlias = (
    Mapping[PanelKey, "PanelData"] | Sequence["PanelData"]
)
"""A recursively nested mapping or non-string sequence of panel values."""

PanelData: TypeAlias = (
    Mapping[PanelKey, "PanelData"] | Sequence["PanelData"] | PanelLeaf
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


def _is_sequence(value: PanelData) -> TypeGuard[Sequence[PanelData]]:
    """Whether ``value`` is a non-string sequence of panel values."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_container(value: PanelData) -> TypeGuard[PanelContainer]:
    """Whether a value is a nested container (mapping or non-string sequence)."""
    return isinstance(value, Mapping) or _is_sequence(value)


def _format_scalar(value: PanelData) -> str:
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
    if _is_sequence(data):
        return _sequence_lines(data, depth)
    return [(depth, "", False, _format_scalar(data))]


def _mapping_lines(
    data: Mapping[PanelKey, PanelData], depth: int
) -> list[Line]:
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


def _sequence_lines(data: Sequence[PanelData], depth: int) -> list[Line]:
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
    #: Whether the legend exposes class-toggle click regions.
    swatches_interactive: bool = True
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
                    swatches_interactive=value.interactive,
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
    """Lay out logical lines into positioned text operations."""
    metrics = _MEASURE.measure_text("Ag", m.size)
    row_h = metrics.height + m.line_gap
    ops: list[_Op] = []
    y = 0.0
    for line in lines:
        line_ops, y = _line_ops(
            line,
            content_w,
            m,
            row_h,
            metrics.ascent,
            y,
        )
        ops.extend(line_ops)
    return ops, y


@dataclass
class _BodyLayout:
    """Shared measurement/drawing state for a panel body."""

    canvas: Canvas | None
    x0: float
    y0: float
    content_w: float
    metrics: _Metrics
    clicks: list[tuple[Rect, str]] | None = None
    text: TextMetrics = field(init=False)
    row_h: float = field(init=False)
    header: TextMetrics = field(init=False)
    header_h: float = field(init=False)

    def __post_init__(self) -> None:
        self.text = _MEASURE.measure_text("Ag", self.metrics.size)
        self.row_h = self.text.height + self.metrics.line_gap
        self.header = _MEASURE.measure_text(
            "Ag",
            self.metrics.header_size,
            weight=_HEADER_WEIGHT,
        )
        self.header_h = self.header.height + self.metrics.line_gap

    def run(self, sections: list[Section]) -> float:
        """Lay out ``sections`` and return their total local height."""
        y = 0.0
        for index, section in enumerate(sections):
            if index:
                y = self._separator(y)
            y = self._heading(section, y)
            y = self._content(section, y)
        return y

    def _separator(self, y: float) -> float:
        """Draw one section rule and return the following content offset."""
        y += self.metrics.section_gap
        if self.canvas is not None:
            self.canvas.line(
                (self.x0, self.y0 + y),
                (self.x0 + self.content_w, self.y0 + y),
                self.metrics.divider,
                width=1.0,
            )
        return y + self.metrics.section_gap

    def _heading(self, section: Section, y: float) -> float:
        """Draw a section heading and optional legend master toggle."""
        if section.heading is None:
            return y
        if self.canvas is not None:
            _draw_tracked(
                self.canvas,
                self.x0,
                self.y0 + y + self.header.ascent,
                section.heading.upper(),
                self.metrics.header_size,
                self.metrics.muted,
                _HEADER_WEIGHT,
                _HEADER_TRACKING,
            )
        if section.swatches is not None and section.swatches_interactive:
            _layout_legend_toggle(
                self.canvas,
                section.swatches,
                self.x0,
                self.y0 + y,
                self.content_w,
                self.header.ascent,
                self.header_h,
                self.metrics,
                self.clicks,
            )
        return y + self.header_h

    def _content(self, section: Section, y: float) -> float:
        """Lay out the populated body variant of ``section``."""
        if section.controls is not None:
            return _layout_controls(
                self.canvas,
                section.controls,
                self.x0,
                self.y0,
                y,
                self.content_w,
                self.metrics,
                self.row_h,
                self.clicks,
            )
        if section.swatches is not None:
            clicks = self.clicks if section.swatches_interactive else None
            return _layout_swatches(
                self.canvas,
                section.swatches,
                self.x0,
                self.y0,
                y,
                self.content_w,
                self.metrics,
                self.row_h,
                clicks,
                section.swatch_reserve,
            )
        if section.block:
            return self._block(section, y)
        return self._lines(section.lines, y)

    def _block(self, section: Section, y: float) -> float:
        """Draw a full-width block value and return the next row offset."""
        if self.canvas is not None:
            value = _middle_ellipsize(
                section.lines[0][3],
                self.content_w,
                self.metrics,
            )
            self.canvas.text(
                (self.x0, self.y0 + y + self.text.ascent),
                value,
                size=self.metrics.size,
                color=self.metrics.value,
                weight=400,
                mono=True,
            )
        return y + self.row_h

    def _lines(self, lines: list[Line], y: float) -> float:
        """Draw ordinary wrapped key/value lines."""
        for line in lines:
            ops, y = _line_ops(
                line,
                self.content_w,
                self.metrics,
                self.row_h,
                self.text.ascent,
                y,
            )
            if self.canvas is None:
                continue
            for op_y, op_x, text, weight, color, mono in ops:
                self.canvas.text(
                    (self.x0 + op_x, self.y0 + op_y),
                    text,
                    size=self.metrics.size,
                    color=color,
                    weight=weight,
                    mono=mono,
                )
        return y


def _layout_body(
    canvas: Canvas | None,
    sections: list[Section],
    x0: float,
    y0: float,
    content_w: float,
    m: _Metrics,
    clicks: "list[tuple[Rect, str]] | None" = None,
) -> float:
    """Measure or draw the panel body and return its height."""
    return _BodyLayout(canvas, x0, y0, content_w, m, clicks).run(sections)


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
    """Draw or measure interactive control rows and collect click targets."""
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
                (Rect(cx, cy, cx + col_w, cy + row_h), f"class:{label}")
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
    widest = _title_width(title, m)
    for section in sections:
        widest = max(widest, _section_width(section, m))
    return min(m.max_width, max(m.min_width, widest + 2 * m.pad))


def _title_width(title: str | None, m: _Metrics) -> float:
    """Return the tracked title width, or zero when there is no title."""
    if title is None:
        return 0.0
    return _tracked_width(
        title.upper(),
        m.size * _TITLE_SCALE,
        _TITLE_WEIGHT,
        _TITLE_TRACKING,
    )


def _section_width(section: Section, m: _Metrics) -> float:
    """Return the natural content width of one typed panel section."""
    heading = _section_heading_width(section, m)
    if section.block:
        return heading
    if section.controls is not None:
        return max(heading, _controls_width(section.controls, m))
    if section.swatches is not None:
        return max(heading, _swatches_width(section, m))
    return max(heading, _lines_width(section.lines, m))


def _section_heading_width(section: Section, m: _Metrics) -> float:
    """Measure a section heading, including the legend master toggle."""
    if section.heading is None:
        return 0.0
    width = _tracked_width(
        section.heading.upper(),
        m.header_size,
        _HEADER_WEIGHT,
        _HEADER_TRACKING,
    )
    if section.swatches is not None:
        width += m.indent + _legend_toggle_width(m)
    return width


def _controls_width(
    rows: tuple[tuple[str, str, str, bool | None], ...],
    m: _Metrics,
) -> float:
    """Return the widest interactive-control row."""
    key_w = _control_key_width(rows, m)
    return max(
        (
            key_w
            + m.line_gap * 2
            + _MEASURE.measure_text(name, m.size).width
            + m.indent
            + _MEASURE.measure_text(
                value,
                m.size,
                weight=600,
                mono=True,
            ).width
            for _key, name, value, _active in rows
        ),
        default=0.0,
    )


def _swatches_width(section: Section, m: _Metrics) -> float:
    """Return the reserved multi-column width of a swatch section."""
    items = section.swatches or ()
    labels = [label for _color, label, _enabled in items]
    if section.swatch_reserve:
        labels.append(section.swatch_reserve)
    return _LEGEND_COLS * _swatch_col_width(labels, m) - m.indent


def _lines_width(lines: list[Line], m: _Metrics) -> float:
    """Return the widest unwrapped key/value line."""
    return max(
        (
            depth * m.indent
            + _MEASURE.measure_text(
                prefix,
                m.size,
                weight=600 if is_key else 400,
            ).width
            + _MEASURE.measure_text(
                body,
                m.size,
                weight=400,
                mono=True,
            ).width
            for depth, prefix, is_key, body in lines
        ),
        default=0.0,
    )


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


@dataclass(frozen=True)
class _PanelContent:
    """Measured body, footer, and title of one panel."""

    body: list[Section]
    footer: list[Section]
    footer_inner: float
    title_metrics: TextMetrics | None
    title_h: float
    panel_h: float


def _measure_panel_content(
    sections: list[Section],
    title: str | None,
    content_w: float,
    m: _Metrics,
) -> _PanelContent:
    """Split and measure the panel's flowing body and pinned footer."""
    body = [section for section in sections if section.swatches is None]
    footer = [section for section in sections if section.swatches is not None]
    body_h = _layout_body(None, body, 0.0, 0.0, content_w, m)
    footer_inner = (
        _layout_body(None, footer, 0.0, 0.0, content_w, m) if footer else 0.0
    )
    footer_h = m.section_gap * 2 + footer_inner if footer else 0.0
    title_metrics = (
        _MEASURE.measure_text(
            title,
            m.size * _TITLE_SCALE,
            weight=_TITLE_WEIGHT,
        )
        if title is not None
        else None
    )
    title_h = (
        title_metrics.height + m.line_gap * 2
        if title_metrics is not None
        else 0.0
    )
    return _PanelContent(
        body=body,
        footer=footer,
        footer_inner=footer_inner,
        title_metrics=title_metrics,
        title_h=title_h,
        panel_h=2 * m.pad + title_h + body_h + footer_h,
    )


@dataclass(frozen=True)
class _PanelPlacement:
    """Resolved canvas and card coordinates for one panel side."""

    out_w: int
    out_h: int
    image_x: float
    image_y: float
    panel_x: float
    panel_y: float
    panel_card_h: float


def _place_panel(
    side: str,
    image_size: tuple[int, int],
    panel_w: float,
    panel_h: float,
    m: _Metrics,
) -> _PanelPlacement:
    """Resolve image and panel cards for the requested side."""
    img_w, img_h = image_size
    if side not in ("right", "left"):
        return _PanelPlacement(
            out_w=int(2 * m.margin + max(float(img_w), panel_w)),
            out_h=int(2 * m.margin + m.gap + img_h + panel_h),
            image_x=m.margin,
            image_y=m.margin,
            panel_x=m.margin,
            panel_y=m.margin + img_h + m.gap,
            panel_card_h=panel_h,
        )

    surface_h = max(float(img_h), panel_h)
    panel_x = m.margin
    image_x = m.margin + panel_w + m.gap
    if side == "right":
        image_x = m.margin
        panel_x = m.margin + img_w + m.gap
    return _PanelPlacement(
        out_w=int(2 * m.margin + m.gap + img_w + panel_w),
        out_h=int(2 * m.margin + surface_h),
        image_x=image_x,
        image_y=m.margin,
        panel_x=panel_x,
        panel_y=m.margin,
        panel_card_h=surface_h,
    )


def _collect_panel_clicks(
    content: _PanelContent,
    *,
    x0: float,
    body_y0: float,
    footer_top: float,
    content_w: float,
    m: _Metrics,
) -> list[tuple[Rect, str]]:
    """Measure interactive rows at their final positions."""
    clicks: list[tuple[Rect, str]] = []
    _layout_body(
        None,
        content.body,
        x0,
        body_y0,
        content_w,
        m,
        clicks,
    )
    if content.footer:
        _layout_body(
            None,
            content.footer,
            x0,
            footer_top,
            content_w,
            m,
            clicks,
        )
    return clicks


@dataclass(frozen=True)
class _PanelPainter:
    """Paint a measured image-and-panel composition."""

    image: Renderable
    image_size: tuple[int, int]
    panel_w: float
    content_w: float
    title: str | None
    content: _PanelContent
    placement: _PanelPlacement
    metrics: _Metrics
    clicks: list[tuple[Rect, str]]

    @property
    def _x0(self) -> float:
        return self.placement.panel_x + self.metrics.pad

    @property
    def _body_y0(self) -> float:
        return self.placement.panel_y + self.metrics.pad + self.content.title_h

    @property
    def _footer_top(self) -> float:
        return (
            self.placement.panel_y
            + self.placement.panel_card_h
            - self.metrics.pad
            - self.content.footer_inner
        )

    def __call__(
        self,
        canvas: Canvas,
        environment: RenderEnvironment,
        capture: InteractionCapture | None,
    ) -> None:
        """Paint the scene and register its panel click regions."""
        self._draw_surfaces(canvas, environment, capture)
        self._draw_content(canvas)
        if capture is not None:
            for rect, action in self.clicks:
                capture.add_click(rect, action)

    def _draw_surfaces(
        self,
        canvas: Canvas,
        environment: RenderEnvironment,
        capture: InteractionCapture | None,
    ) -> None:
        """Draw the page, clipped image card, and panel card."""
        p = self.placement
        m = self.metrics
        img_w, img_h = self.image_size
        canvas.rounded_rect(Rect(0, 0, p.out_w, p.out_h), 0.0, fill=m.page)
        image_rect = Rect(
            p.image_x,
            p.image_y,
            p.image_x + img_w,
            p.image_y + img_h,
        )
        with canvas.clip_rounded(image_rect, m.radius):
            self.image._draw_onto(
                canvas,
                p.image_x,
                p.image_y,
                self.image_size,
                environment=environment,
                capture=capture,
            )
        canvas.rounded_rect(
            image_rect,
            m.radius,
            stroke=m.border,
            stroke_width=m.border_width,
        )
        canvas.rounded_rect(
            Rect(
                p.panel_x,
                p.panel_y,
                p.panel_x + self.panel_w,
                p.panel_y + p.panel_card_h,
            ),
            m.radius,
            fill=m.surface,
            stroke=m.border,
            stroke_width=m.border_width,
        )

    def _draw_content(self, canvas: Canvas) -> None:
        """Draw the optional title, flowing body, and pinned footer."""
        title_metrics = self.content.title_metrics
        if self.title is not None and title_metrics is not None:
            _draw_tracked(
                canvas,
                self._x0,
                self.placement.panel_y
                + self.metrics.pad
                + title_metrics.ascent,
                self.title.upper(),
                self.metrics.size * _TITLE_SCALE,
                self.metrics.title,
                _TITLE_WEIGHT,
                _TITLE_TRACKING,
            )
        _layout_body(
            canvas,
            self.content.body,
            self._x0,
            self._body_y0,
            self.content_w,
            self.metrics,
        )
        if not self.content.footer:
            return
        rule_y = self._footer_top - self.metrics.section_gap
        canvas.line(
            (self._x0, rule_y),
            (self._x0 + self.content_w, rule_y),
            self.metrics.divider,
            width=1.0,
        )
        _layout_body(
            canvas,
            self.content.footer,
            self._x0,
            self._footer_top,
            self.content_w,
            self.metrics,
        )


def with_panel(
    image: Renderable,
    data: "PanelData",
    *,
    side: str = "right",
    width: float | None = None,
    title: str | None = None,
    style: Style | None = None,
    bg: ColorLike | None = None,
) -> Renderable:
    """Attach a non-overlapping metadata panel, returning a target-aware scene.

    Mappings and sequences are flattened into an indented tree; long scalar
    values wrap to the available width. A right or left panel may increase the
    output height to fit its content. A bottom panel keeps the source image above
    the panel. The input image is not mutated.

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
        A `Composite` of the image plus the panel — render it to raster or SVG,
        or `save` it to either. The input image is not mutated.

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
    image: Renderable,
    data: "PanelData",
    *,
    side: str = "right",
    width: float | None = None,
    title: str | None = None,
    style: Style | None = None,
    bg: ColorLike | None = None,
) -> tuple[Renderable, tuple[float, float], list[tuple[Rect, str]]]:
    """Assemble the framed image-plus-panel as a lazy, target-aware `Composite`.

    The image and the panel are laid out as two separate rounded surfaces — each
    bordered, with a uniform outer margin and a gap between them — floating on the
    composite background, and the panel's sections are ruled apart. Nothing is
    rasterized here: a `Composite` is returned that draws the image (via its own
    `Image._draw_onto`, so its annotations stay vector in an SVG) and the panel
    chrome (rounded cards, title, key/value rows, legend) to raster or SVG.
    Also returns the ``(dx, dy)`` the source image was translated by (so a caller
    carrying a hover `HitMap` can shift it to stay aligned) and the ``(region,
    action)`` click targets of the panel's controls and legend swatches, in
    composed-image pixels (see `luxonis_ml.vizlab.frame.Frame.with_panel`).
    """
    _ = style or DEFAULT_STYLE
    # The image is placed at its display (render_at) size, not its source size.
    img_w, img_h = image._resolved_size(None)
    background = Color.parse(bg) if bg is not None else image.theme.background
    m = _metrics(_style_scale(img_w, img_h), background)

    sections = _format_sections(data)
    panel_w = width if width is not None else _auto_width(sections, title, m)
    content_w = panel_w - 2 * m.pad
    content = _measure_panel_content(sections, title, content_w, m)
    placement = _place_panel(
        side,
        (img_w, img_h),
        panel_w,
        content.panel_h,
        m,
    )
    x0 = placement.panel_x + m.pad
    body_y0 = placement.panel_y + m.pad + content.title_h
    footer_top = (
        placement.panel_y
        + placement.panel_card_h
        - m.pad
        - content.footer_inner
    )
    clicks = _collect_panel_clicks(
        content,
        x0=x0,
        body_y0=body_y0,
        footer_top=footer_top,
        content_w=content_w,
        m=m,
    )
    paint = _PanelPainter(
        image=image,
        image_size=(img_w, img_h),
        panel_w=panel_w,
        content_w=content_w,
        title=title,
        content=content,
        placement=placement,
        metrics=m,
        clicks=clicks,
    )
    composite = Composite(
        (placement.out_w, placement.out_h),
        paint,
        options=image.options,
    )
    return (
        composite,
        (float(placement.image_x), float(placement.image_y)),
        clicks,
    )


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
