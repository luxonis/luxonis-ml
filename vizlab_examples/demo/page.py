"""Draw a documentation page — prose, code, and picture — with vizlab.

The toolkit behind `vizlab_examples.demo`: it turns a `Card` (a title, some
prose, a snippet, and whatever picture that snippet evaluates to) into one
rendered page, using nothing but vizlab's own drawing surface. The snippet a
card shows is the snippet that produced its picture, because the runner
executes it and renders the value it evaluates to.

Two pieces here are working around gaps in vizlab rather than using it:

- `text_block` draws its rows one at a time with `Canvas.markup` instead of
  using an `InfoCard`. `InfoCard` wraps its rows, wrapping splits on
  whitespace, and so a snippet's leading indentation is lost — a non-breaking
  space does not help either, since Python counts U+00A0 as whitespace. A
  ``CodeBlock`` annotation would make this unnecessary.
- `stack` and `row` are a small flow layout. `vstack`/`hstack`/`grid` are
  uniform-cell grids — every cell is as tall as the tallest input — which is
  right for a comparison strip and wrong for a document, where a short prose
  block above a tall figure should not inherit the figure's height.

Everything else is vizlab as it ships: the bundled JetBrains Mono reached
through ``<code>``, the inline-markup vocabulary for syntax coloring, and
`Renderable.render` for the pictures.
"""

import builtins
import io
import itertools
import keyword
import re
import tokenize
from collections.abc import Callable

import numpy as np

from luxonis_ml.vizlab import (
    Color,
    HitMap,
    Rect,
    Renderable,
    Tooltip,
    escape,
)
from luxonis_ml.vizlab.render.canvas import Canvas

#: Page and card fills. The card sits a touch lighter than the page so a block
#: of text reads as its own surface rather than a hole in the composition.
PAGE_BG = (14, 16, 22)
CARD_BG = Color(24, 27, 36)
HEADING = Color(232, 238, 247)
PROSE_FG = Color(190, 200, 214)
CODE_FG = Color.parse("#dbe2ee")
SECTION_FG = Color(126, 231, 205)

#: How big the whole deck is drawn, against a 1600x900 design. Every length and
#: type size below is derived from it, and the slides size their scenes from
#: the picture frame, so this one number scales the deck without changing its
#: proportions. Bigger is worth it: the viewer re-*renders* a slide to fit the
#: screen rather than resampling it, so a larger design is sharper on a large
#: display and no worse on a small one, and the saved ONGs gain the resolution.
SCALE = 1.4
PROSE_COLUMNS = 54


def scaled(length: float) -> int:
    """Return a design length in the deck's own pixels."""
    return round(length * SCALE)


#: Inner padding of a card, gutter between blocks, margin around the page.
PAD, GUTTER, MARGIN = scaled(20), scaled(18), scaled(26)
#: Type sizes and line spacing for the two kinds of text block.
PROSE_SIZE, PROSE_LEADING = 15.0 * SCALE, 1.55
CODE_SIZE, CODE_LEADING = 13.0 * SCALE, 1.5
#: How far a picture may be scaled to meet the text column's height. It is
#: never enlarged past its natural size — an upscaled render is a soft render,
#: and every picture here is already authored at the size it wants. It may be
#: shrunk a little to sit beside a short snippet; past that the row carries the
#: slack, because a 90-line column has no illustration tall enough to match it.
MIN_SCALE, MAX_SCALE = 0.75, 1.0

#: Wraps a glossary term inside a marked-up row. The pair never reaches the
#: canvas — it only tells `text_block` where one segment ends and the next
#: begins, so a term can be measured, painted and hit-tested on its own.
TERM_OPEN, TERM_CLOSE = "\x01", "\x02"
#: The ends of the ramp a glossary term is painted with.
TERM_FROM, TERM_TO = Color(126, 231, 205), Color(139, 178, 255)

_PROBE = Canvas.blank(1, 1)


def _hex(color: Color) -> str:
    """Return ``#rrggbb`` for a color, for use inside a markup tag."""
    return f"#{color.r:02x}{color.g:02x}{color.b:02x}"


def gradient_span(text: str) -> str:
    """Return ``text`` in bold, its color ramped across the characters.

    A flat accent would do the same job, but the deck is a showcase and the
    ramp is one more thing the markup vocabulary can already do — every
    character is its own ``<span color=...>``.
    """
    last = max(len(text) - 1, 1)
    parts = []
    for index, character in enumerate(text):
        share = index / last
        color = Color(
            round(TERM_FROM.r + (TERM_TO.r - TERM_FROM.r) * share),
            round(TERM_FROM.g + (TERM_TO.g - TERM_FROM.g) * share),
            round(TERM_FROM.b + (TERM_TO.b - TERM_FROM.b) * share),
        )
        parts.append(f'<span color="{_hex(color)}">{escape(character)}</span>')
    return f"<b>{''.join(parts)}</b>"


def term_chunk(name: str) -> str:
    """Return a self-contained, hit-testable chunk for a glossary term."""
    return f"{TERM_OPEN}<code>{gradient_span(name)}</code>{TERM_CLOSE}"


def split_terms(row: str) -> "list[tuple[str, bool]]":
    """Split a marked-up row into ``(markup, is_term)`` segments."""
    segments: list[tuple[str, bool]] = []
    rest = row
    while TERM_OPEN in rest:
        head, _, rest = rest.partition(TERM_OPEN)
        term, _, rest = rest.partition(TERM_CLOSE)
        if head:
            segments.append((head, False))
        segments.append((term, True))
    if rest:
        segments.append((rest, False))
    return segments


def plain(markup: str) -> str:
    """Strip every tag and marker, leaving the text as drawn."""
    text = re.sub(r"<[^>]+>", "", markup)
    return text.replace(TERM_OPEN, "").replace(TERM_CLOSE, "")


# --- markdown and Python, both as inline markup -----------------------------


def markdown(text: str, terms: "frozenset[str]" = frozenset()) -> str:
    """Convert the markdown a tour's prose uses into vizlab markup.

    Handles the four forms the cells actually use: a backticked name becomes
    monospace, ``**word**`` bold, ``*word*`` italic, and a leading ``- ``
    becomes a bullet. The text is escaped first, so a stray ``<`` stays
    literal.
    """
    bullet = text.lstrip().startswith("- ")
    body = text.lstrip()[2:] if bullet else text
    marked = escape(body)
    marked = re.sub(
        r"`([^`]+)`",
        lambda m: (
            term_chunk(m.group(1))
            if m.group(1) in terms
            else f"<code>{m.group(1)}</code>"
        ),
        marked,
    )
    marked = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", marked)
    marked = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", marked)
    return f"• {marked}" if bullet else marked


#: Token kind -> color, tuned for the card fill above.
_SYNTAX = {
    "keyword": "#c792ea",
    "string": "#7ee787",
    "number": "#ffab70",
    "comment": "#6b7589",
    "builtin": "#79c0ff",
    "call": "#5fd7c0",
    "op": "#8a93a5",
}


def _token_kind(token: tokenize.TokenInfo, following: str) -> str | None:
    """Classify one token, or ``None`` to leave it the default foreground."""
    if token.type == tokenize.COMMENT:
        return "comment"
    if token.type == tokenize.STRING:
        return "string"
    if token.type == tokenize.NUMBER:
        return "number"
    if token.type == tokenize.OP:
        return "op"
    if token.type == tokenize.NAME:
        if keyword.iskeyword(token.string):
            return "keyword"
        if hasattr(builtins, token.string):
            return "builtin"
        # A name immediately followed by "(" is being called — the one cue that
        # separates the API surface from the locals holding its results.
        return "call" if following == "(" else None
    return None


def highlight(source: str, terms: "frozenset[str]" = frozenset()) -> list[str]:
    """Return one marked-up line per source line, monospaced and colored.

    Colors are painted per character and then run-length encoded, so a token
    spanning several lines (a triple-quoted string) colors each of them
    correctly. Token text is escaped before it becomes markup, so a ``<`` in
    the code cannot open a tag. Source that does not tokenize — a partial
    snippet — still renders, just without colors.
    """
    lines = source.splitlines()
    kinds: list[list[str | None]] = [[None] * len(line) for line in lines]

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = []
    for index, token in enumerate(tokens):
        following = next(
            (
                t.string
                for t in tokens[index + 1 :]
                if t.type not in (tokenize.NL, tokenize.NEWLINE)
            ),
            "",
        )
        kind = (
            "term"
            if token.type == tokenize.NAME and token.string in terms
            else _token_kind(token, following)
        )
        if kind is None:
            continue
        (row0, col0), (row1, col1) = token.start, token.end
        for row in range(row0, row1 + 1):
            painted = kinds[row - 1]
            start = col0 if row == row0 else 0
            stop = col1 if row == row1 else len(painted)
            for column in range(start, min(stop, len(painted))):
                painted[column] = kind

    marked: list[str] = []
    for line, painted in zip(lines, kinds, strict=True):
        parts = []
        for kind, group in itertools.groupby(
            zip(line, painted, strict=True), key=lambda pair: pair[1]
        ):
            run = "".join(character for character, _ in group)
            if kind == "term":
                # Close the line's code span, drop in a self-contained chunk,
                # and reopen — so splitting on the markers leaves every
                # segment's tags balanced and separately measurable.
                parts.append(f"</code>{term_chunk(run)}<code>")
                continue
            text = escape(run)
            if kind is not None:
                text = f'<span color="{_SYNTAX[kind]}">{text}</span>'
            parts.append(text)
        marked.append(f"<code>{''.join(parts)}</code>")
    return marked


# --- text blocks, drawn a line at a time so indentation survives ------------


def text_block(
    rows: list[str],
    *,
    title: str | None = None,
    size: float,
    color: Color,
    leading: float,
    width: int | None = None,
    title_color: Color = HEADING,
    regions: "list[tuple[Rect, str]] | None" = None,
) -> np.ndarray:
    """Draw ``rows`` as a rounded card, one canvas line per row.

    Each row goes through `Canvas.markup`, which lays out a single line and
    never wraps, so whatever indentation a row carries is what gets drawn.

    Returns:
        An ``(H, W, 4)`` RGBA array: the card, with nothing around it.

    """
    title_size = size * 1.2
    widths = [_row_width(row, size) for row in rows] or [0.0]
    if title is not None:
        widths.append(_PROBE.measure_markup(title, title_size).width)
    card_w = width or int(max(widths)) + 2 * PAD
    line_h = size * leading
    head_h = title_size * 1.9 if title is not None else 0.0
    card_h = int(2 * PAD + head_h + line_h * len(rows))

    canvas = Canvas.blank(max(card_w, 1), max(card_h, 1))
    canvas.rounded_rect(
        Rect(0, 0, card_w, card_h), radius=scaled(10.0), fill=CARD_BG
    )
    y = PAD + size
    if title is not None:
        canvas.markup(
            (PAD, PAD + title_size),
            title,
            size=title_size,
            color=title_color,
            weight=700,
        )
        y += head_h
    for row in rows:
        x = float(PAD)
        for segment, is_term in split_terms(row):
            canvas.markup((x, y), segment, size=size, color=color)
            span = _PROBE.measure_markup(segment, size).width
            if is_term and regions is not None:
                regions.append(
                    (
                        Rect(x, y - size, x + span, y + size * 0.32),
                        plain(segment),
                    )
                )
            x += span
        y += line_h
    return canvas.to_rgba()


def _row_width(row: str, size: float) -> float:
    """Width of a row as drawn, summed over its segments."""
    return (
        sum(
            _PROBE.measure_markup(segment, size).width
            for segment, _ in split_terms(row)
        )
        or 0.0
    )


#: Stands in for a space inside a term chunk while a line is being wrapped.
#: ``<span color="...">`` contains a space, and the wrapper breaks on spaces —
#: without this it happily splits a line in the middle of a tag, and the rest
#: of the tag is drawn as literal text.
_KEEP = "\x03"


def _protect(marked: str) -> str:
    """Hide the spaces inside term chunks from the wrapper."""
    return re.sub(
        r"\x01.*?\x02",
        lambda match: match.group(0).replace(" ", _KEEP),
        marked,
        flags=re.DOTALL,
    )


#: The tags `markdown` emits. They never nest in prose, so one open tag at a
#: time is all the wrapper has to track.
_TAG = re.compile(r"</?(?:code|b|i)>")


def _visible(text: str) -> int:
    """Length of ``text`` as drawn — tags take up no room on the line."""
    return len(plain(text))


def _after(state: "str | None", word: str) -> "str | None":
    """Return the tag left open once ``word`` has been read."""
    for tag in _TAG.findall(word):
        state = None if tag.startswith("</") else tag
    return state


def wrap_marked(marked: str, fits: "Callable[[str], bool]") -> list[str]:
    """Wrap already-marked-up text, breaking wherever ``fits`` says to.

    Wrapping has to happen after the markup is applied, not before: emphasis
    that straddles a line break would never match its own closing tag if each
    line were converted on its own. So a span still open at a break is closed
    at the end of the line and reopened at the start of the next.

    Args:
        marked: The text, already converted to vizlab markup.
        fits: Predicate on a candidate line — ``True`` while it still has room.
            Taking a predicate rather than a column count lets prose be wrapped
            to a measured pixel width, which is what a proportional face
            actually needs.

    Returns:
        One marked-up line per wrapped line.

    """
    lines: list[str] = []
    current: list[str] = []
    opened: str | None = None  # open at the start of `current`
    running: str | None = None  # open at the current position

    def compose(
        words: list[str], start: "str | None", end: "str | None"
    ) -> str:
        text = " ".join(words)
        if start:
            text = start + text
        if end:
            text += f"</{end[1:]}"
        return text.replace(_KEEP, " ")

    for word in _protect(marked).split(" "):
        candidate = compose([*current, word], opened, _after(running, word))
        if current and not fits(candidate):
            lines.append(compose(current, opened, running))
            current, opened = [], running
        current.append(word)
        running = _after(running, word)
    if current:
        lines.append(compose(current, opened, running))
    return lines


def fits_within(limit: float, size: float) -> "Callable[[str], bool]":
    """Return a `wrap_marked` predicate for a pixel width at a type size."""

    def fits(line: str) -> bool:
        return _PROBE.measure_markup(line, size).width <= limit

    return fits


def wrap_prose(
    paragraphs: list[str],
    *,
    width: int | None = None,
    columns: int = PROSE_COLUMNS,
    terms: "frozenset[str]" = frozenset(),
) -> list[str]:
    """Wrap prose paragraphs, blank line between each.

    Args:
        paragraphs: The paragraphs, in markdown.
        width: Card width in pixels; the text is wrapped to what is left after
            the card's padding. Prose set to a character count on a
            proportional face leaves a ragged band of unused card down one
            side, which is what this avoids.
        columns: Fallback measure, in characters, when no width is given.
        terms: Names to draw as hoverable glossary terms.

    Returns:
        One marked-up row per line, ready for `text_block`.

    """
    fits = (
        fits_within(width - 2 * PAD, PROSE_SIZE)
        if width is not None
        else (lambda line: _visible(line) <= columns)
    )
    rows: list[str] = []
    for paragraph in paragraphs:
        if rows:
            rows.append("")
        marked = markdown(" ".join(paragraph.split()), terms)
        wrapped = wrap_marked(marked, fits) or [""]
        if marked.startswith("• "):
            wrapped = [wrapped[0], *(f"  {line}" for line in wrapped[1:])]
        rows.extend(wrapped)
    return rows


def prose_block(
    paragraphs: list[str], *, title: str | None, width: int | None = None
) -> np.ndarray:
    """Draw the prose card: a title and its wrapped paragraphs."""
    return text_block(
        wrap_prose(paragraphs, width=width),
        title=None if title is None else markdown(title),
        size=PROSE_SIZE,
        color=PROSE_FG,
        leading=PROSE_LEADING,
        width=width,
    )


def code_block(
    source: str,
    *,
    width: int | None = None,
    terms: "frozenset[str]" = frozenset(),
    regions: "list[tuple[Rect, str]] | None" = None,
    size: float = CODE_SIZE,
) -> np.ndarray:
    """Draw the code card: ``source``, monospaced and syntax-colored."""
    return text_block(
        highlight(source.rstrip(), terms),
        regions=regions,
        title="code",
        size=size,
        color=CODE_FG,
        leading=CODE_LEADING,
        width=width,
    )


# --- flow layout: every block keeps its own height --------------------------


def _rgba(block: "np.ndarray | Renderable") -> np.ndarray:
    """Render a block to RGBA, whatever kind of block it is."""
    return block if isinstance(block, np.ndarray) else block.render()


def blit(page: np.ndarray, block: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite ``block`` onto ``page`` at ``(x, y)``, clipped to it."""
    height = min(block.shape[0], page.shape[0] - y)
    width = min(block.shape[1], page.shape[1] - x)
    if height <= 0 or width <= 0:
        return
    block = block[:height, :width]
    target = page[y : y + height, x : x + width, :3]
    if block.shape[2] == 4:
        alpha = block[:, :, 3:4].astype(np.float32) / 255.0
        target[:] = (block[:, :, :3] * alpha + target * (1.0 - alpha)).astype(
            np.uint8
        )
    else:
        target[:] = block[:, :, :3]


def page(width: int, height: int) -> np.ndarray:
    """Return a page-colored RGBA canvas to compose blocks onto."""
    canvas = np.empty((max(height, 1), max(width, 1), 4), dtype=np.uint8)
    canvas[:, :, :3] = PAGE_BG
    canvas[:, :, 3] = 255
    return canvas


def stack(
    blocks: list["np.ndarray | Renderable"],
    *,
    margin: int = MARGIN,
    gutter: int = GUTTER,
    center: bool = False,
) -> np.ndarray:
    """Stack blocks top to bottom, each keeping its own height."""
    arrays = [_rgba(block) for block in blocks]
    width = max(a.shape[1] for a in arrays) + 2 * margin
    height = (
        sum(a.shape[0] for a in arrays)
        + gutter * (len(arrays) - 1)
        + 2 * margin
    )
    canvas = page(width, height)
    y = margin
    for array in arrays:
        x = (width - array.shape[1]) // 2 if center else margin
        blit(canvas, array, x, y)
        y += array.shape[0] + gutter
    return canvas


def row(
    blocks: list["np.ndarray | Renderable"], *, margin: int = 0
) -> np.ndarray:
    """Place blocks left to right, top-aligned, each keeping its own width."""
    arrays = [_rgba(block) for block in blocks]
    width = (
        sum(a.shape[1] for a in arrays)
        + GUTTER * (len(arrays) - 1)
        + 2 * margin
    )
    height = max(a.shape[0] for a in arrays) + 2 * margin
    canvas = page(width, height)
    x = margin
    for array in arrays:
        blit(canvas, array, x, margin)
        x += array.shape[1] + GUTTER
    return canvas


def fit_picture(picture: Renderable, column_height: int) -> np.ndarray:
    """Render ``picture`` scaled toward ``column_height``, within limits.

    A picture beside a text column looks best when the two are the same
    height, but only up to a point: a very long snippet must not blow its
    illustration up to match, and a very short one must not shrink it to a
    stamp. The scale is clamped to ``[MIN_SCALE, MAX_SCALE]`` and the row
    carries whatever slack is left.
    """
    natural_w, natural_h = int(picture.width), int(picture.height)
    if natural_h <= 0:
        return _rgba(picture)
    scale = min(MAX_SCALE, max(MIN_SCALE, column_height / natural_h))
    return picture.render((round(natural_w * scale), round(natural_h * scale)))


# --- a presentation slide ---------------------------------------------------

#: Every slide is drawn at one size, the way slides in a deck are, so nothing
#: jumps as you page through. 16:9, large enough for a readable snippet.
SLIDE_W, SLIDE_H = scaled(1600), scaled(900)
#: Fallback width of the text column; a deck measures its own widest snippet
#: and passes that instead, so every slide's column lines up.
COLUMN_W = scaled(560)
#: However wide a snippet is, the column stops here — past this the picture
#: has no room left, and a snippet that long belongs on a slide of its own.
MAX_COLUMN_W = scaled(720)
#: Type for the slide's own chrome — its heading and its counter.
TITLE_SIZE, COUNTER_SIZE = 27.0 * SCALE, 17.0 * SCALE

#: The slide's fixed skeleton. Content is *top-aligned* to `CONTENT_TOP` rather
#: than centred in what is left: a centred block of variable height puts its
#: top edge somewhere new on every slide, so the reader has to find the start
#: of the text again each time they page. Anchored to a constant, the prose is
#: always in the same place and only the ragged bottom edge moves — and that
#: edge is invisible against the page.
CONTENT_TOP = scaled(132)
#: Left edge of the text column, and the gap from it to the picture.
CONTENT_X, COLUMN_GAP = MARGIN + scaled(18), scaled(40)
#: The picture's frame: fixed on every slide, so the picture's box is a
#: constant the eye can lock onto even as its contents change. The right edge
#: lines up with the counter's, so the page has one right margin.
PICTURE_RIGHT = SLIDE_W - MARGIN - scaled(18)
PICTURE_BOTTOM = SLIDE_H - scaled(64)
#: The progress rule along the foot of the page.
RULE_Y, RULE_H = SLIDE_H - scaled(40), 3.0 * SCALE
RULE_TRACK = Color(38, 43, 55)


def fitted_code_size(sources: list[str], prose_heights: list[int]) -> float:
    """Return one code size the deck's tallest snippet still fits inside.

    A snippet set narrow runs tall, and the column has a hard floor: the page
    ends. Rather than let one long snippet overflow — or give each slide its
    own type size, which would make the deck's code jump between slides — the
    whole deck drops to the largest size that still fits its worst case.

    Args:
        sources: Every snippet in the deck.
        prose_heights: Each slide's already-measured prose-card height.

    Returns:
        A size no larger than `CODE_SIZE`.

    """
    sizes = [CODE_SIZE]
    for source, prose_h in zip(sources, prose_heights, strict=True):
        rows = len(source.rstrip().splitlines())
        room = SLIDE_H - MARGIN - CONTENT_TOP - prose_h - GUTTER - 2 * PAD
        # card height = title (1.2 * 1.9 sizes) + one leading per row
        sizes.append(room / (1.2 * 1.9 + CODE_LEADING * max(rows, 1)))
    return max(min(sizes), 6.0)


def compose_slide(
    title: str,
    body: list[str],
    source: str,
    picture: "Renderable | np.ndarray | None",
    *,
    position: str = "",
    column_w: int = COLUMN_W,
    progress: float = 0.0,
    glossary: "dict[str, Tooltip] | None" = None,
    code_size: float = CODE_SIZE,
) -> "tuple[np.ndarray, HitMap]":
    """Lay one slide out: heading, prose, snippet, and the picture they made.

    Prose and code stack in a fixed-width column on the left, top-aligned to
    `CONTENT_TOP`; the picture is fitted into a frame of fixed size and centred
    in it. The canvas is always `SLIDE_W` x `SLIDE_H`, and every landmark on it
    — heading, column, picture frame, rule — is at the same place on every
    slide, so paging through the deck moves nothing but the content.

    Args:
        title: The slide's heading.
        body: Prose paragraphs, in markdown.
        source: The snippet that drew ``picture``.
        picture: What the snippet evaluated to, or ``None``.
        position: The counter text, e.g. ``"3 / 17"``.
        column_w: Width of the text column, shared across the deck.
        progress: How far through the deck this slide is, in ``[0, 1]``.
        code_size: Type size for the snippet, shared across the deck so the
            tallest one still fits the page (see `fitted_code_size`).
        glossary: ``{name: Tooltip}`` for the constants and helpers the snippet
            leans on. Each name is painted as a gradient term wherever it
            appears in the prose or the snippet, and explains itself on hover —
            the definitions live in ``SETUP``, which is never shown.

    Returns:
        The slide, and the hover map of whatever the picture drew — already
        shifted to where the picture landed, so a viewer can hit-test it
        against the slide's own pixels.

    """
    canvas = Canvas.blank(SLIDE_W, SLIDE_H)
    canvas.rounded_rect(Rect(0, 0, SLIDE_W, SLIDE_H), fill=Color(*PAGE_BG))
    canvas.markup(
        (CONTENT_X, MARGIN + TITLE_SIZE + scaled(6)),
        markdown(title),
        size=TITLE_SIZE,
        color=HEADING,
        weight=700,
    )
    if position:
        metrics = canvas.measure_markup(position, COUNTER_SIZE)
        canvas.markup(
            (PICTURE_RIGHT - metrics.width, MARGIN + TITLE_SIZE),
            position,
            size=COUNTER_SIZE,
            color=SECTION_FG,
        )
    # A rule along the foot: the deck is 17 screens deep and the counter alone
    # is a number to read, not a position to see.
    canvas.rounded_rect(
        Rect(CONTENT_X, RULE_Y, PICTURE_RIGHT, RULE_Y + RULE_H),
        radius=RULE_H / 2,
        fill=RULE_TRACK,
    )
    if progress > 0.0:
        travelled = (PICTURE_RIGHT - CONTENT_X) * min(progress, 1.0)
        canvas.rounded_rect(
            Rect(CONTENT_X, RULE_Y, CONTENT_X + travelled, RULE_Y + RULE_H),
            radius=RULE_H / 2,
            fill=SECTION_FG,
        )
    slide = canvas.to_rgba()

    terms = frozenset(glossary or ())
    prose_spots: list[tuple[Rect, str]] = []
    code_spots: list[tuple[Rect, str]] = []
    prose = text_block(
        wrap_prose(body, width=column_w, terms=terms),
        size=PROSE_SIZE,
        color=PROSE_FG,
        leading=PROSE_LEADING,
        width=column_w,
        regions=prose_spots,
    )
    code = code_block(
        source,
        width=column_w,
        terms=terms,
        regions=code_spots,
        size=code_size,
    )
    code_y = CONTENT_TOP + prose.shape[0] + GUTTER
    blit(slide, prose, CONTENT_X, CONTENT_TOP)
    blit(slide, code, CONTENT_X, code_y)

    # A name in the prose and the same name in the snippet both explain
    # themselves under the cursor: the constants a snippet leans on are
    # defined in `SETUP`, which the slide deliberately never shows.
    spots: list[tuple[Rect, Tooltip]] = []
    for origin_y, found in ((CONTENT_TOP, prose_spots), (code_y, code_spots)):
        for rect, name in found:
            tip = (glossary or {}).get(name)
            if tip is not None:
                spots.append(
                    (
                        Rect(
                            rect.left + CONTENT_X,
                            rect.top + origin_y,
                            rect.right + CONTENT_X,
                            rect.bottom + origin_y,
                        ),
                        tip,
                    )
                )

    hits = HitMap.empty()
    if picture is not None:
        x0 = CONTENT_X + column_w + COLUMN_GAP
        frame_w, frame_h = PICTURE_RIGHT - x0, PICTURE_BOTTOM - CONTENT_TOP
        drawn, hits = _fit_into(picture, frame_w, frame_h)
        at_x = x0 + (frame_w - drawn.shape[1]) // 2
        at_y = CONTENT_TOP
        blit(slide, drawn, at_x, at_y)
        hits = hits.offset(at_x, at_y)
    return slide, HitMap(items=[*spots, *hits.items])


def _fit_into(
    picture: "Renderable | np.ndarray", width: int, height: int
) -> "tuple[np.ndarray, HitMap]":
    """Render ``picture`` to fit inside ``width`` x ``height``, never larger.

    Uses `Renderable.render_hits`, so the hover map comes back in the very
    pixels the picture was drawn at — no rescaling of rectangles needed.
    """
    if isinstance(picture, np.ndarray):
        return picture, HitMap.empty()
    natural_w, natural_h = int(picture.width), int(picture.height)
    if natural_w <= 0 or natural_h <= 0:
        return picture.render(), HitMap.empty()
    scale = min(1.0, width / natural_w, height / natural_h)
    size = (max(1, round(natural_w * scale)), max(1, round(natural_h * scale)))
    return picture.render_hits(size)
