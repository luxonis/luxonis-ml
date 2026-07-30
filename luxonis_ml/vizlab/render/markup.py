"""An inline-markup vocabulary for styled text runs.

Any string vizlab draws may carry HTML-like tags that switch weight, slant,
family, decoration, color, or size mid-line. The vocabulary follows `Pango
markup <https://docs.gtk.org/Pango/pango_markup.html>`_ — the same dialect GTK
and Matplotlib-adjacent tooling use — so it should already be familiar:

===========================  ==============================================
``<b>`` ``<strong>``         bold
``<i>`` ``<em>``             italic
``<u>``                      underline
``<s>`` ``<del>``            strikethrough
``<code>`` ``<tt>``          monospace (JetBrains Mono)
``<span …>``                 ``color``, ``weight``, and ``size`` attributes
===========================  ==============================================

Tags nest and combine (``<b>a <i>b</i></b>``), and are case-insensitive. The
``<span>`` attributes take the values Pango accepts:

    <span color="#ff6b6b">error</span>
    <span weight="semibold" size="120%">a touch louder</span>

To draw a literal ``<`` or ``&``, use the ``&lt;``, ``&gt;``, and ``&amp;``
entities, or call `escape` on text you did not author. Anything that is not a
recognized tag — including a bare ``<`` — is left as literal text, so ordinary
strings pass through untouched and no caller is forced to escape.

`parse` turns a marked-up string into a flat list of `Span`; `Canvas.markup`
and `Canvas.measure_markup` draw and measure one in a single call.

.. image:: TODO-HOST/markup.png
   :alt: Each tag beside its own rendered effect, then the same tags in a
         scene's labels, metadata panel, and hover tooltip.

Every string vizlab draws goes through this — annotation labels and captions,
grid and panel titles, panel keys and values, legend entries, chart labels, and
hover tooltips — so the same tags work everywhere.

Examples:
    >>> from luxonis_ml.vizlab.render.markup import parse
    >>> [
    ...     (s.text, s.weight, s.italic, s.mono)
    ...     for s in parse("a <b>b</b> <i>c</i>")
    ... ]
    [('a ', 400, False, False), ('b', 700, False, False), (' ', 400, False, False), ('c', 400, True, False)]
    >>> parse("<span size='150%'>big</span>")[0].scale
    1.5
    >>> parse("&lt;b&gt;")[0].text
    '<b>'

"""

import re
from dataclasses import dataclass, replace

from luxonis_ml.vizlab.color import Color

# Capturing split keeps the tags as their own tokens. Longest alternatives come
# first so ``<span …>`` is never mistaken for the ``<s>`` branch, and attribute
# text excludes angle brackets so a malformed tag cannot swallow the rest of the
# string — ``<span color="a>b">`` ends at the first ``>``, leaving an unterminated
# value that `_span_attributes` rejects with a clear error.
_TAG_RE = re.compile(
    r"(</?(?:strong|span|code|mono|del|em|tt|b|i|u|s)(?:\s[^<>]*)?>)",
    re.IGNORECASE,
)

#: Tags that toggle one boolean/weight attribute, keyed by their canonical name.
_SIMPLE_TAGS = {
    "b": "bold",
    "strong": "bold",
    "i": "italic",
    "em": "italic",
    "u": "underline",
    "s": "strike",
    "del": "strike",
    "code": "mono",
    "tt": "mono",
    "mono": "mono",
}

_ENTITY_RE = re.compile(r"&(lt|gt|amp|quot|apos);")
_ENTITIES = {
    "lt": "<",
    "gt": ">",
    "amp": "&",
    "quot": '"',
    "apos": "'",
}

_ATTR_RE = re.compile(r"""([A-Za-z_][\w-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""")

#: Pango's named weights, mapped to their OpenType values.
_WEIGHT_NAMES = {
    "thin": 100,
    "ultralight": 200,
    "extralight": 200,
    "light": 300,
    "semilight": 350,
    "book": 380,
    "normal": 400,
    "regular": 400,
    "medium": 500,
    "semibold": 600,
    "demibold": 600,
    "bold": 700,
    "ultrabold": 800,
    "extrabold": 800,
    "heavy": 900,
    "black": 900,
}

#: Pango's named sizes, as multipliers of the surrounding type size.
_SIZE_NAMES = {
    "xx-small": 1 / 1.2**3,
    "x-small": 1 / 1.2**2,
    "small": 1 / 1.2,
    "medium": 1.0,
    "large": 1.2,
    "x-large": 1.2**2,
    "xx-large": 1.2**3,
}


def escape(text: object) -> str:
    """Escape ``text`` so it renders verbatim rather than as markup.

    The counterpart to `parse`, and the right thing to call on any string vizlab
    did not author — dataset metadata, class names, file paths — before it is
    interpolated into text that will be parsed.

    Args:
        text: The value to escape; non-strings are stringified first.

    Returns:
        The text with ``&``, ``<``, and ``>`` replaced by their entities.

    Examples:
        >>> escape("<b>not bold</b>")
        '&lt;b&gt;not bold&lt;/b&gt;'
        >>> parse(escape("a < b & c"))[0].text
        'a < b & c'

    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@dataclass(frozen=True)
class Span:
    """A run of text sharing one weight, slant, family, decoration, and size.

    Attributes:
        text: The literal characters (never contains markup tags).
        weight: OpenType weight (100-900) to render at.
        italic: Whether the run is italic.
        mono: Whether the run uses the monospace family.
        underline: Whether the run is underlined.
        strike: Whether the run is struck through.
        color: An explicit color from ``<span color=…>``, or ``None`` to inherit
            whatever color the caller draws the line in.
        scale: Multiplier on the surrounding type size, from ``<span size=…>``.

    """

    text: str
    weight: int = 400
    italic: bool = False
    mono: bool = False
    underline: bool = False
    strike: bool = False
    color: "Color | None" = None
    scale: float = 1.0

    @property
    def decorated(self) -> bool:
        """Whether this run needs a decoration line drawn."""
        return self.underline or self.strike

    def with_text(self, text: str) -> "Span":
        """Return a copy of this span carrying ``text`` instead.

        Args:
            text: The replacement characters.

        Returns:
            An otherwise identical `Span`.

        """
        return replace(self, text=text)


def _parse_weight(value: str) -> int:
    """Resolve a ``<span weight=…>`` value to an OpenType weight."""
    named = _WEIGHT_NAMES.get(value.strip().lower())
    if named is not None:
        return named
    try:
        return int(float(value))
    except ValueError:
        raise ValueError(
            f"invalid markup weight {value!r}: expected 100-900 or one of "
            f"{', '.join(sorted(_WEIGHT_NAMES))}"
        ) from None


def _parse_scale(value: str) -> float:
    """Resolve a ``<span size=…>`` value to a multiplier of the ambient size."""
    text = value.strip().lower()
    named = _SIZE_NAMES.get(text)
    if named is not None:
        return named
    try:
        scale = float(text[:-1]) / 100.0 if text.endswith("%") else float(text)
    except ValueError:
        raise ValueError(
            f"invalid markup size {value!r}: expected a multiplier (``1.5``), a "
            f"percentage (``150%``), or one of {', '.join(_SIZE_NAMES)}"
        ) from None
    if scale <= 0.0:
        raise ValueError(f"invalid markup size {value!r}: must be positive")
    return scale


def _span_attributes(body: str) -> dict[str, object]:
    """Parse the attribute list of a ``<span …>`` tag into `Span` field values.

    Args:
        body: The text between the tag name and the closing ``>``.

    Returns:
        A mapping of `Span` field names to values, ready to apply.

    Raises:
        ValueError: If an attribute is unknown, malformed, or has an
            unparsable value.

    """
    fields: dict[str, object] = {}
    consumed = 0
    for match in _ATTR_RE.finditer(body):
        if body[consumed : match.start()].strip():
            break
        consumed = match.end()
        name = match.group(1).lower()
        value = (
            match.group(2) if match.group(2) is not None else match.group(3)
        )
        if name in ("color", "foreground", "fgcolor"):
            fields["color"] = Color.parse(value)
        elif name == "weight":
            fields["weight"] = _parse_weight(value)
        elif name == "size":
            fields["scale"] = _parse_scale(value)
        else:
            raise ValueError(
                f"unknown markup attribute {name!r} in <span>: vizlab supports "
                f"color, weight, and size"
            )
    if body[consumed:].strip():
        raise ValueError(
            f"malformed <span> attributes: {body[consumed:].strip()!r} "
            f'(attribute values must be quoted, e.g. color="#ff0000")'
        )
    return fields


def _decode(text: str) -> str:
    """Replace the supported XML entities with the characters they stand for."""
    if "&" not in text:
        return text
    return _ENTITY_RE.sub(lambda m: _ENTITIES[m.group(1)], text)


def _tag_parts(token: str) -> "tuple[str, bool, str] | None":
    """Split a tag token into ``(name, closing, attribute_body)``."""
    inner = token[1:-1]
    closing = inner.startswith("/")
    if closing:
        inner = inner[1:]
    name, _, body = inner.partition(" ")
    name = name.lower()
    if name not in _SIMPLE_TAGS and name != "span":
        return None  # pragma: no cover - the regex only yields known tags
    return name, closing, body


def parse(
    markup: str,
    *,
    weight: int = 400,
    italic: bool = False,
    mono: bool = False,
    bold_weight: int = 700,
) -> list[Span]:
    """Parse inline markup into a flat list of styled `Span`.

    Tags layer on top of the ``weight`` / ``italic`` / ``mono`` baseline, which
    is what the surrounding `Style` resolves to. Nesting is tracked with a stack:
    a closing tag closes everything opened inside it, and a closing tag with no
    matching open tag is ignored.

    Args:
        markup: The (possibly tagged) string.
        weight: Baseline OpenType weight for untagged text.
        italic: Whether untagged text is italic.
        mono: Whether untagged text uses the monospace family.
        bold_weight: Weight applied inside ``<b>`` runs.

    Returns:
        The spans in reading order (empty runs dropped). A tag-free string
        yields a single span.

    Raises:
        ValueError: If a ``<span>`` tag carries an unknown attribute or a value
            that cannot be parsed. Unrecognized *tags* never raise — they are
            literal text.

    Examples:
        >>> [
        ...     (s.text, s.underline, s.strike)
        ...     for s in parse("<u>a</u><s>b</s>")
        ... ]
        [('a', True, False), ('b', False, True)]
        >>> parse('<span color="#ff0000">x</span>')[0].color.rgb
        (255, 0, 0)

    """
    base = Span("", weight, italic, mono)
    current = base
    # Each entry is the style to restore when that tag closes.
    stack: list[tuple[str, Span]] = []
    spans: list[Span] = []
    for token in _TAG_RE.split(markup):
        if not token:
            continue
        parts = _tag_parts(token) if _TAG_RE.fullmatch(token) else None
        if parts is None:
            text = _decode(token)
            if text:
                spans.append(current.with_text(text))
            continue
        name, closing, body = parts
        if closing:
            current = _close(stack, name, current)
        else:
            stack.append((name, current))
            current = _open(current, name, body, bold_weight)
    return spans


def _open(current: Span, name: str, body: str, bold_weight: int) -> Span:
    """Return the style in effect inside a newly opened tag."""
    if name == "span":
        return replace(current, **_span_attributes(body))  # pyright: ignore[reportArgumentType]
    field = _SIMPLE_TAGS[name]
    if field == "bold":
        return replace(current, weight=bold_weight)
    return replace(current, **{field: True})


def _close(stack: list[tuple[str, Span]], name: str, current: Span) -> Span:
    """Close the nearest matching open tag, restoring the style it replaced.

    Tags opened inside the one being closed are closed with it, so crossed
    nesting (``<b>a<i>b</b>c</i>``) resolves the way an HTML parser resolves it
    rather than leaking style past the closing tag. An unmatched closing tag
    leaves the style untouched.
    """
    for index in range(len(stack) - 1, -1, -1):
        if stack[index][0] == name:
            restored = stack[index][1]
            del stack[index:]
            return restored
    return current
