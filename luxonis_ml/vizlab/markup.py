"""A tiny inline-markup vocabulary for styled text runs.

Text drawn through the rich-text path may carry a handful of HTML-like tags to
switch weight, slant, or family mid-string:

- ``<b>…</b>`` (alias ``<strong>``) — bold
- ``<i>…</i>`` (alias ``<em>``) — italic
- ``<code>…</code>`` (alias ``<mono>``) — monospace (JetBrains Mono)

Tags nest and combine (``<b>a <i>b</i></b>``). Anything that is not one of these
exact tags — including a stray ``<`` — is treated as literal text, so ordinary
strings pass through untouched. `parse` turns a marked-up string into a flat list
of `Span`; the canvas measures and draws spans with the bundled fonts.

Examples:
    >>> from luxonis_ml.vizlab.markup import parse
    >>> [(s.text, s.weight, s.italic, s.mono) for s in parse("a <b>b</b> <i>c</i>")]
    [('a ', 400, False, False), ('b', 700, False, False), (' ', 400, False, False), ('c', 400, True, False)]

"""

import re
from dataclasses import dataclass

# Capturing split keeps the tags as their own tokens.
_TAG_RE = re.compile(r"(</?(?:b|strong|i|em|code|mono)>)", re.IGNORECASE)
_BOLD = frozenset({"b", "strong"})
_ITALIC = frozenset({"i", "em"})
_MONO = frozenset({"code", "mono"})


@dataclass(frozen=True)
class Span:
    """A run of text sharing one weight, slant, and family.

    Attributes:
        text: The literal characters (never contains markup tags).
        weight: OpenType weight (100-900) to render at.
        italic: Whether the run is italic.
        mono: Whether the run uses the monospace family.

    """

    text: str
    weight: int = 400
    italic: bool = False
    mono: bool = False


def _classify(token: str) -> "tuple[str, bool] | None":
    """Return ``(group, closing)`` for a tag token, or ``None`` for plain text."""
    if not (token.startswith("<") and token.endswith(">")):
        return None
    inner = token[1:-1].lower()
    closing = inner.startswith("/")
    name = inner[1:] if closing else inner
    if name in _BOLD:
        return "bold", closing
    if name in _ITALIC:
        return "italic", closing
    if name in _MONO:
        return "mono", closing
    return None  # pragma: no cover - regex only yields known tags


def parse(
    markup: str,
    *,
    weight: int = 400,
    italic: bool = False,
    mono: bool = False,
    bold_weight: int = 700,
) -> list[Span]:
    """Parse inline markup into a flat list of styled `Span`.

    Tags toggle a style on/off (nesting-aware); the surrounding ``weight`` /
    ``italic`` / ``mono`` set the baseline that tags layer on top of.

    Args:
        markup: The (possibly tagged) string.
        weight: Baseline OpenType weight for untagged text.
        italic: Whether untagged text is italic.
        mono: Whether untagged text uses the monospace family.
        bold_weight: Weight applied inside ``<b>`` runs.

    Returns:
        The spans in reading order (empty runs dropped). A tag-free string yields
        a single span.

    """
    depth = {"bold": 0, "italic": 0, "mono": 0}
    spans: list[Span] = []
    for token in _TAG_RE.split(markup):
        if not token:
            continue
        tag = _classify(token)
        if tag is None:
            spans.append(
                Span(
                    token,
                    bold_weight if depth["bold"] else weight,
                    italic or depth["italic"] > 0,
                    mono or depth["mono"] > 0,
                )
            )
            continue
        group, closing = tag
        depth[group] = (
            max(0, depth[group] - 1) if closing else depth[group] + 1
        )
    return spans
