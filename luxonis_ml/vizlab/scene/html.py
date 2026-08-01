"""Export a rendered scene as one self-contained, interactive HTML page.

A scene already knows everything a hover-capable web page needs: `Renderable`
draws itself to true vector SVG, and the same pass fills an `InteractionCapture`
with the region of every annotation that carries a `Tooltip`. `html_document`
welds the two together into a single file — inline SVG, inline CSS, inline
script, no external request — that a browser shows with working tooltips.

The hover regions ride *inside* the SVG, as transparent ``<rect>`` elements in
the drawing's own coordinate system. Re-rooting the document on a ``viewBox``
then makes one CSS rule (``width: 100%``) scale the picture and its hit regions
together, at every window width, with no JavaScript in the loop.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from html import escape as escape_html
from typing import NamedTuple

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab.color import Color, ColorLike
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.render.capture import HitMap
from luxonis_ml.vizlab.render.markup import Span, parse
from luxonis_ml.vizlab.style import Theme
from luxonis_ml.vizlab.tooltip import Tooltip

#: The bundled families, named so a browser that happens to have them installed
#: uses them, and falls back to its own stack otherwise. Naming them rather than
#: embedding the ~800 KB of variable fonts keeps the page small; the drawing
#: itself is unaffected, since its glyphs are already outlines.
_SANS = 'Inter, "Segoe UI", system-ui, -apple-system, sans-serif'
_MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

_STYLE = """
*,::before,::after{box-sizing:border-box}
html,body{margin:0}
body{background:var(--vl-bg);color:var(--vl-text);padding:16px;overflow-x:hidden;font-family:var(--vl-sans)}
.vl-figure{position:relative;width:100%;max-width:var(--vl-width);margin:0 auto}
.vl-figure>svg{display:block;width:100%;height:auto}
.vl-tip{position:absolute;left:0;top:0;display:none;pointer-events:none;z-index:1;max-width:min(26em,80%);padding:.7em;border-radius:.55em;font-size:13px;line-height:1.45;background:var(--vl-card-bg);border:1px solid var(--vl-card-border);box-shadow:0 .14em .5em rgba(0,0,0,.45);backdrop-filter:blur(.7em);-webkit-backdrop-filter:blur(.7em)}
.vl-head{display:flex;align-items:center;gap:.45em;margin-bottom:.4em}
.vl-head:last-child{margin-bottom:0}
.vl-swatch{flex:none;width:.95em;height:.95em;border-radius:.3em}
.vl-title{font-weight:700;font-size:1.06em;color:var(--vl-card-title)}
.vl-key{color:var(--vl-card-key);font-weight:600}
.vl-val{color:var(--vl-card-text);font-weight:500;font-family:var(--vl-mono)}
"""

# Hover is still resolved without script: `:has()` asks whether the region with
# a given index is under the pointer, and shows the matching card. The one thing
# CSS cannot read is where the pointer *is*, so this publishes it as two custom
# properties; the only other thing it does is tick the class checkboxes for the
# all/none buttons, because a selector can read a checkbox but never set one.
# Both are inputs to the stylesheet — nothing here decides what is visible.
# Which side of the cursor a card opens on is decided at emit time from its
# region and expressed as a transform, so the browser does the clamping.
_SCRIPT = """
(function () {
  var figure = document.querySelector(".vl-figure");
  figure.addEventListener("pointermove", function (event) {
    var box = figure.getBoundingClientRect();
    figure.style.setProperty("--vl-x", event.clientX - box.left + "px");
    figure.style.setProperty("--vl-y", event.clientY - box.top + "px");
  });

  document.querySelectorAll(".vl-bulk").forEach(function (button) {
    button.addEventListener("click", function () {
      document.querySelectorAll(".vl-class").forEach(function (box) {
        box.checked = !!button.dataset.check;
      });
    });
  });
})();
"""

#: Skia writes the SVG viewport as fixed ``width``/``height`` attributes and no
#: ``viewBox``; both are dropped in favor of one (see `_responsive_svg`).
_SIZE_ATTRS = re.compile(r'\s+(?:width|height)="[^"]*"')


def _css_color(value: ColorLike) -> str:
    """Render a color as a CSS literal, coercing it the way `Canvas` does."""
    color = Color.parse(value)
    if color.a == 255:
        return f"#{color.r:02x}{color.g:02x}{color.b:02x}"
    return f"rgba({color.r},{color.g},{color.b},{color.a / 255:.3f})"


def _num(value: float) -> str:
    """Format a coordinate for an SVG attribute, without trailing zeros."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _page_variables(theme: Theme, size: tuple[int, int]) -> str:
    """Emit the ``:root`` custom properties the stylesheet reads.

    The page takes its background from the scene's own theme and its card chrome
    from the `luxonis_ml.utils.color.brand` palette that suits that background,
    so a light-theme scene lands on a light page with dark-text cards.

    Args:
        theme: The theme the scene rendered with.
        size: The scene's ``(width, height)`` in render pixels.

    Returns:
        A ``:root { ... }`` CSS rule.

    """
    chrome = brand.chrome_for(theme.background)
    values = {
        "width": f"{size[0]}px",
        "sans": _SANS,
        "mono": _MONO,
        "bg": _css_color(theme.background),
        "text": _css_color(chrome.card_text),
        "card-bg": _css_color(chrome.card_bg),
        "card-text": _css_color(chrome.card_text),
        "card-title": _css_color(chrome.card_title),
        "card-key": _css_color(chrome.card_key),
        # Control chrome: an accent for the active state, and a hairline
        # that reads on either background without a second palette.
        "key": _css_color(chrome.card_key),
        "dim": _css_color(Color.parse(chrome.card_text).with_alpha(0.62)),
        "edge": _css_color(Color.parse(chrome.card_text).with_alpha(0.24)),
        # Dark cards lean on their shadow instead of an outline.
        "card-border": (
            "transparent"
            if chrome.border is None
            else _css_color(chrome.border)
        ),
    }
    body = "".join(f"--vl-{name}:{value};" for name, value in values.items())
    return f":root{{{body}}}"


def _span_style(span: Span, base: Span) -> str:
    """Return the inline CSS carrying whatever ``span`` changes from ``base``."""
    parts = []
    if span.weight != base.weight:
        parts.append(f"font-weight:{span.weight}")
    if span.italic != base.italic:
        parts.append(f"font-style:{'italic' if span.italic else 'normal'}")
    if span.mono != base.mono:
        family = "mono" if span.mono else "sans"
        parts.append(f"font-family:var(--vl-{family})")
    lines = []
    if span.underline:
        lines.append("underline")
    if span.strike:
        lines.append("line-through")
    if lines:
        parts.append(f"text-decoration:{' '.join(lines)}")
    if span.color is not None:
        parts.append(f"color:{_css_color(span.color)}")
    if span.scale != 1.0:
        parts.append(f"font-size:{span.scale:.4g}em")
    return ";".join(parts)


def markup_html(
    text: str, *, weight: int = 400, italic: bool = False, mono: bool = False
) -> str:
    """Translate one inline-markup string into styled, escaped HTML.

    The string is parsed by `luxonis_ml.vizlab.render.markup.parse`, so it obeys
    exactly the tags the rest of the library draws, and each resulting `Span`
    becomes a ``<span>`` carrying only what it changes from the baseline the
    surrounding CSS already supplies. Literal text is HTML-escaped, so arbitrary
    dataset strings render verbatim instead of reaching the browser as markup.

    Args:
        text: The (possibly tagged) string.
        weight: Baseline OpenType weight for untagged text.
        italic: Whether untagged text is italic.
        mono: Whether untagged text uses the monospace family.

    Returns:
        An HTML fragment safe to place in element content.

    Examples:
        >>> markup_html("a <b>b</b>")
        'a <span style="font-weight:700">b</span>'
        >>> markup_html("<script>")
        '&lt;script&gt;'

    """
    base = Span("", weight, italic, mono)
    out = []
    for span in parse(text, weight=weight, italic=italic, mono=mono):
        style = _span_style(span, base)
        body = escape_html(span.text)
        out.append(f'<span style="{style}">{body}</span>' if style else body)
    return "".join(out)


def _tooltip_html(tooltip: Tooltip) -> str:
    """Render a `Tooltip` as the body of a hover card.

    Follows the desktop card (see
    `luxonis_ml.vizlab.viewer.tooltip_render.render_tooltip_card`): a tinted
    swatch and bold title, then ``key: value`` rows with accent-colored keys and
    monospace values.
    """
    swatch = title = ""
    if tooltip.tint is not None:
        fill = _css_color(tooltip.tint)
        swatch = f'<i class="vl-swatch" style="background:{fill}"></i>'
    if tooltip.title:
        tint = (
            f' style="color:{_css_color(tooltip.tint)}"'
            if tooltip.tint is not None
            else ""
        )
        heading = markup_html(tooltip.title, weight=700)
        title = f'<span class="vl-title"{tint}>{heading}</span>'
    rows = []
    for key, value in tooltip.rows:
        name = markup_html(f"{key}: ", weight=600)
        text = markup_html(value, weight=500, mono=True)
        rows.append(
            f'<div><span class="vl-key">{name}</span>'
            f'<span class="vl-val">{text}</span></div>'
        )
    head = (
        f'<div class="vl-head">{swatch}{title}</div>'
        if swatch or title
        else ""
    )
    return head + "".join(rows)


def _hit_order(
    items: list[tuple[Rect, Tooltip]],
) -> list[tuple[Rect, Tooltip]]:
    """Order hit entries so the browser's topmost hit is `HitMap.hit`'s answer.

    A pointer lands on the last matching element in document order, so the
    winner has to come last: entries are sorted by ascending area — ties broken
    by draw order, as `HitMap.hit` breaks them — and then reversed.
    """
    ranked = sorted(
        enumerate(items), key=lambda pair: (pair[1][0].area, pair[0])
    )
    return [entry for _, entry in reversed(ranked)]


def _card_position(rect: Rect, size: tuple[int, int]) -> str:
    """Pin a card to the cursor, opening away from whichever edge it is near.

    The whole offset lives in a transform, and the card's layout box stays at
    the figure's origin. That is not a stylistic choice: an absolutely
    positioned box whose ``left`` is set gets a shrink-to-fit width bounded by
    the space *remaining to its right*, so driving ``left`` from the pointer
    makes a card near the right edge wrap its text and change width as the
    mouse moves. Transforms are applied after layout, so moving the card there
    leaves its width alone.

    Offsetting by ``-100%`` resolves against the card's own width, which is how
    a region near the right or bottom edge opens back towards the middle
    without anyone measuring anything.
    """
    width, height = size
    pad = 16
    horizontal = (
        f"calc(var(--vl-x,0px) - 100% - {pad}px)"
        if rect.left + rect.width / 2 > width * 0.6
        else f"calc(var(--vl-x,0px) + {pad}px)"
    )
    vertical = (
        f"calc(var(--vl-y,0px) - 100% - {pad}px)"
        if rect.top + rect.height / 2 > height * 0.6
        else f"calc(var(--vl-y,0px) + {pad}px)"
    )
    return f"transform:translate({horizontal},{vertical})"


def _hover_regions(
    hits: HitMap, size: tuple[int, int]
) -> tuple[str, list[str], str]:
    """Build the SVG hover layer, the tooltip cards, and the rules linking them.

    Cards are per *region* rather than per distinct content: each one is
    positioned at the region it belongs to, so two annotations with identical
    tooltips still need their own card in their own place.

    Args:
        hits: The scene's hover regions, in render pixels.
        size: The scene's ``(width, height)`` in render pixels.

    Returns:
        The ``<g>`` of hover rectangles, the positioned card elements, and the
        CSS that reveals each card while its region is hovered.

    """
    cards: list[str] = []
    rules: list[str] = []
    rects = []
    for rect, tooltip in _hit_order(hits.items):
        box = (
            f'x="{_num(rect.left)}" y="{_num(rect.top)}" '
            f'width="{_num(rect.width)}" height="{_num(rect.height)}"'
        )
        if tooltip.is_empty:
            # No card, but the region still swallows the hover, so an empty
            # tooltip nested in a larger one blanks it — as on the desktop,
            # where `HitMap.hit` wins it and `draw_tooltip` draws nothing.
            rects.append(f"<rect {box}/>")
            continue
        index = len(cards)
        rects.append(f'<rect {box} data-tip="{index}"/>')
        cards.append(
            f'<div class="vl-tip" data-tip="{index}" '
            f'style="{_card_position(rect, size)}">{_tooltip_html(tooltip)}'
            "</div>"
        )
        rules.append(
            f'.vl-figure:has([data-tip="{index}"]:hover) '
            f'.vl-tip[data-tip="{index}"]{{display:block}}'
        )
    layer = "".join(rects)
    return (
        f'<g class="vl-hit" fill="none" pointer-events="all">{layer}</g>',
        cards,
        "".join(rules),
    )


def split_svg(svg: bytes) -> tuple[str, str]:
    """Split an SVG document into its opening tag and its body.

    A pass that paints nothing at all closes the root on itself, so there is no
    body and no ``</svg>`` to find. That is the ordinary outcome for a layer the
    scene turns out not to contain, not an error.

    Args:
        svg: The SVG document bytes from `Canvas.finish_svg`.

    Returns:
        The opening tag up to but excluding its ``>``, and the body between the
        tags (empty when the document draws nothing).

    """
    text = svg.decode("utf-8")
    start = text.index("<svg")
    head = text.index(">", start)
    root = text[start:head].rstrip("/").rstrip()
    end = text.rfind("</svg>")
    return root, "" if end < 0 else text[head + 1 : end]


def _responsive_svg(svg: bytes, size: tuple[int, int], layer: str) -> str:
    """Re-root the SVG on a ``viewBox`` and append the hover layer to it.

    Trading the fixed ``width``/``height`` viewport for a ``viewBox`` lets the
    browser fit the drawing to whatever width the page gives it; because the
    hover rectangles are appended inside the same document, they are laid out in
    the drawing's coordinates and rescale with it for free. Appending them last
    also puts them above every mark, so nothing painted can shadow a region.

    Args:
        svg: The SVG document bytes from `Canvas.finish_svg`.
        size: The scene's ``(width, height)`` in render pixels.
        layer: The hover-layer element to append.

    Returns:
        The root ``<svg>`` element, without the XML declaration that would be
        out of place inside an HTML document.

    """
    root, body = split_svg(svg)
    return (
        f'{_SIZE_ATTRS.sub("", root)} viewBox="0 0 {size[0]} {size[1]}">'
        f"{body}{layer}</svg>"
    )


def html_document(
    svg: bytes,
    hits: HitMap,
    size: tuple[int, int],
    theme: Theme,
    *,
    nav: "Nav | None" = None,
    title: str = "vizlab",
) -> str:
    """Assemble a vector render and its hover map into one HTML page.

    The seam behind `luxonis_ml.vizlab.Renderable.render_html`, which is where
    the two inputs are produced (in a single draw pass, so they cannot drift
    apart). Everything the page needs is inlined: it loads no stylesheet, font,
    script, or image from anywhere.

    Args:
        svg: The vector render, as `Canvas.finish_svg` bytes.
        hits: The hover regions, in the same render pixels the SVG uses.
        size: The scene's ``(width, height)`` in render pixels.
        theme: The theme the scene rendered with; the page background and card
            chrome are derived from it.
        nav: Links to this page's neighbours in a series, if it is part of one.
        title: The page's ``<title>``, as plain text.

    Returns:
        The HTML document.

    """
    layer, cards, rules = _hover_regions(hits, size)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{escape_html(title)}</title>\n"
        f"<style>{_page_variables(theme, size)}{_STYLE}"
        f"{_NAV_STYLE}{rules}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{_nav_html(nav)}"
        f'<div class="vl-figure">{_responsive_svg(svg, size, layer)}'
        f"{''.join(cards)}</div>\n"
        f"<script>{_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )


#: The layer a given annotation belongs to, as ``(slug, label)``. Order is the
#: order the controls appear in. Mirrors `luxonis_ml.vizlab.viewer.LayerState`
#: so a saved page and the desktop viewer offer the same vocabulary.
LAYERS: tuple[tuple[str, str], ...] = (
    ("box", "boxes"),
    ("mask", "masks"),
    ("keypoint", "keypoints"),
    ("field", "arrays"),
    ("overlay", "overlays"),
    ("label", "labels"),
)

_ID_ATTR = re.compile(r'id="([^"]+)"')
_ID_REF = re.compile(r'(href="#|url\(#)([^")]+)')


def slug(value: str) -> str:
    """Reduce a class name to something usable in a CSS class."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "unnamed"


def class_slugs(names: "Iterable[str]") -> dict[str, str]:
    """Map each class name to a unique slug for its checkbox id and CSS class.

    `slug` alone can fold two distinct names into one key ("Car" and "car",
    say), which would emit two inputs with the same ``id`` — the browser then
    binds both labels to the first one and the second control goes dead. A
    collision gets a numeric suffix instead, in input order.

    Examples:
        >>> class_slugs(["Car", "car"])
        {'Car': 'car', 'car': 'car-2'}

    """
    keys: dict[str, str] = {}
    taken: set[str] = set()
    for name in names:
        base = slug(name)
        candidate, counter = base, 1
        while candidate in taken:
            counter += 1
            candidate = f"{base}-{counter}"
        keys[name] = candidate
        taken.add(candidate)
    return keys


def _scope_ids(body: str, prefix: str) -> str:
    """Namespace a fragment's ``id``s and the references pointing at them.

    Every fragment is emitted by a fresh skia canvas that restarts its counter,
    so ``cl_3`` means something different in each one. SVG ids are document
    global, so without this the second fragment's ``url(#cl_3)`` silently
    resolves to the first fragment's clip.
    """
    body = _ID_ATTR.sub(lambda m: f'id="{prefix}{m.group(1)}"', body)
    return _ID_REF.sub(lambda m: f"{m.group(1)}{prefix}{m.group(2)}", body)


def _fragment_body(svg: bytes, prefix: str) -> str:
    """Strip a fragment down to the marks it draws, with its ids namespaced.

    Args:
        svg: The fragment's SVG document bytes.
        prefix: Namespace for this fragment's ids.

    Returns:
        The fragment's body, with its ids namespaced.

    """
    return _scope_ids(split_svg(svg)[1], prefix)


def layered_svg(
    base: bytes,
    fragments: "Sequence[tuple[str, bytes]]",
    size: tuple[int, int],
    hover: str,
) -> str:
    """Compose a base render and its per-layer fragments into one SVG.

    Args:
        base: The vector render of the scene with no annotations.
        fragments: ``(css_classes, svg)`` pairs, in draw order.
        size: The scene's ``(width, height)`` in render pixels.
        hover: The hover-layer element, appended last.

    Returns:
        The root ``<svg>`` element, ready to inline in a page.

    """
    root, body = split_svg(base)
    root = _SIZE_ATTRS.sub("", root)
    picture = _scope_ids(body, "b_")
    groups = "".join(
        f'<g class="{classes}">{_fragment_body(svg, f"f{index}_")}</g>'
        for index, (classes, svg) in enumerate(fragments)
    )
    return (
        f'{root} viewBox="0 0 {size[0]} {size[1]}">'
        f"{picture}{groups}{hover}</svg>"
    )


_CONTROL_STYLE = """
.vl-toggle{position:absolute;opacity:0;pointer-events:none}
.vl-bar{display:flex;flex-wrap:wrap;gap:.4em;align-items:center;
 margin:0 auto .9em;max-width:var(--vl-width);padding:0 .1em;
 font:500 .82em/1 var(--vl-sans)}
.vl-bar .vl-group{display:flex;flex-wrap:wrap;gap:.4em;align-items:center}
.vl-bar .vl-sep{flex:1 1 auto}
.vl-bar label{cursor:pointer;user-select:none;border-radius:.5em;
 padding:.45em .7em;border:1px solid var(--vl-edge);color:var(--vl-dim);
 background:transparent;transition:background .12s,color .12s,border-color .12s}
.vl-bar label:hover{border-color:var(--vl-key)}
.vl-bar .vl-swatch{width:.62em;height:.62em;border-radius:50%;
 display:inline-block;margin-right:.45em;vertical-align:baseline;
 background:currentColor}
.vl-bar .vl-bulk{cursor:pointer;font:inherit;border-radius:.5em;padding:.45em .7em;border:1px dashed var(--vl-edge);color:var(--vl-dim);background:transparent}
.vl-bar .vl-bulk:hover{border-color:var(--vl-key);color:var(--vl-text)}
.vl-hint{color:var(--vl-dim);opacity:.75;font:400 .78em/1 var(--vl-sans)}
"""


def _control_css(
    kinds: "Sequence[str]",
    classes: "Sequence[str]",
    slugs: "Mapping[str, str]",
) -> str:
    """Build the rules that make the controls actually hide things.

    Every rule is a sibling-combinator match on a hidden ``<input>``, which is
    what lets the page do real work with no script: a checkbox *is* the state,
    and CSS reads it. The inputs therefore have to precede the figure.
    """
    rules = [_CONTROL_STYLE]
    for kind in kinds:
        rules.append(
            f"#vl-l-{kind}:checked~.vl-bar label[for=vl-l-{kind}]"
            "{background:var(--vl-key);color:var(--vl-bg);"
            "border-color:var(--vl-key)}"
            f"#vl-l-{kind}:not(:checked)~.vl-figure .vl-{kind}"
            "{display:none}"
        )
    for name in classes:
        key = slugs[name]
        rules.append(
            f"#vl-c-{key}:checked~.vl-bar label[for=vl-c-{key}]"
            "{color:var(--vl-text);border-color:currentColor}"
            f"#vl-c-{key}:not(:checked)~.vl-figure .vl-cls-{key}"
            "{display:none}"
        )
    return "".join(rules)


def _control_html(
    kinds: "Sequence[str]",
    classes: "Sequence[str]",
    swatches: dict[str, str],
    slugs: "Mapping[str, str]",
) -> str:
    """Build the hidden inputs and the visible bar of labels."""
    labels = dict(LAYERS)
    inputs = [
        f'<input type="checkbox" class="vl-toggle" id="vl-l-{kind}" checked>'
        for kind in kinds
    ]
    chips = [
        f'<label for="vl-l-{kind}">{escape_html(labels[kind])}</label>'
        for kind in kinds
    ]

    class_chips: list[str] = []
    for name in classes:
        key = slugs[name]
        tint = swatches.get(name, "currentColor")
        inputs.append(
            f'<input type="checkbox" class="vl-toggle vl-class" '
            f'id="vl-c-{key}" checked>'
        )
        swatch = f'<i class="vl-swatch" style="color:{tint}"></i>'
        class_chips.append(
            f'<label for="vl-c-{key}">{swatch}{escape_html(name)}</label>'
        )

    groups = [f'<span class="vl-group">{"".join(chips)}</span>']
    if class_chips:
        groups.append('<span class="vl-sep"></span>')
        # Selecting every class at once is the one control CSS cannot express:
        # a selector can read a checkbox but never set one.
        groups.append(
            '<span class="vl-group">'
            '<button type="button" class="vl-bulk" data-check="1">all</button>'
            '<button type="button" class="vl-bulk" data-check="">none</button>'
            "</span>"
        )
        groups.append(f'<span class="vl-group">{"".join(class_chips)}</span>')
    bar = f'<div class="vl-bar">{"".join(groups)}</div>'
    return "".join(inputs) + bar


def layered_document(
    base: bytes,
    fragments: "Sequence[tuple[str, bytes]]",
    hits: HitMap,
    size: tuple[int, int],
    theme: Theme,
    *,
    kinds: "Sequence[str]",
    classes: "Sequence[str]",
    swatches: "dict[str, str] | None" = None,
    slugs: "Mapping[str, str] | None" = None,
    nav: "Nav | None" = None,
    title: str = "vizlab",
) -> str:
    """Assemble a layered render into one page with working controls.

    The richer counterpart of `html_document`: the scene arrives already split
    into per-layer fragments, each tagged with the CSS classes that identify
    its annotation kind and class name. One checkbox per layer and per class,
    sitting in front of the figure, then hides them through sibling selectors
    alone — the page carries no script for any of it.

    Args:
        base: The vector render of the scene with no annotations.
        fragments: ``(css_classes, svg)`` pairs, in draw order.
        hits: The hover regions, in the same render pixels.
        size: The scene's ``(width, height)`` in render pixels.
        theme: The theme the scene rendered with.
        kinds: Layer slugs present, in `LAYERS` order.
        classes: Class names present, in legend order.
        swatches: Class name to CSS colour, for the control chips.
        slugs: Class name to unique slug (see `class_slugs`), matching the
            ``vl-cls-…`` classes on ``fragments``; derived from ``classes``
            when omitted.
        nav: Links to this page's neighbours in a series, if it is part of one.
        title: The page's ``<title>``, as plain text.

    Returns:
        The HTML document.

    """
    hover, cards, rules = _hover_regions(hits, size)
    keys = slugs if slugs is not None else class_slugs(classes)
    controls = _control_html(kinds, classes, swatches or {}, keys)
    figure = layered_svg(base, fragments, size, hover)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{escape_html(title)}</title>\n"
        f"<style>{_page_variables(theme, size)}{_STYLE}"
        f"{_control_css(kinds, classes, keys)}{_NAV_STYLE}{rules}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{_nav_html(nav)}"
        f"{controls}"
        f'<div class="vl-figure">{figure}{"".join(cards)}</div>\n'
        f"<script>{_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )


_INDEX_STYLE = """
*,::before,::after{box-sizing:border-box}
html,body{margin:0}
body{background:var(--vl-bg);color:var(--vl-text);padding:2.5rem 1.5rem 5rem;
 font-family:var(--vl-sans)}
main{max-width:52rem;margin:0 auto}
h1{font-size:1.5rem;letter-spacing:-.02em;margin:0 0 .2rem}
p.count{color:var(--vl-dim);margin:0 0 1.8rem;font-size:.9rem}
ol{list-style:none;margin:0;padding:0;display:grid;gap:.4rem}
a{display:flex;gap:.8rem;align-items:baseline;text-decoration:none;
 color:var(--vl-text);border:1px solid var(--vl-edge);border-radius:.5rem;
 padding:.7rem .9rem;transition:border-color .12s,background .12s}
a:hover{border-color:var(--vl-key);background:color-mix(in srgb,var(--vl-key) 8%,transparent)}
a .n{color:var(--vl-dim);font-family:var(--vl-mono);font-size:.82rem;min-width:2.6rem}
"""


def index_document(
    entries: "Sequence[tuple[str, str]]",
    theme: Theme,
    *,
    title: str = "vizlab",
) -> str:
    """Build an entry-point page linking a directory of rendered pages.

    A directory of ``0000_frame01.html`` files is not something anyone wants to
    open one at a time; this is the file you actually send someone.

    Args:
        entries: ``(href, label)`` pairs, in the order to list them.
        theme: The theme the renders used, so the index matches them.
        title: The page's ``<title>`` and heading, as plain text.

    Returns:
        The HTML document.

    """
    items = "".join(
        f'<li><a href="{escape_html(href)}">'
        f'<span class="n">{index:04d}</span>'
        f"<span>{escape_html(label)}</span></a></li>"
        for index, (href, label) in enumerate(entries)
    )
    count = len(entries)
    plural = "" if count == 1 else "s"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{escape_html(title)}</title>\n"
        f"<style>{_page_variables(theme, (960, 540))}{_INDEX_STYLE}</style>\n"
        "</head>\n"
        "<body>\n<main>\n"
        f"<h1>{escape_html(title)}</h1>\n"
        f'<p class="count">{count} render{plural}</p>\n'
        f"<ol>{items}</ol>\n"
        "</main>\n</body>\n</html>\n"
    )


class Nav(NamedTuple):
    """Where a page sits in a series, so it can link to its neighbours.

    Attributes:
        index: Href of the entry-point page (labelled "home"), or ``None``
            for a lone render.
        previous: Href of the preceding page, or ``None`` on the first.
        following: Href of the next page, or ``None`` on the last.
        position: This page's 1-based position, or ``None`` to omit the count.
        total: How many pages there are, or ``None`` if unknown.

    """

    index: "str | None" = None
    previous: "str | None" = None
    following: "str | None" = None
    position: "int | None" = None
    total: "int | None" = None


_NAV_STYLE = """
.vl-nav{display:flex;gap:.4em;align-items:center;margin:0 auto .7em;
 max-width:var(--vl-width);font:500 .82em/1 var(--vl-sans)}
.vl-nav a,.vl-nav span.step{border-radius:.5em;padding:.45em .75em;
 border:1px solid var(--vl-edge);text-decoration:none;color:var(--vl-dim)}
.vl-nav a:hover{border-color:var(--vl-key);color:var(--vl-text)}
.vl-nav span.step{opacity:.35}
.vl-nav .grow{flex:1 1 auto}
.vl-nav .pos{color:var(--vl-dim);font-family:var(--vl-mono);font-size:.92em}
"""


def _nav_html(nav: "Nav | None") -> str:
    """Build the previous/index/next bar, greying out the ends of the series."""
    if nav is None:
        return ""

    def step(href: "str | None", label: str, rel: str) -> str:
        if href is None:
            # An end of the series is shown, not hidden, so the control does
            # not jump sideways as you page through.
            return f'<span class="step">{label}</span>'
        return f'<a href="{escape_html(href)}" rel="{rel}">{label}</a>'

    parts = [step(nav.previous, "&larr; prev", "prev")]
    if nav.index is not None:
        parts.append(f'<a href="{escape_html(nav.index)}">home</a>')
    parts.append(step(nav.following, "next &rarr;", "next"))
    parts.append('<span class="grow"></span>')
    if nav.position is not None:
        # The total is unknown while a stream is still being written, so the
        # position stands on its own rather than the counter disappearing.
        count = (
            f"{nav.position} / {nav.total}"
            if nav.total is not None
            else f"#{nav.position}"
        )
        parts.append(f'<span class="pos">{count}</span>')
    return f'<nav class="vl-nav">{"".join(parts)}</nav>'


def draws_anything(svg: bytes) -> bool:
    """Return whether a fragment paints anything at all.

    A layer can legitimately come out empty — asking for the label chips of an
    unlabeled field, say. Emitting it anyway would put a control in the bar
    that switches nothing.

    Args:
        svg: The fragment's SVG document bytes.

    Returns:
        Whether it contains any mark.

    """
    return bool(_MARKS.search(split_svg(svg)[1]))


#: A blit counts: a fill-only layer draws nothing but its own ``<use>``.
_MARKS = re.compile(r"<(?:path|circle|ellipse|line|polygon|text|image|use)\b")
