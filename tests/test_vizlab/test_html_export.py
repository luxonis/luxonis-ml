"""Tests for `Renderable.render_html` — the self-contained interactive page."""

import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.utils.color import brand
from luxonis_ml.vizlab import (
    LIGHT_THEME,
    BBox,
    Color,
    Image,
    Keypoints,
    RenderOptions,
    ScalarField,
    Tooltip,
    escape,
    grid,
    with_panel,
)
from luxonis_ml.vizlab.geometry import Rect

_SVG_NS = "{http://www.w3.org/2000/svg}"
_XLINK = "{http://www.w3.org/1999/xlink}"
_TIP_RE = re.compile(
    r'<div class="vl-tip" data-tip="(\d+)"[^>]*>(.*?)</div>\s*'
    r'(?=<div class="vl-tip"|</div>)',
    re.DOTALL,
)


def _blank(h: int = 60, w: int = 100) -> Image:
    return Image(np.zeros((h, w, 3), np.uint8))


def _tile(title: str = "car", options: RenderOptions | None = None) -> Image:
    """Return a 100x60 image with one box at ``(20, 12)``, 40x30."""
    return Image(np.zeros((60, 100, 3), np.uint8), options=options).add(
        BBox(x=0.2, y=0.2, w=0.4, h=0.5, tooltip=Tooltip(title=title))
    )


def _svg_root(page: str) -> ET.Element:
    """Parse the page's inline SVG, which must be well-formed on its own."""
    end = page.index("</svg>") + len("</svg>")
    # The page is one we just built, not untrusted input.
    return ET.fromstring(page[page.index("<svg") : end])  # noqa: S314


def _layer_alpha(page: str, css: str, width: int, height: int) -> np.ndarray:
    """Composite a layer's rasters into a scene-sized opacity map.

    A fill layer's raster is cropped to what it paints and positioned by the
    ``<use>`` that blits it, so reading its coverage means honouring that
    placement rather than assuming the image covers the frame.
    """
    import base64
    import io

    from PIL import Image as PILImage

    layer = next(
        g for g in _svg_root(page).iter(f"{_SVG_NS}g") if g.get("class") == css
    )
    rasters = {
        # An embedded raster that is opaque throughout is written without an
        # alpha channel, so it is normalized here rather than assumed.
        f"#{image.get('id')}": np.array(
            PILImage.open(
                io.BytesIO(
                    base64.b64decode(
                        image.get(f"{_XLINK}href", "").partition("base64,")[2]
                    )
                )
            ).convert("RGBA")
        )
        for image in layer.iter(f"{_SVG_NS}image")
    }
    alpha = np.zeros((height, width), dtype=np.uint8)
    for use in layer.iter(f"{_SVG_NS}use"):
        raster = rasters[use.get(f"{_XLINK}href", "")]
        transform = use.get("transform", "")
        assert transform.startswith("translate(") or not transform, transform
        offsets = [int(float(n)) for n in re.findall(r"-?[\d.]+", transform)]
        left, top = offsets or [0, 0]
        rows, cols = raster.shape[:2]
        window = alpha[top : top + rows, left : left + cols]
        window[:] = np.maximum(window, raster[..., 3])
    return alpha


def _hit_rects(page: str) -> list[ET.Element]:
    """Return a page's hover rectangles, in document order."""
    layer = _svg_root(page).find(f'{_SVG_NS}g[@class="vl-hit"]')
    assert layer is not None
    return list(layer)


def _box(rect: ET.Element) -> tuple[float, float, float, float]:
    """Return a hover rectangle's ``(x, y, width, height)``."""
    return (
        float(rect.get("x", 0)),
        float(rect.get("y", 0)),
        float(rect.get("width", 0)),
        float(rect.get("height", 0)),
    )


def _placed(rect: Rect) -> tuple[float, float, float, float]:
    """Return a `Rect` the way the page writes it, rounded to 2 decimals."""
    return (
        round(rect.left, 2),
        round(rect.top, 2),
        round(rect.width, 2),
        round(rect.height, 2),
    )


def _cards(page: str) -> list[str]:
    """Return each tooltip card's markup, in ``data-tip`` index order.

    Cards live inside the figure now, each positioned at its own region, so
    they are read by their index rather than sliced out of a trailing block.
    """
    found = {int(index): body for index, body in _TIP_RE.findall(page)}
    return [found[key] for key in sorted(found)]


def _card_of(page: str, rect: ET.Element) -> str:
    """Return the card a hover rectangle points at."""
    return _cards(page)[int(rect.get("data-tip", ""))]


def _hex(color: Color) -> str:
    return f"#{color.r:02x}{color.g:02x}{color.b:02x}"


class _Structure(HTMLParser):
    """Collects tag nesting and script bodies, to check a page holds together."""

    def __init__(self) -> None:
        super().__init__()
        self.open: list[str] = []
        self.mismatched: list[str] = []
        self.scripts: list[str] = []
        self._void = {"meta", "br", "img", "input", "link"}

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        """Ignore XML self-closing tags (the SVG marks) — they never nest."""

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag not in self._void:
            self.open.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.open and self.open[-1] == tag:
            self.open.pop()
        else:
            self.mismatched.append(tag)

    def handle_data(self, data: str) -> None:
        if self.open and self.open[-1] == "script":
            self.scripts.append(data)


def _structure(page: str) -> _Structure:
    parser = _Structure()
    parser.feed(page)
    parser.close()
    return parser


def test_page_is_a_balanced_standalone_document() -> None:
    page = _tile().render_html(title='a & "b" <c>')

    assert page.startswith("<!DOCTYPE html>\n")
    assert page.rstrip().endswith("</html>")
    parsed = _structure(page)
    assert parsed.open == []
    assert parsed.mismatched == []
    assert "<title>a &amp; &quot;b&quot; &lt;c&gt;</title>" in page


def test_every_tooltip_region_lands_where_the_hit_map_put_it() -> None:
    image = _tile()
    image.add(BBox(x=0.7, y=0.1, w=0.2, h=0.2))  # no tooltip -> no region
    image.add(Keypoints(keypoints=[(0.3, 0.9, 2)], tooltip=Tooltip(title="p")))

    _, hits = image.render_hits()
    rects = _hit_rects(image.render_html())

    assert len(rects) == len(hits.items) == 2
    assert {_box(rect) for rect in rects} == {
        _placed(rect) for rect, _ in hits.items
    }


def test_smaller_nested_region_wins_the_hover() -> None:
    outer, inner = Tooltip(title="car"), Tooltip(title="plate")
    image = _blank(100, 100)
    # The small box is added *first*, so plain z-order would hand the hover to
    # the big one — the mistake the page must not make.
    image.add(BBox(x=0.4, y=0.4, w=0.2, h=0.2, tooltip=inner))
    image.add(BBox(x=0.0, y=0.0, w=1.0, h=1.0, tooltip=outer))
    # The semantics the page has to reproduce without a hit test of its own.
    assert image.render_hits()[1].hit(50, 50) is inner

    page = image.render_html()
    rects = _hit_rects(page)

    # A pointer lands on the last matching element, so the regions run
    # largest-first and the small box ends up on top.
    areas = [width * height for _, _, width, height in map(_box, rects)]
    assert areas == sorted(areas, reverse=True)
    assert "plate" in _card_of(page, rects[-1])
    assert "car" in _card_of(page, rects[0])


def test_empty_tooltip_takes_the_hover_without_showing_a_card() -> None:
    image = _blank(100, 100)
    image.add(BBox(x=0.0, y=0.0, w=1.0, h=1.0, tooltip=Tooltip(title="car")))
    image.add(BBox(x=0.4, y=0.4, w=0.2, h=0.2, tooltip=Tooltip()))

    page = image.render_html()
    rects = _hit_rects(page)

    assert len(rects) == 2
    assert rects[-1].get("data-tip") is None  # smallest, but nothing to show
    assert len(_cards(page)) == 1


def test_identical_tooltips_get_their_own_card_at_their_own_region() -> None:
    """Cards are per region, not per distinct content.

    Placement is what forces this: with no script, a card is positioned by CSS
    at the coordinates of the region it belongs to, so two annotations with
    byte-identical tooltips still need one card each, in two different places.
    """
    tip = Tooltip(title="car", rows=(("id", "7"),))
    image = _blank(100, 100)
    image.add(BBox(x=0.0, y=0.0, w=0.4, h=0.4, tooltip=tip))
    image.add(BBox(x=0.5, y=0.5, w=0.4, h=0.4, tooltip=Tooltip(**vars(tip))))

    page = image.render_html()

    assert len(_hit_rects(page)) == 2
    cards = _cards(page)
    assert len(cards) == 2
    assert cards[0] == cards[1]  # same content...
    placements = re.findall(
        r'<div class="vl-tip" data-tip="\d+" style="([^"]*)"', page
    )
    assert len(set(placements)) == 2  # ...in different places


def test_the_script_only_feeds_the_stylesheet() -> None:
    """The script may supply inputs to the CSS; it may not decide any output.

    Two things are genuinely beyond a selector: reading where the pointer is,
    and *writing* a checkbox (CSS can read ``:checked`` but never set it). Both
    are inputs. The moment the script starts showing, hiding, measuring or
    restyling, the stylesheet has stopped being the source of truth.
    """
    image = _blank().add(
        BBox(x=0.1, y=0.1, w=0.3, h=0.3, tooltip=Tooltip(title="car"))
    )
    page = image.render_html()
    body = _structure(page).scripts
    assert len(body) == 1
    script = body[0]
    assert "--vl-x" in script
    assert "--vl-y" in script
    assert ".checked" in script
    for forbidden in ("display", "classList", "innerHTML", "offsetWidth"):
        assert forbidden not in script
    # Hover itself is still resolved declaratively.
    assert ':has([data-tip="0"]:hover)' in page


def test_classes_can_be_selected_and_deselected_in_bulk() -> None:
    """Every class checkbox is reachable by both buttons."""
    page = _layered_scene().render_html()
    buttons = re.findall(
        r'<button type="button" class="vl-bulk" data-check="([^"]*)">([^<]*)</button>',
        page,
    )
    assert buttons == [("1", "all"), ("", "none")]
    # The buttons address the class boxes only, never the layer boxes — so the
    # script has to select the narrow class, not the one every toggle shares.
    assert page.count('class="vl-toggle vl-class"') == 2
    for kind in ("box", "keypoint"):
        assert f'class="vl-toggle" id="vl-l-{kind}"' in page
    script = _structure(page).scripts[0]
    assert '".vl-class"' in script
    assert '".vl-toggle"' not in script


def test_a_card_is_pinned_to_the_cursor() -> None:
    """The card tracks the pointer, entirely through its transform."""
    image = _blank().add(
        BBox(x=0.1, y=0.1, w=0.3, h=0.3, tooltip=Tooltip(title="car"))
    )
    page = image.render_html()
    style = re.search(
        r'<div class="vl-tip" data-tip="0" style="([^"]*)"', page
    )
    assert style is not None
    # Near the top-left, the card opens down and to the right of the cursor.
    assert style.group(1) == (
        "transform:translate(calc(var(--vl-x,0px) + 16px),"
        "calc(var(--vl-y,0px) + 16px))"
    )


def test_a_cards_width_does_not_depend_on_where_the_cursor_is() -> None:
    """The layout box stays at the origin; only the transform moves.

    An absolutely positioned box with ``left`` set gets a shrink-to-fit width
    bounded by the space remaining to its right. Driving ``left`` from the
    pointer therefore squeezed any card near the right edge, which re-wrapped
    its text and changed width as the mouse moved — visible only on the
    right-most annotation, where that remaining space is small.
    """
    page = (
        _blank()
        .add(BBox(x=0.7, y=0.1, w=0.25, h=0.3, tooltip=Tooltip(title="truck")))
        .render_html()
    )

    rule = re.search(r"\.vl-tip\{([^}]*)\}", page)
    assert rule is not None
    assert "left:0" in rule.group(1)
    assert "top:0" in rule.group(1)
    # The pointer may only ever reach the card through its transform.
    for style in re.findall(
        r'<div class="vl-tip" data-tip="\d+" style="([^"]*)"', page
    ):
        assert style.startswith("transform:")
        assert "--vl-x" not in style.split("transform:")[0]


def test_a_card_near_an_edge_opens_back_towards_the_middle() -> None:
    """``-100%`` resolves against the card's own width, so nothing overflows.

    That is the clamp the old script did at runtime, expressed declaratively:
    a region past the far edge opens on the other side of the cursor instead.
    """
    image = _blank(100, 100)
    image.add(
        BBox(x=0.02, y=0.02, w=0.2, h=0.2, tooltip=Tooltip(title="near"))
    )
    image.add(BBox(x=0.75, y=0.78, w=0.2, h=0.2, tooltip=Tooltip(title="far")))

    joined = " ".join(
        re.findall(
            r'<div class="vl-tip" data-tip="\d+" style="([^"]*)"',
            image.render_html(),
        )
    )
    assert "calc(var(--vl-x,0px) + 16px)" in joined  # the near corner
    assert "calc(var(--vl-x,0px) - 100% - 16px)" in joined  # the far one
    assert "calc(var(--vl-y,0px) - 100% - 16px)" in joined


def test_dataset_text_is_escaped_and_cannot_break_out() -> None:
    hostile = '</span><script>alert("x")</script> a & b'
    image = _blank().add(
        BBox(
            x=0.2,
            y=0.2,
            w=0.4,
            h=0.5,
            tooltip=Tooltip(title="a & b", rows=(("path", hostile),)),
        )
    )

    page = image.render_html()

    assert hostile not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in page
    assert "a &amp; b" in _cards(page)[0]
    # Nothing hostile reached a real script element.
    assert not any("alert" in body for body in _structure(page).scripts)


def test_markup_tags_become_styled_html() -> None:
    tip = Tooltip(
        title="<i>car</i>",
        rows=(("id", '<b>7</b> <span color="#ff0000">!</span>'),),
    )
    image = _blank().add(BBox(x=0.2, y=0.2, w=0.4, h=0.5, tooltip=tip))

    card = _cards(image.render_html())[0]

    assert "font-style:italic" in card
    assert "font-weight:700" in card
    assert "color:#ff0000" in card
    assert "<i>" not in card
    assert "&lt;i&gt;" not in card


def test_escaped_markup_stays_literal_text() -> None:
    tip = Tooltip(rows=(("class", escape("<b>bold</b>")),))
    image = _blank().add(BBox(x=0.2, y=0.2, w=0.4, h=0.5, tooltip=tip))

    card = _cards(image.render_html())[0]

    assert "&lt;b&gt;bold&lt;/b&gt;" in card
    assert "font-weight:700" not in card


def test_tooltip_tint_colors_the_swatch_and_title() -> None:
    tip = Tooltip(title="car", tint=Color(255, 0, 0))
    image = _blank().add(BBox(x=0.2, y=0.2, w=0.4, h=0.5, tooltip=tip))

    card = _cards(image.render_html())[0]

    assert 'class="vl-swatch" style="background:#ff0000"' in card
    assert 'class="vl-title" style="color:#ff0000"' in card


def test_grid_regions_are_offset_by_their_placement() -> None:
    composite = grid([_tile("A"), _tile("B")], ncols=2, pad=10)

    _, hits = composite.render_hits()
    page = composite.render_html()

    boxes = {_box(rect) for rect in _hit_rects(page)}
    assert boxes == {_placed(rect) for rect, _ in hits.items}
    # Same box in both tiles, so only the x offset tells the cells apart: the
    # left cell starts at the pad, the right one a whole cell further along.
    assert sorted(x for x, _, _, _ in boxes) == [10 + 20.0, 120 + 20.0]
    assert _svg_root(page).get("viewBox") == "0 0 230 80"


def test_paneled_composite_keeps_regions_over_the_image() -> None:
    image = _tile()
    composite = with_panel(image, {"split": "train"})

    _, hits = composite.render_hits()
    rects = _hit_rects(composite.render_html())

    assert len(rects) == 1
    x, y, width, height = _box(rects[0])
    assert (x, y, width, height) == _placed(hits.items[0][0])
    # The image is inset by the panel's frame, so the box moved with it...
    assert (x, y) > (20.0, 12.0)
    # ...at its own size, and still within the image half of the composite.
    assert (width, height) == (40.0, 30.0)
    assert x + width <= image.width
    assert composite.width > image.width  # the panel really was appended


def test_drawing_and_regions_share_one_scalable_coordinate_system() -> None:
    page = _tile().render_html((200, 120))
    root = _svg_root(page)

    # A viewBox instead of a fixed viewport: the browser fits the drawing to the
    # page width, and the hover rects (same coordinates) follow it for free.
    assert root.get("viewBox") == "0 0 200 120"
    assert root.get("width") is None
    assert root.get("height") is None
    assert "--vl-width:200px" in page
    # The region doubled with the drawing, in the same coordinates.
    assert _box(_hit_rects(page)[0]) == (40.0, 24.0, 80.0, 60.0)


def test_page_chrome_follows_the_scene_theme() -> None:
    dark = _tile().render_html()
    light = _tile(options=RenderOptions(theme=LIGHT_THEME)).render_html()

    assert f"--vl-bg:{_hex(brand.BACKGROUND)}" in dark
    assert f"--vl-card-title:{_hex(brand.CARD_TITLE)}" in dark
    assert f"--vl-bg:{_hex(LIGHT_THEME.background)}" in light
    assert f"--vl-card-title:{_hex(brand.LIGHT_CARD_TITLE)}" in light


def test_page_references_nothing_off_document() -> None:
    page = _tile().render_html()

    # The only absolute URLs are the SVG/XLink namespace names, which name a
    # vocabulary rather than a resource anything fetches.
    without_namespaces = re.sub(r'xmlns(:\w+)?="[^"]*"', "", page)
    assert "http://" not in without_namespaces
    assert "https://" not in without_namespaces
    assert "src=" not in page
    assert "@import" not in page
    # Every reference is either inline data or a link within the document.
    references = re.findall(r'(?:xlink:)?href="([^"]*)"', page)
    assert references
    assert all(url.startswith(("data:", "#")) for url in references)


def test_save_writes_a_page_for_html_and_htm(tmp_path: Path) -> None:
    page = tmp_path / "out.html"
    _tile().save(page)
    assert page.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    short = tmp_path / "out.htm"
    _tile().save(short)
    assert "<svg" in short.read_text(encoding="utf-8")


def test_save_still_rejects_an_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported output format"):
        _tile().save(tmp_path / "out.tiff")


def _layered_scene() -> "Image":
    """Build a scene with two kinds and two classes, so buckets split."""
    image = Image(np.zeros((200, 320, 3), dtype=np.uint8))
    image.add(BBox(x=0.1, y=0.1, w=0.3, h=0.4).tag("car", score=0.9))
    image.add(BBox(x=0.55, y=0.2, w=0.2, h=0.5).tag("person"))
    image.add(
        Keypoints(keypoints=[(0.6, 0.3, 2), (0.62, 0.5, 2)]).tag("person")
    )
    return image


def _svg_of(page: str) -> str:
    return page[page.index("<svg") : page.index("</svg>") + 6]


def test_layers_are_tagged_by_kind_and_class() -> None:
    """Shapes split by (layer, class); chips are a layer of their own."""
    groups = re.findall(
        r'<g class="(vl-layer[^"]*)"', _layered_scene().render_html()
    )
    assert groups == [
        "vl-layer vl-box vl-cls-car",
        "vl-layer vl-box vl-cls-person",
        "vl-layer vl-keypoint vl-cls-person",
        "vl-layer vl-label vl-cls-car",
        "vl-layer vl-label vl-cls-person",
    ]


def test_a_composed_scene_is_layered_too() -> None:
    """The mask travels in the environment, so it reaches inside a composite.

    A `Composite` holds a paint closure rather than a list of children, so
    nothing can filter its annotations from outside — which is why the earlier
    scene-level approach left every panelled or gridded page uncontrolled.
    """
    tile = _layered_scene()
    page = with_panel(
        grid([tile, tile], ncols=2), {"split": "train"}, title="ds"
    ).render_html()

    assert 'class="vl-bar"' in page
    kinds = re.findall(r'id="vl-l-(\w+)"', page)
    assert kinds == ["box", "keypoint", "label"]
    assert sorted(set(re.findall(r'id="vl-c-([\w-]+)"', page))) == [
        "car",
        "person",
    ]


_DRAWN = re.compile(r"<(path|rect|circle|ellipse|line|polygon)\s([^>]*)/?>")


def _marks(svg: str) -> list[str]:
    """Every *vector* element the SVG paints, as tag plus geometry.

    Ids are namespaced per fragment, so they are dropped — two pages drawing
    the same picture differ in their scoping and nothing else.

    Rasters are deliberately excluded: a flat page composites a mask or field
    into one image with the photo, while a layered page has to keep them
    apart. `test_chrome_is_not_repeated_into_every_layer` covers those.
    """
    # Clip definitions are not marks: each fragment declares its own viewport
    # clip, which says nothing about what the page paints.
    svg = re.sub(r"<clipPath.*?</clipPath>", "", svg, flags=re.DOTALL)
    found = []
    for tag, attrs in _DRAWN.findall(svg):
        clean = re.sub(r'\s*(?:id|clip-path)="[^"]*"', "", attrs).strip()
        found.append(f"{tag} {clean}")
    return sorted(found)


def test_a_layered_page_draws_exactly_the_same_marks() -> None:
    """The strongest statement of correctness: splitting changes nothing.

    Compares the actual path geometry rather than a count, so it also pins
    *where* each mark lands — which is what catches a pass that fails to
    reserve a chip position it is not going to paint, and chrome repeated into
    every fragment instead of drawn once.
    """
    crowded = _blank(120, 200)
    # Close enough that the two chips compete for the same position, so a pass
    # that forgot to reserve one would move the other.
    crowded.add(BBox(x=0.10, y=0.30, w=0.30, h=0.30).tag("car"))
    crowded.add(BBox(x=0.34, y=0.30, w=0.30, h=0.30).tag("person"))
    crowded.add(BBox(x=0.58, y=0.30, w=0.30, h=0.30).tag("truck"))

    tile = _layered_scene()
    for name, scene in (
        ("image", tile),
        ("crowded", crowded),
        ("filled", _filled_scene()),
        ("grid", grid([tile, _layered_scene()], ncols=2)),
        # The *same* scene object in both cells, which draws each of its
        # annotations twice.
        ("repeated", grid([tile, tile], ncols=2)),
        (
            "panelled",
            with_panel(
                grid([tile, _layered_scene()], ncols=2),
                {"split": "train"},
                title="ds",
            ),
        ),
    ):
        flat = _marks(_svg_of(scene.render_html(controls=False)))
        layered = _marks(_svg_of(scene.render_html()))
        assert layered == flat, name


def test_one_annotation_drawn_twice_gets_both_its_chip_positions() -> None:
    """A reused placement has to follow the chip, not the annotation.

    The same annotation object can be added to a scene twice, and it then places
    two chips in one layout — deliberately in *different* spots, since the
    second must dodge the first. A layered render resolves placement once and
    reuses it; keyed by the annotation alone, the second chip would inherit the
    first one's position. The mark count would not change, only where one of
    them sits.
    """
    box = BBox(x=0.2, y=0.2, w=0.4, h=0.5).tag("car")
    scene = _blank().add(box).add(box)

    layered = _marks(_svg_of(scene.render_html()))
    assert layered == _marks(_svg_of(scene.render_html(controls=False)))


def test_chrome_is_not_repeated_into_every_layer() -> None:
    """A fragment carries its own marks only — no photo, no page, no card."""
    # A filled tile, so a layer that wrongly composites over the photo carries
    # a raster that is not blank and therefore cannot be stripped away.
    tile = _filled_scene()
    scene = with_panel(grid([tile, tile], ncols=2), {"a": "b"}, title="ds")
    svg = _svg_of(scene.render_html())

    # Two tile photos plus each tile's field layer. A layer that repeated the
    # base would embed its own copy of the photo on top of that.
    assert svg.count("<image") == 4


def test_layering_does_not_change_the_picture() -> None:
    """Splitting into fragments must partition the marks, not drop or clone them."""
    scene = _layered_scene()
    flat = _svg_of(scene.render_html(controls=False))
    layered = _svg_of(scene.render_html(controls=True))
    assert layered.count("<path") == flat.count("<path")
    # And the photo is still embedded exactly once, under every fragment.
    assert layered.count("<image") == flat.count("<image") == 1


def test_every_layer_and_class_gets_a_control_that_hides_it() -> None:
    page = _layered_scene().render_html()
    for kind in ("box", "keypoint"):
        assert f'id="vl-l-{kind}"' in page
        assert (
            f"#vl-l-{kind}:not(:checked)~.vl-figure .vl-{kind}{{display:none}}"
            in page
        )
    for name in ("car", "person"):
        assert f'id="vl-c-{name}"' in page
        assert (
            f"#vl-c-{name}:not(:checked)~.vl-figure .vl-cls-{name}{{display:none}}"
            in page
        )


def test_a_class_has_exactly_one_control() -> None:
    """Visibility is a checkbox per class, and nothing else offers the same job.

    An earlier cut also emitted an isolate radio per class, which meant two
    controls competing over one piece of state and two rows saying "car".
    """
    page = _layered_scene().render_html()
    assert 'type="radio"' not in page
    assert page.count('class="vl-bar"') == 1
    for name in ("car", "person"):
        assert page.count(f'for="vl-c-{name}"') == 1


def test_fragment_ids_are_scoped_so_clips_cannot_cross_layers() -> None:
    """Each fragment restarts skia's id counter, so raw ids would collide."""
    page = _layered_scene().render_html()
    ids = re.findall(r'id="([^"]+)"', page)
    assert len(ids) == len(set(ids))
    # Every reference still resolves to an id that exists in the document.
    refs = {
        a or b for a, b in re.findall(r'href="#([^"]+)"|url\(#([^)]+)\)', page)
    }
    assert refs <= set(ids)


def test_controls_off_falls_back_to_the_flat_page() -> None:
    page = _layered_scene().render_html(controls=False)
    assert "vl-layer" not in page
    assert 'type="checkbox"' not in page


def test_class_names_are_slugged_and_escaped() -> None:
    """A class name is arbitrary dataset text; it lands in CSS *and* in HTML."""
    image = Image(np.zeros((60, 80, 3), dtype=np.uint8))
    image.add(BBox(x=0.1, y=0.1, w=0.3, h=0.3).tag('a <b> & "c"'))
    page = image.render_html()
    assert "vl-cls-a-b-c" in page  # a selector-safe slug
    assert "a &lt;b&gt; &amp; &quot;c&quot;" in page  # the chip text, escaped
    # The raw tag must not survive anywhere in the document.
    assert "a <b> &" not in page
    assert '<label for="vl-c-a-b-c">' in page


def _filled_scene() -> "Image":
    """Build a scene with a layer that paints pixels, not just vectors."""
    values = np.linspace(0, 1, 60 * 80).reshape(60, 80).astype(np.float32)
    image = Image(np.zeros((60, 80, 3), dtype=np.uint8))
    image.add(ScalarField(values=values, gradient="viridis", alpha=1.0))
    image.add(BBox(x=0.1, y=0.1, w=0.3, h=0.3).tag("car"))
    return image


def test_a_filled_layer_keeps_its_pixels() -> None:
    """A mask or field composites onto the fragment's transparent base.

    Its raster is therefore structurally identical to the blank base a
    vector-only layer carries, so anything that drops blank bases by shape
    silently deletes exactly the layers that are made of pixels.
    """
    svg = _svg_of(_filled_scene().render_html())
    # The base photo, plus the field layer's own raster. Nothing else.
    assert svg.count("<image") == 2
    assert 'class="vl-layer vl-field"' in svg


def test_a_vector_only_layer_drops_its_blank_base() -> None:
    svg = _svg_of(_filled_scene().render_html())
    # The box layer paints no pixels, so it must not carry an empty raster.
    box = svg[svg.index('class="vl-layer vl-box') :]
    assert "<image" not in box[: box.index("</g>")]


def test_ids_are_scoped_across_layers_that_each_embed_a_raster() -> None:
    """Two rasters both arrive as ``img_0``; unscoped, the second is lost."""
    svg = _svg_of(_filled_scene().render_html())
    ids = re.findall(r'id="([^"]+)"', svg)
    assert len(ids) == len(set(ids))
    assert "b_img_0" in ids
    assert "f0_img_0" in ids
    refs = {
        a or b for a, b in re.findall(r'href="#([^"]+)"|url\(#([^)]+)\)', svg)
    }
    assert refs <= set(ids)


def test_hover_regions_do_not_change_the_mouse_cursor() -> None:
    """The card is the affordance; a help cursor is a badge over the picture.

    ``cursor: help`` renders as the OS arrow-and-question-mark, which sits on
    top of the frame the whole time a region is hovered.
    """
    page = (
        _blank()
        .add(BBox(x=0.1, y=0.1, w=0.3, h=0.3, tooltip=Tooltip(title="car")))
        .render_html()
    )
    style = page[page.index("<style>") : page.index("</style>")]
    assert "cursor:help" not in style
    # The controls are genuinely clickable, so they keep theirs.
    assert "cursor:pointer" in style


def test_chip_placement_survives_being_split_across_layers() -> None:
    """Label layout is collision-aware, so every pass must reserve alike.

    A pass that skipped reserving a chip it was not going to paint would let
    the remaining chips settle somewhere else, and the layered page would no
    longer be the same picture as the flat one.
    """
    image = _blank(120, 200)
    # Two boxes close enough that their chips compete for the same spot.
    image.add(BBox(x=0.10, y=0.30, w=0.30, h=0.30).tag("car"))
    image.add(BBox(x=0.34, y=0.30, w=0.30, h=0.30).tag("person"))

    flat = _svg_of(image.render_html(controls=False))
    layered = _svg_of(image.render_html())
    assert layered.count("<path") == flat.count("<path")


def test_labels_have_their_own_toggle() -> None:
    page = _layered_scene().render_html()
    assert 'id="vl-l-label"' in page
    assert (
        "#vl-l-label:not(:checked)~.vl-figure .vl-label{display:none}" in page
    )


def test_a_layer_that_draws_nothing_gets_no_control() -> None:
    """An array field has no chip; asking for its class's chips draws nothing."""
    values = np.linspace(0, 1, 60 * 80).reshape(60, 80).astype(np.float32)
    image = Image(np.zeros((60, 80, 3), dtype=np.uint8))
    image.add(ScalarField(values=values, gradient="viridis", alpha=1.0))
    page = image.render_html()

    assert 'id="vl-l-field"' in page
    # No labelled annotation at all, so no chip layer and no chip control.
    assert 'id="vl-l-label"' not in page
    assert "vl-layer vl-label" not in page


def test_a_fill_layer_carries_only_its_own_pixels() -> None:
    """A layer must be transparent everywhere it paints nothing.

    Counting rasters cannot show this: a field layer that wrongly composited
    over the photo still emits exactly one image. Only its pixels tell you
    whether the photo came with it — and if it did, hiding the layer would
    take the picture with it.
    """
    pytest.importorskip("PIL")
    # A field covering the left half only, over an opaque photo.
    values = np.zeros((40, 80), dtype=np.float32)
    values[:, :40] = np.linspace(0.2, 1.0, 40)
    scene = Image(np.full((40, 80, 3), 200, dtype=np.uint8))
    scene.add(
        ScalarField(
            values=values, gradient="viridis", alpha=1.0, ignore_value=0.0
        )
    )

    alpha = _layer_alpha(scene.render_html(), "vl-layer vl-field", 80, 40)
    # The right half has no field, so the layer must not paint there.
    assert alpha[:, 45:].max() == 0
    assert alpha[:, :35].max() > 0  # ...and it does paint where it should


def test_a_fill_layer_is_cropped_to_what_it_paints_and_put_back() -> None:
    """A layer's raster covers its marks, not the frame, and is placed by offset.

    Encoding a full frame per layer is what made the export slow — a mask over
    one car in a car park cost a whole-photo PNG. Cropping to the painted region
    only works if the blit is offset by exactly what was cropped away, and an
    error there is invisible in every count-based check: the raster is still
    there, still one of it, just in the wrong place.
    """
    pytest.importorskip("PIL")
    values = np.zeros((40, 80), dtype=np.float32)
    values[10:30, 40:70] = 0.5  # an off-origin block, nowhere near a corner
    scene = Image(np.full((40, 80, 3), 200, dtype=np.uint8))
    scene.add(
        ScalarField(
            values=values, gradient="viridis", alpha=1.0, ignore_value=0.0
        )
    )
    page = scene.render_html()

    field = page[page.index('class="vl-layer vl-field"') :]
    embedded = re.search(r'<image[^>]*width="(\d+)" height="(\d+)"', field)
    assert embedded is not None
    assert (int(embedded.group(1)), int(embedded.group(2))) == (30, 20)

    alpha = _layer_alpha(page, "vl-layer vl-field", 80, 40)
    assert alpha[10:30, 40:70].min() > 0  # every painted pixel landed
    assert alpha.sum() == alpha[10:30, 40:70].sum()  # and nothing else did


def test_an_annotation_that_paints_nothing_gets_no_control() -> None:
    """Registering a layer is not the same as putting marks on the page.

    A keypoint set whose joints are all invisible still reaches the renderer,
    so it would otherwise contribute a layer toggle and a class chip that
    switch nothing at all.
    """
    image = _blank()
    image.add(BBox(x=0.1, y=0.1, w=0.3, h=0.3).tag("car"))
    image.add(Keypoints(keypoints=[(0.5, 0.5, 0), (0.6, 0.6, 0)]).tag("ghost"))
    page = image.render_html()

    assert re.findall(r'id="vl-l-(\w+)"', page) == ["box", "label"]
    assert re.findall(r'id="vl-c-([\w-]+)"', page) == ["car"]
    assert "vl-cls-ghost" not in page


def test_class_names_that_share_a_slug_keep_distinct_controls() -> None:
    """Class names that collapse to one slug must not share one control.

    "Car" and "car" slug identically; duplicate checkbox ids would bind both
    labels to the first input, so one chip toggles both classes and the other
    goes dead.
    """
    image = _blank()
    image.add(BBox(x=0.1, y=0.1, w=0.3, h=0.3).tag("Car"))
    image.add(BBox(x=0.5, y=0.5, w=0.3, h=0.3).tag("car"))
    page = image.render_html()

    ids = re.findall(r'id="(vl-c-[\w-]+)"', page)
    assert len(ids) == 2
    assert len(set(ids)) == 2
    for key in ids:
        assert page.count(f'for="{key}"') == 1
        group = key.removeprefix("vl-c-")
        assert f'<g class="vl-layer vl-box vl-cls-{group}">' in page
        assert f"#{key}:not(:checked)~.vl-figure .vl-cls-{group}" in page
