"""Bundled fonts (Inter + JetBrains Mono) and inline markup / styled spans."""

import numpy as np
import pytest

from luxonis_ml.vizlab import Caption, Image, InfoCard
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.render.fonts import DEFAULT_FONTS
from luxonis_ml.vizlab.render.markup import Span, parse
from luxonis_ml.vizlab.style import Style


def _canvas(w: int = 400, h: int = 80) -> Canvas:
    return Canvas.blank(w, h)


# --- markup parsing ---------------------------------------------------------


def test_plain_text_is_one_span() -> None:
    spans = parse("hello world")
    assert spans == [Span("hello world", 400, False, False)]


def test_tags_switch_weight_slant_family() -> None:
    styled = [
        (s.text, s.weight, s.italic, s.mono)
        for s in parse("a <b>b</b> <i>c</i> <code>d</code>")
    ]
    assert styled == [
        ("a ", 400, False, False),
        ("b", 700, False, False),
        (" ", 400, False, False),
        ("c", 400, True, False),
        (" ", 400, False, False),
        ("d", 400, False, True),
    ]


def test_tags_nest_and_combine() -> None:
    inner = parse("<b>x <i>y</i></b>")[1]
    assert (inner.text, inner.weight, inner.italic) == ("y", 700, True)


def test_aliases_match_primary_tags() -> None:
    assert parse("<strong>x</strong>")[0].weight == 700
    assert parse("<em>x</em>")[0].italic is True
    assert parse("<mono>x</mono>")[0].mono is True


def test_unknown_tags_and_stray_bracket_stay_literal() -> None:
    # A non-tag "<foo>" and a bare "<" render as text, untouched.
    assert parse("3 < 4 <foo>bar</foo>") == [
        Span("3 < 4 <foo>bar</foo>", 400, False, False)
    ]


def test_unbalanced_close_tag_is_ignored() -> None:
    spans = parse("plain </b> text")
    # The stray close tag is dropped; the text stays regular weight throughout.
    assert "".join(s.text for s in spans) == "plain  text"
    assert all(s.weight == 400 for s in spans)


def test_baseline_style_layers_under_tags() -> None:
    spans = parse("a <b>b</b>", weight=500, italic=True, mono=True)
    # Untagged text keeps the baseline; <b> only overrides the weight.
    assert (spans[0].weight, spans[0].italic, spans[0].mono) == (
        500,
        True,
        True,
    )
    assert (spans[1].weight, spans[1].italic, spans[1].mono) == (
        700,
        True,
        True,
    )


# --- font selection ---------------------------------------------------------


def test_mono_family_has_uniform_advance() -> None:
    cv = _canvas()
    narrow = cv.measure_text("iiiii", 24, mono=True).width
    wide = cv.measure_text("mmmmm", 24, mono=True).width
    assert narrow == pytest.approx(wide)


def test_sans_family_is_proportional() -> None:
    cv = _canvas()
    narrow = cv.measure_text("iiiii", 24).width
    wide = cv.measure_text("mmmmm", 24).width
    assert wide > narrow


def test_font_manager_serves_every_axis() -> None:
    # Each (weight, italic, mono) combination yields a usable font.
    for mono in (False, True):
        for italic in (False, True):
            font = DEFAULT_FONTS.font(20, weight=700, italic=italic, mono=mono)
            assert font.getSize() == 20


# --- styled spans on the canvas ---------------------------------------------


def test_measure_spans_sums_run_widths() -> None:
    cv = _canvas()
    spans = parse("foo <b>bar</b> <code>baz</code>")
    total = sum(cv._measure_span(s, 22).width for s in spans)
    assert cv.measure_spans(spans, 22).width == pytest.approx(total)


def test_draw_spans_paints_pixels() -> None:
    cv = _canvas()
    cv.draw_spans(
        (5, 40), parse("<b>hi</b>"), size=28, color=Color(255, 255, 255)
    )
    assert int((cv.to_rgba()[..., 3] > 0).sum()) > 0


def test_wrap_spans_preserves_style_and_breaks_on_newline() -> None:
    cv = _canvas()
    lines = cv.wrap_spans(parse("a <b>B</b>\nc"), 20, max_width=1000)
    assert len(lines) == 2  # the \n forces a second line
    # The bold run kept its weight through wrapping.
    assert any(s.text == "B" and s.weight == 700 for s in lines[0])


def test_wrap_spans_hard_breaks_overlong_word() -> None:
    cv = _canvas()
    lines = cv.wrap_spans(
        parse("<code>" + "x" * 60 + "</code>"), 20, max_width=80
    )
    assert len(lines) > 1
    # Every produced line fits the width budget.
    assert all(cv.measure_spans(line, 20).width <= 80 + 1.0 for line in lines)


# --- Style.font_family ------------------------------------------------------


def test_style_font_family_defaults_to_sans() -> None:
    assert Style().font_family == "sans"
    assert Style().mono is False
    assert Style(font_family="mono").mono is True


def test_style_scaled_keeps_family() -> None:
    assert Style(font_family="mono").scaled(2.0).font_family == "mono"


# --- integration: markup is visible in cards --------------------------------


def _render(card: object) -> np.ndarray:
    img = Image(np.full((160, 320, 3), 24, np.uint8))
    return img.add(card).render()[..., :3]  # type: ignore[arg-type]


def test_bold_markup_changes_card_pixels() -> None:
    plain = _render(InfoCard(rows=["value"], title="t"))
    bold = _render(InfoCard(rows=["<b>value</b>"], title="t"))
    assert not np.array_equal(plain, bold)


def test_mono_markup_changes_caption_pixels() -> None:
    plain = _render(Caption(text="frame_0007"))
    mono = _render(Caption(text="<code>frame_0007</code>"))
    assert not np.array_equal(plain, mono)
