"""Bundled fonts (Inter + JetBrains Mono) and inline markup / styled spans."""

import numpy as np
import pytest

from luxonis_ml.vizlab import Caption, Image, InfoCard
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.render.fonts import DEFAULT_FONTS
from luxonis_ml.vizlab.render.markup import Span, escape, parse
from luxonis_ml.vizlab.render.text_layout import tracked_spans_width, width
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


def test_decoration_tags_and_their_aliases() -> None:
    assert parse("<u>x</u>")[0].underline is True
    assert parse("<s>x</s>")[0].strike is True
    assert parse("<del>x</del>")[0].strike is True
    assert parse("<tt>x</tt>")[0].mono is True


def test_tags_are_case_insensitive() -> None:
    assert parse("<B>x</B>")[0].weight == 700
    assert parse("<U>x</U>")[0].underline is True


def test_span_carries_color_weight_and_size() -> None:
    span = parse("<span color='#ff0000' weight='300' size='1.5'>x</span>")[0]
    assert span.color is not None
    assert span.color.rgb == (255, 0, 0)
    assert span.weight == 300
    assert span.scale == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("value", "weight"),
    [("bold", 700), ("semibold", 600), ("light", 300), ("850", 850)],
)
def test_span_accepts_named_and_numeric_weights(
    value: str, weight: int
) -> None:
    assert parse(f'<span weight="{value}">x</span>')[0].weight == weight


@pytest.mark.parametrize(
    ("value", "scale"),
    [("2", 2.0), ("150%", 1.5), ("large", 1.2), ("small", 1 / 1.2)],
)
def test_span_accepts_multiplier_percent_and_named_sizes(
    value: str, scale: float
) -> None:
    got = parse(f'<span size="{value}">x</span>')[0].scale
    assert got == pytest.approx(scale)


def test_span_attributes_apply_only_inside_the_tag() -> None:
    spans = parse("a<span color='#00ff00'>b</span>c")
    assert [s.color is None for s in spans] == [True, False, True]


@pytest.mark.parametrize(
    "markup",
    [
        '<span colour="#fff">x</span>',  # misspelled attribute
        '<span color="definitely-not-a-color">x</span>',
        '<span size="0">x</span>',  # non-positive
        '<span weight="heavyish">x</span>',
        "<span color=#fff>x</span>",  # unquoted value
    ],
)
def test_malformed_span_raises_rather_than_failing_silently(
    markup: str,
) -> None:
    # <span> is explicit authored intent, so a typo is an error, not a shrug.
    with pytest.raises(ValueError, match=r"invalid|unknown|malformed"):
        parse(markup)


def test_unknown_tags_never_raise() -> None:
    # Only <span> is strict; an unrecognized tag is text, not a mistake.
    assert parse("<blink>x</blink>")[0].text == "<blink>x</blink>"


@pytest.mark.parametrize(
    ("markup", "text"),
    [
        ("&lt;b&gt;", "<b>"),
        ("a &amp; b", "a & b"),
        ("&quot;q&quot; &apos;a&apos;", "\"q\" 'a'"),
    ],
)
def test_entities_decode_to_their_characters(markup: str, text: str) -> None:
    assert parse(markup)[0].text == text


def test_escape_round_trips_through_parse() -> None:
    raw = 'a < b & c > d "q" <b>not bold</b>'
    assert "".join(s.text for s in parse(escape(raw))) == raw
    # And nothing survives as styling.
    assert all(s.weight == 400 for s in parse(escape(raw)))


def test_crossed_nesting_closes_inner_tags_html_style() -> None:
    # </b> closes the <i> opened inside it, so "c" is neither bold nor italic.
    spans = parse("<b>a<i>b</b>c</i>")
    assert [(s.text, s.weight, s.italic) for s in spans] == [
        ("a", 700, False),
        ("b", 700, True),
        ("c", 400, False),
    ]


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


def _ink(markup: str, **kwargs: object) -> int:
    """Count painted pixels for one line of markup on a fresh canvas."""
    cv = _canvas()
    cv.markup(
        (5, 50),
        markup,
        size=28,
        color=Color(255, 255, 255),
        **kwargs,  # type: ignore[arg-type]
    )
    return int((cv.to_rgba()[..., 3] > 0).sum())


def test_underline_and_strike_add_ink() -> None:
    plain = _ink("word")
    assert _ink("<u>word</u>") > plain
    assert _ink("<s>word</s>") > plain
    # Both rules together add more than either alone.
    assert _ink("<u><s>word</s></u>") > max(
        _ink("<u>word</u>"), _ink("<s>word</s>")
    )


def test_span_color_paints_its_own_color_not_the_draw_color() -> None:
    cv = _canvas()
    cv.markup(
        (5, 50),
        "a<span color='#ff0000'>R</span>b",
        size=28,
        color=Color(255, 255, 255),
    )
    rgb = cv.to_rgba()[..., :3].reshape(-1, 3)
    red = (rgb[:, 0] > 150) & (rgb[:, 1] < 80) & (rgb[:, 2] < 80)
    assert int(red.sum()) > 0


def test_span_size_scales_measurement_and_ink() -> None:
    cv = _canvas()
    plain = cv.measure_markup("word", 20)
    big = cv.measure_markup("<span size='2'>word</span>", 20)
    assert big.width == pytest.approx(plain.width * 2, rel=0.02)
    assert big.height > plain.height
    assert _ink("<span size='2'>word</span>") > _ink("word")


def test_measure_markup_does_not_count_tags_as_characters() -> None:
    cv = _canvas()
    tagged = cv.measure_markup("<b>car</b>", 14)
    assert tagged.width == pytest.approx(
        cv.measure_text("car", 14, weight=700).width
    )
    # Whereas measuring the same string as plain text does count them.
    assert cv.measure_text("<b>car</b>", 14, weight=700).width > tagged.width


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


def test_tracked_span_widths_measure_in_the_spans_own_face() -> None:
    # Letterspaced headings advance by measured char widths; measuring a mono
    # run in the sans face makes the glyphs and the advances disagree.
    mono = tracked_spans_width([Span("mm", mono=True)], 16.0)
    sans = tracked_spans_width([Span("mm")], 16.0)
    assert mono == pytest.approx(2 * width("m", 16.0, mono=True))
    assert sans == pytest.approx(2 * width("m", 16.0))
    assert mono != pytest.approx(sans)
