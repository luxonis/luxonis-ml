"""Text that would overflow the canvas is wrapped (or the canvas is extended)."""

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    Caption,
    Image,
    InfoCard,
    Legend,
    RenderContext,
    grid,
)
from luxonis_ml.vizlab.canvas import Canvas

# --- Canvas.wrap_text -------------------------------------------------------


def test_wrap_text_empty_is_no_lines() -> None:
    assert Canvas.blank(1, 1).wrap_text("", 16, max_width=100) == []


def test_wrap_text_short_line_stays_one_line() -> None:
    cv = Canvas.blank(1, 1)
    assert cv.wrap_text("hi there", 16, max_width=500) == ["hi there"]


def test_wrap_text_wraps_on_spaces_within_width() -> None:
    cv = Canvas.blank(1, 1)
    lines = cv.wrap_text("one two three four five six seven", 16, max_width=70)
    assert len(lines) > 1
    for line in lines:
        assert cv.measure_text(line, 16).width <= 70 + 0.5


def test_wrap_text_hard_breaks_a_too_long_word() -> None:
    cv = Canvas.blank(1, 1)
    # A single word with no spaces still cannot exceed the width.
    lines = cv.wrap_text(
        "supercalifragilisticexpialidocious", 20, max_width=50
    )
    assert len(lines) > 1
    for line in lines:
        assert cv.measure_text(line, 20).width <= 50 + 1.0


# --- overlay cards fit within the canvas ------------------------------------


@pytest.mark.parametrize(
    "card",
    [
        Legend(
            entries=["a_really_long_class_name_that_overflows", "x"],
            title="classes",
        ),
        Caption(text="a_really_long_caption_that_exceeds_the_image_width"),
        InfoCard(
            rows=["text: a_really_long_recognized_value_that_overflows"],
            title="meta",
        ),
    ],
)
def test_overlay_card_fits_within_canvas(card: object) -> None:
    width, height = 240, 200
    canvas = Canvas.blank(width, height)
    ctx = RenderContext(canvas=canvas)
    cells = card._cells(ctx, card.resolve_style(ctx))  # type: ignore[attr-defined]
    assert cells
    # Every card stays inside the canvas (its width fits the margin box).
    for cell in cells:
        assert cell.width <= width - 2 * card.margin + 1.0  # type: ignore[attr-defined]


def test_overlay_card_renders_within_small_image() -> None:
    out = (
        Image(np.full((160, 220, 3), 30, np.uint8))
        .add(Legend(entries=["a_really_long_class_name_here"], title="key"))
        .render()
    )
    assert out.shape == (160, 220, 4)


# --- grid titles ------------------------------------------------------------


def test_grid_wraps_long_titles_taller() -> None:
    img = Image(np.full((80, 120, 3), 30, np.uint8))
    short = grid([img], titles=["hi"]).render()
    long = grid(
        [img],
        titles=["a_really_long_title_that_must_wrap_across_several_lines"],
    ).render()
    # The wrapped title occupies more vertical space, so the grid is taller.
    assert long.shape[0] > short.shape[0]
