"""Inline markup is honoured by every construct that draws text.

Two properties are checked per site, and together they pin the contract down:

1. **Tags are consumed, not drawn.** ``<zz>`` is not a tag, so it renders as
   eleven literal characters and must perturb the output far more than a
   recognized tag does.
2. **Tags take effect.** ``<code>`` switches to the monospace face, so the
   output must change.

Sites that size themselves to their text are checked more sharply still, by
exact width: a recognized tag has to cost exactly nothing.
"""

from collections.abc import Callable

import numpy as np
import pytest

from luxonis_ml.vizlab import (
    BBox,
    Caption,
    ClassDistribution,
    Image,
    InfoCard,
    Keypoints,
    Legend,
    Tooltip,
    grid,
    with_panel,
)
from luxonis_ml.vizlab.annotations.chip import chip_size
from luxonis_ml.vizlab.render.canvas import Canvas
from luxonis_ml.vizlab.style import Style
from luxonis_ml.vizlab.viewer.tooltip_render import render_tooltip_card

WORD = "WWWW"
PLAIN = WORD
STYLED = f"<b>{WORD}</b>"
MONO = f"<code>{WORD}</code>"
LITERAL = f"<zz>{WORD}</zz>"  # not a tag: eleven extra characters if drawn


def _bg(height: int = 220, width: int = 340) -> np.ndarray:
    return np.full((height, width, 3), 128, np.uint8)


def _diff(first: np.ndarray, second: np.ndarray) -> int:
    """Count pixels that differ between two renders.

    A shape mismatch means the content resized, which is itself a difference
    far larger than any glyph change, so it scores as effectively infinite.
    """
    left = np.asarray(first)[..., :3].astype(int)
    right = np.asarray(second)[..., :3].astype(int)
    if left.shape != right.shape:
        return 10**9
    return int((np.abs(left - right).sum(-1) > 12).sum())


# --- constructs that size themselves to their text --------------------------
# Here the contract is exact: a recognized tag must cost zero width.


def test_tooltip_title_parses_markup() -> None:
    """Regression: tooltips used to draw every tag as literal characters."""
    # The title always draws at weight 700, so <b> is a visual no-op and the
    # card must come out exactly as wide as the untagged one.
    plain = render_tooltip_card(Tooltip(title=PLAIN), 14).shape[1]
    styled = render_tooltip_card(Tooltip(title=STYLED), 14).shape[1]
    literal = render_tooltip_card(Tooltip(title=LITERAL), 14).shape[1]
    assert styled == plain
    assert literal > plain


def test_tooltip_rows_parse_markup() -> None:
    def width(value: str) -> int:
        return render_tooltip_card(Tooltip(rows=(("k", value),)), 14).shape[1]

    assert width(STYLED) < width(LITERAL)
    assert width(LITERAL) > width(PLAIN)


def test_panel_keys_and_values_parse_markup() -> None:
    """Regression: panel rows used to draw every tag as literal characters."""

    def width(text: str) -> int:
        return np.asarray(
            with_panel(Image(_bg()), {text: text}).render()
        ).shape[1]

    key = "mmmmmmmmmmmm"  # drives the panel width, but stays under the clamp
    assert width(f"<b>{key}</b>") == width(key)
    assert width(f"<zz>{key}</zz>") > width(key)


def test_label_chip_parses_markup() -> None:
    """Regression: annotation label chips used to draw tags literally."""
    canvas, style = Canvas.blank(4, 4), Style()

    def width(text: str) -> float:
        return chip_size(canvas, text, style)[0]

    assert width(LITERAL) > 2 * width(PLAIN)  # eleven characters' worth
    assert width(STYLED) > width(PLAIN)  # bold is a little wider
    assert width(MONO) < width(PLAIN)  # mono W is narrower than Inter's


# --- constructs drawn onto a fixed canvas -----------------------------------


def _caption(text: str) -> np.ndarray:
    return Image(_bg()).add(Caption(text=text)).render()


def _info_card(text: str) -> np.ndarray:
    return Image(_bg()).add(InfoCard(rows=[text], title="t")).render()


def _info_card_title(text: str) -> np.ndarray:
    return Image(_bg()).add(InfoCard(rows=["r"], title=text)).render()


def _legend(text: str) -> np.ndarray:
    return Image(_bg()).add(Legend(entries=[text])).render()


def _legend_overflowing(text: str) -> np.ndarray:
    """Render a `Legend` tall enough to be forced into its column layout.

    The canvas is wide so the names are not also ellipsized, which would hide
    the difference this checks for.
    """
    entries = [text] + [f"c{i}" for i in range(12)]
    return (
        Image(_bg(height=150, width=900)).add(Legend(entries=entries)).render()
    )


def _grid_title(text: str) -> np.ndarray:
    return grid([Image(_bg())], titles=[text]).render()


def _panel_title(text: str) -> np.ndarray:
    return with_panel(Image(_bg()), {"k": "v"}, title=text).render()


def _keypoint_labels(text: str) -> np.ndarray:
    return (
        Image(_bg())
        .add(
            Keypoints(
                keypoints=[(0.3, 0.3, 2), (0.6, 0.6, 2)],
                keypoint_names=[text, "other"],
                point_labels="names",
            )
        )
        .render()
    )


def _distribution_names(text: str) -> np.ndarray:
    return (
        Image(_bg())
        .add(ClassDistribution(probabilities={text: 0.7, "b": 0.3}))
        .render()
    )


def _distribution_title(text: str) -> np.ndarray:
    return (
        Image(_bg())
        .add(ClassDistribution(probabilities={"a": 0.7}, title=text))
        .render()
    )


SITES = [
    ("Caption.text", _caption),
    ("InfoCard.rows", _info_card),
    ("InfoCard.title", _info_card_title),
    ("Legend.entries", _legend),
    ("Legend.entries overflowing", _legend_overflowing),
    ("grid titles", _grid_title),
    ("with_panel title", _panel_title),
    ("Keypoints.keypoint_names", _keypoint_labels),
    ("ClassDistribution.probabilities", _distribution_names),
    ("ClassDistribution.title", _distribution_title),
]
_IDS = [name for name, _ in SITES]
_RENDERERS = [render for _, render in SITES]


@pytest.mark.parametrize("render", _RENDERERS, ids=_IDS)
def test_unknown_tags_perturb_far_more_than_real_ones(
    render: "Callable[[str], np.ndarray]",
) -> None:
    """A drawn ``<zz>…</zz>`` moves many more pixels than a consumed tag."""
    plain = render(PLAIN)
    assert _diff(render(LITERAL), plain) > _diff(render(STYLED), plain)


@pytest.mark.parametrize("render", _RENDERERS, ids=_IDS)
def test_markup_changes_what_is_drawn(
    render: "Callable[[str], np.ndarray]",
) -> None:
    """``<code>`` switches the face, so the pixels have to change."""
    assert not np.array_equal(render(PLAIN), render(MONO))


def _card_width(array: np.ndarray) -> int:
    """Width in pixels of the drawn card, as its non-background bounding box."""
    rgb = np.asarray(array)[..., :3].astype(int)
    painted = np.abs(rgb - rgb[0, 0]).sum(-1) > 12
    columns = np.where(painted.any(0))[0]
    return int(columns[-1] - columns[0] + 1) if len(columns) else 0


def test_legend_overflow_columns_size_entries_like_the_single_column() -> None:
    """Regression: the overflow column layout used to ignore markup entirely.

    `Legend` flows into columns once its card would not fit the canvas. That
    path measured and drew raw strings, so one and the same legend rendered
    italic on a tall canvas and showed literal ``<i>`` tags on a short one —
    and, because the column width is derived from the widest name, the whole
    card was sized to the tags too.

    Both layouts must now agree: a recognized tag costs no width, an
    unrecognized one costs its characters.
    """
    for entries_fit_in_one_column in (False, True):
        height = 600 if entries_fit_in_one_column else 150

        def card(text: str, height: int = height) -> int:
            legend = Legend(entries=[text] + [f"c{i}" for i in range(12)])
            return _card_width(
                Image(_bg(height=height, width=900)).add(legend).render()
            )

        assert card(f"<i>{WORD}</i>") == card(WORD)
        assert card(LITERAL) > card(WORD)


# --- escaping at the adapter boundary ---------------------------------------


def test_authored_annotation_text_is_still_markup() -> None:
    """Escaping is for dataset text; what the caller writes stays markup."""
    canvas, style = Canvas.blank(4, 4), Style()
    assert (
        chip_size(canvas, MONO, style)[0] != chip_size(canvas, PLAIN, style)[0]
    )
    image = Image(_bg()).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.5, label=f"<i>{WORD}</i>")
    )
    plain_box = Image(_bg()).add(BBox(x=0.1, y=0.1, w=0.5, h=0.5, label=WORD))
    assert not np.array_equal(image.render(), plain_box.render())
