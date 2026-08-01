"""The Luxonis brand colors and the UI-chrome palette built from them.

These are the company colors, taken from the Luxonis design tokens. They are the
single home for every non-label color in the stack: visualization *chrome* —
composite backgrounds, card fills, panel keys and titles, dividers, chart tracks,
and verdict marks — is colored from here so the framing reads as on-brand out of
the box. Per-class *label* colors deliberately do **not** come from here: they
stay maximally distinct via the golden-ratio generator (see
`luxonis_ml.utils.color.palette`), because a fixed brand set can't keep many
classes apart. Reach for these constants when coloring anything that is not a
class label.

Examples:
    >>> from luxonis_ml.utils.color import brand
    >>> brand.PURPLE
    Color(r=76, g=79, b=241, a=255)
    >>> brand.CARD_KEY is brand.PERIWINKLE
    True
    >>> brand.chrome_for(brand.LIGHT_BACKGROUND).card_text is brand.PURPLE
    True

"""

from dataclasses import dataclass

from .base import Color

# -- Core brand colors (the Luxonis design tokens) --------------------------
PURPLE = Color(76, 79, 241)  # #4c4ff1 — the primary "Luxonis Purple"
GREEN = Color(18, 183, 106)  # #12B76A
ORANGE = Color(220, 104, 3)  # #DC6803
RED = Color(240, 68, 56)  # #F04438

PERIWINKLE = Color(141, 164, 244)  # #8da4f4 — light purple
MINT = Color(108, 233, 166)  # #6CE9A6 — light green
AMBER = Color(254, 200, 75)  # #FEC84B — light orange
SALMON = Color(253, 162, 155)  # #FDA29B — light red

#: A deep-indigo shade of :data:`PURPLE` (same hue) for light-mode titles and
#: headings: a stronger, higher-contrast purple that outranks the regular
#: brand purple used for body text (≈ 15:1 on white).
PURPLE_TITLE = Color(22, 24, 112)  # #161870

# Neutral "ink" ramp (brand grays), dark to light.
INK = Color(29, 41, 57)  # #1D2939 — darkest neutral
SLATE = Color(71, 84, 103)  # #475467 — muted mid neutral
STEEL = Color(102, 112, 133)  # #667085 — light neutral

# -- Chrome palette (semantic; used across the annotations) -----------------
#: Deep navy composite background painted behind stacks, grids, and pad gaps.
BACKGROUND = INK.darken(0.28)
#: Light-mode composite background — a soft, cool lavender-gray (a hair of brand
#: purple mixed into white). Deliberately *not* pure white so white cards and
#: panels read as distinct surfaces on top of it.
LIGHT_BACKGROUND = Color(237, 239, 248)  # #edeff8

#: Card fill — a translucent navy sitting a touch lighter than ``BACKGROUND`` so
#: stacked cards (legend, info card, distribution panel) read as one family.
CARD_BG = INK.with_alpha(150)
#: Caption chips use the same navy, more opaque for a single short line.
CAPTION_BG = INK.with_alpha(235)
#: Light-purple (periwinkle) body text — the dark-mode counterpart to the
#: regular-purple body text on light surfaces, so both themes read as on-brand
#: rather than plain white on dark.
CARD_TEXT = PERIWINKLE
#: A lighter lavender heading, brighter than the body so it outranks it on dark.
CARD_TITLE = PERIWINKLE.lighten(0.4)  # ≈ #bbc8f8
CARD_KEY = PERIWINKLE  # keys / secondary accents on cards
DIVIDER = PERIWINKLE.with_alpha(40)  # subtle rule between image and panel
MUTED = SLATE  # empty track fills and the "other" segment

# -- Light-mode chrome (white cards + brand-purple text on the light surface) --
#: Near-opaque white card fill, so panels pop against the lavender background.
LIGHT_CARD_BG = Color(255, 255, 255, 236)
#: Caption chips in light mode: opaque white for a single crisp line.
LIGHT_CAPTION_BG = Color(255, 255, 255, 240)
LIGHT_CARD_TEXT = PURPLE  # body text in the regular brand purple
LIGHT_CARD_TITLE = PURPLE_TITLE  # deeper heading, to outrank the body text
LIGHT_CARD_KEY = PURPLE  # keys / accents in the bright brand purple
#: Hairline card border and image/panel rule, a translucent brand purple.
LIGHT_CARD_BORDER = PURPLE.with_alpha(38)
LIGHT_DIVIDER = PURPLE.with_alpha(46)

# Semantic accents for chart chrome (not class labels).
ACCENT = PURPLE  # primary highlight
SUCCESS = GREEN  # a correct ✓ verdict
WARNING = ORANGE
ERROR = RED  # an incorrect ✗ verdict


@dataclass(frozen=True)
class Chrome:
    """Resolved card/panel chrome for a given composite background.

    Bundles the handful of non-label colors a stacked card or side panel needs —
    fill, body/heading text, accent key, divider, and an optional hairline
    border — so the framing adapts to a dark or light background as one set.
    Resolve it with `chrome_for`; the dark and light presets are `DARK_CHROME`
    and `LIGHT_CHROME`.

    Attributes:
        card_bg: Rounded-card fill.
        caption_bg: Caption-chip fill (a more opaque variant of ``card_bg``).
        card_text: Body/value text on a card.
        card_title: Heading text on a card.
        card_key: Key/accent color (labels of key/value rows, chart accents).
        divider: Subtle rule between an image and its side panel.
        border: Hairline card outline, or ``None`` to draw none (dark cards rely
            on their shadow instead of a border).

    """

    card_bg: Color
    caption_bg: Color
    card_text: Color
    card_title: Color
    card_key: Color
    divider: Color
    border: Color | None = None


DARK_CHROME = Chrome(
    card_bg=CARD_BG,
    caption_bg=CAPTION_BG,
    card_text=CARD_TEXT,
    card_title=CARD_TITLE,
    card_key=CARD_KEY,
    divider=DIVIDER,
    border=None,
)
"""Chrome for dark composite backgrounds: navy cards, near-white text."""

LIGHT_CHROME = Chrome(
    card_bg=LIGHT_CARD_BG,
    caption_bg=LIGHT_CAPTION_BG,
    card_text=LIGHT_CARD_TEXT,
    card_title=LIGHT_CARD_TITLE,
    card_key=LIGHT_CARD_KEY,
    divider=LIGHT_DIVIDER,
    border=LIGHT_CARD_BORDER,
)
"""Chrome for light composite backgrounds: white cards, deep-purple text."""


def chrome_for(background: Color) -> Chrome:
    """Return the card/panel chrome that suits ``background``.

    Picks `LIGHT_CHROME` for a light background (one that reads best with dark
    text) and `DARK_CHROME` otherwise, so cards drawn on top stay legible and
    on-brand either way.

    Args:
        background: The composite background the chrome will sit on.

    Returns:
        The matching `Chrome` preset.

    Examples:
        >>> chrome_for(BACKGROUND) is DARK_CHROME
        True
        >>> chrome_for(LIGHT_BACKGROUND) is LIGHT_CHROME
        True

    """
    return (
        LIGHT_CHROME
        if background.readable_text_color().r < 128
        else DARK_CHROME
    )
