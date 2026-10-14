"""The Luxonis brand colors and the UI-chrome palette built from them.

These are the company colors, transcribed from the shared Luxonis design tokens
that the frontend renders from, so a chart or overlay produced here matches the
web UI rather than drifting from it. They are the
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
    >>> brand.chrome_for(brand.LIGHT_BACKGROUND).card_text is brand.PURPLE_TEXT
    True

"""

from dataclasses import dataclass

from .base import Color

# -- Core brand colors ------------------------------------------------------
# The saturated fill of each semantic family, for filled surfaces rather than
# for text (the `*_TEXT` ramp below is what labels and icons should use).

#: The primary "Luxonis Purple" (#4C4FF1), the fill of the purple family.
PURPLE = Color(76, 79, 241)
#: The fill of the green family (#12B76A).
GREEN = Color(18, 183, 106)
#: The fill of the orange family (#DC6803).
ORANGE = Color(220, 104, 3)
#: The fill of the red family (#F04438).
RED = Color(240, 68, 56)

# The lighter "decoration" variant of each family, which the design system also
# uses as that family's outline/border color.

#: Light purple (#8DA4F4), the decoration and border color of the purple
#: family.
PERIWINKLE = Color(141, 164, 244)
#: Light green (#6CE9A6), the decoration and border color of the green family.
MINT = Color(108, 233, 166)
#: Light orange (#FEC84B), the decoration and border color of the orange
#: family.
AMBER = Color(254, 200, 75)
#: Light red (#FDA29B), the decoration and border color of the red family.
SALMON = Color(253, 162, 155)

#: A deep-indigo shade of :data:`PURPLE` (same hue) for light-mode titles and
#: headings: a stronger, higher-contrast purple that outranks :data:`PURPLE_TEXT`
#: used for body text (≈ 15:1 on white). The design system has no heading token,
#: so this one is ours.
PURPLE_TITLE = Color(22, 24, 112)  # #161870

# Neutral "ink" ramp (brand grays), dark to light: body text, labels, and the
# muted/disabled end of the scale.

#: The darkest neutral (#1D2939).
INK = Color(29, 41, 57)
#: A muted mid neutral (#475467).
SLATE = Color(71, 84, 103)
#: A light neutral (#667085).
STEEL = Color(102, 112, 133)
#: The lightest neutral (#D3D3D3), for disabled text and placeholders.
FAINT = Color(211, 211, 211)

# -- Soft tints -------------------------------------------------------------
# Pale fills for chips, badges, and callouts, one per family. The active tint is
# a pale blue rather than a wash of the indigo brand purple; that is what the
# design system specifies, not a transcription slip.

#: The pale fill of the purple (active) family (#DCEFFC).
PURPLE_SOFT = Color(220, 239, 252)
#: The pale fill of the neutral family (#F2F4F7).
GRAY_SOFT = Color(242, 244, 247)
#: The pale fill of the green family (#ECFDF3).
GREEN_SOFT = Color(236, 253, 243)
#: The pale fill of the orange family (#FFFAEB).
ORANGE_SOFT = Color(255, 250, 235)
#: The pale fill of the red family (#FEF3F2).
RED_SOFT = Color(254, 243, 242)

# -- Foreground text colors -------------------------------------------------
# Darkened per-family colors for text and icons, contrast-tuned to sit on the
# soft tints above (or on white). Use these instead of the solids for anything
# a reader has to actually read.

#: Text and icons of the purple family (#5724E8).
PURPLE_TEXT = Color(87, 36, 232)
#: Neutral text and icons (#344054).
GRAY_TEXT = Color(52, 64, 84)
#: Text and icons of the green family (#027A48).
GREEN_TEXT = Color(2, 122, 72)
#: Text and icons of the orange family (#B54708).
ORANGE_TEXT = Color(181, 71, 8)
#: Text and icons of the red family (#B42318).
RED_TEXT = Color(180, 35, 24)

#: Neutral hairline border, for rules and outlines with no semantic color.
GRAY_BORDER = Color(229, 229, 229)  # #E5E5E5

#: The design system draws hover states as the base color at 85% alpha, so
#: ``PURPLE.with_alpha(HOVER_ALPHA)`` is the hover fill for a purple surface.
HOVER_ALPHA = 0xD9

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
#: Keys and secondary accents on dark cards.
CARD_KEY = PERIWINKLE
#: The subtle rule between an image and its side panel on dark backgrounds.
DIVIDER = PERIWINKLE.with_alpha(40)
#: Empty track fills and the "other" segment of a chart.
MUTED = SLATE

# -- Light-mode chrome (white cards + brand-purple text on the light surface) --
#: Near-opaque white card fill, so panels pop against the lavender background.
LIGHT_CARD_BG = Color(255, 255, 255, 236)
#: Caption chips in light mode, opaque white for a single crisp line.
LIGHT_CAPTION_BG = Color(255, 255, 255, 240)
#: Body text on light cards, in the text purple of the family.
LIGHT_CARD_TEXT = PURPLE_TEXT
#: Headings on light cards, deeper than the body text so they outrank it.
LIGHT_CARD_TITLE = PURPLE_TITLE
#: Keys and accents on light cards, in the same text purple as the body.
LIGHT_CARD_KEY = PURPLE_TEXT
#: Hairline card border and image/panel rule, a translucent brand purple.
LIGHT_CARD_BORDER = PURPLE.with_alpha(38)
#: The rule between an image and its side panel on light backgrounds.
LIGHT_DIVIDER = PURPLE.with_alpha(46)

# Semantic accents for chart chrome (not class labels).

#: The primary highlight of chart chrome.
ACCENT = PURPLE
#: A correct (✓) verdict.
SUCCESS = GREEN
#: A warning in chart chrome.
WARNING = ORANGE
#: An incorrect (✗) verdict.
ERROR = RED


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
"""Chrome for dark backgrounds, with navy cards and light periwinkle text."""

LIGHT_CHROME = Chrome(
    card_bg=LIGHT_CARD_BG,
    caption_bg=LIGHT_CAPTION_BG,
    card_text=LIGHT_CARD_TEXT,
    card_title=LIGHT_CARD_TITLE,
    card_key=LIGHT_CARD_KEY,
    divider=LIGHT_DIVIDER,
    border=LIGHT_CARD_BORDER,
)
"""Chrome for light backgrounds, with white cards and deep-purple text."""


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
    return LIGHT_CHROME if background.is_light else DARK_CHROME
