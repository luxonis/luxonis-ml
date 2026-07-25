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

"""

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

# Neutral "ink" ramp (brand grays), dark to light.
INK = Color(29, 41, 57)  # #1D2939 — darkest neutral
SLATE = Color(71, 84, 103)  # #475467 — muted mid neutral
STEEL = Color(102, 112, 133)  # #667085 — light neutral

# -- Chrome palette (semantic; used across the annotations) -----------------
#: Deep navy composite background painted behind stacks, grids, and pad gaps.
BACKGROUND = INK.darken(0.28)
#: Light-mode composite background (the brand light-gray surface, #F2F4F7).
LIGHT_BACKGROUND = Color(242, 244, 247)

#: Card fill — a translucent navy sitting a touch lighter than ``BACKGROUND`` so
#: stacked cards (legend, info card, distribution panel) read as one family.
CARD_BG = INK.with_alpha(224)
#: Caption chips use the same navy, more opaque for a single short line.
CAPTION_BG = INK.with_alpha(235)
CARD_TEXT = Color(240, 242, 247)  # cool near-white body text
CARD_TITLE = Color(245, 246, 250)  # slightly brighter card heading
CARD_KEY = PERIWINKLE  # keys / secondary accents on cards
DIVIDER = PERIWINKLE.with_alpha(40)  # subtle rule between image and panel
MUTED = SLATE  # empty track fills and the "other" segment

# Semantic accents for chart chrome (not class labels).
ACCENT = PURPLE  # primary highlight
SUCCESS = GREEN  # a correct ✓ verdict
WARNING = ORANGE
ERROR = RED  # an incorrect ✗ verdict
