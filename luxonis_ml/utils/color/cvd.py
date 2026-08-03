"""Simulating color-vision deficiency, and measuring how far apart colors look.

Roughly one man in twelve and one woman in two hundred has some form of
color-vision deficiency (CVD), so a palette that separates classes only by hue
separates them only for most viewers. This module is the evidence behind the
word "colorblind-safe" in :data:`luxonis_ml.utils.color.palette.PALETTES`: it is
what the shipped palettes are checked with, and what
`luxonis_ml.utils.color.palette.CVDDistinctColors` searches against when it has
to invent colors past the end of a named set.

Two pieces, which are only useful together:

- `simulate` renders a color the way a viewer with protanopia, deuteranopia, or
  tritanopia sees it, using the Machado, Oliveira & Fernandes (2009) model — the
  same one Chrome DevTools and matplotlib-adjacent tooling use. It is a
  *dichromacy* model at full severity, i.e. the worst case, which is the case
  worth designing for.
- `delta_e` measures how far apart two colors look, as a CIEDE2000 distance in
  CIELAB. The unit is calibrated so ``1.0`` is about a just-noticeable
  difference for adjacent patches; ``10`` is comfortably "different colors" for
  the small, separated, textured marks a visualization actually draws.

`min_separation` puts them together and answers the only question that matters
for a class palette: once the eye has collapsed a dimension of color space, can
two classes still be told apart? It is public rather than a test helper on
purpose — pinned class colors (`Palette.pin`) bypass every guarantee the
built-in palettes make, and a caller who picks their own colors should be able
to run the same check the library runs on its own.

Examples:
    Red and green are a textbook confusion pair — far apart to most viewers,
    nearly the same color to a deuteranope:

    >>> from luxonis_ml.utils.color.cvd import delta_e, simulate
    >>> round(delta_e("#d62728", "#2ca02c"))
    72
    >>> round(
    ...     delta_e(
    ...         simulate("#d62728", "deuteranopia"),
    ...         simulate("#2ca02c", "deuteranopia"),
    ...     )
    ... )
    5

    Blue and orange survive the same treatment, which is why every
    colorblind-safe palette leans on that axis:

    >>> round(
    ...     delta_e(
    ...         simulate("#0072b2", "deuteranopia"),
    ...         simulate("#e69f00", "deuteranopia"),
    ...     )
    ... )
    63

"""

import math
from collections.abc import Iterable, Sequence
from typing import Literal, TypeAlias

from .base import Color, ColorLike

Deficiency: TypeAlias = Literal["protanopia", "deuteranopia", "tritanopia"]
"""One of the three dichromacies `simulate` models.

Protanopia (no long-wavelength cones) and deuteranopia (no medium-wavelength
cones) are the common red-green forms; tritanopia (no short-wavelength cones) is
the rare blue-yellow one.
"""

DEFICIENCIES: tuple[Deficiency, ...] = (
    "protanopia",
    "deuteranopia",
    "tritanopia",
)
"""Every :data:`Deficiency`, for looping over all three."""

VISION: tuple[Deficiency | None, ...] = (None, *DEFICIENCIES)
"""Normal vision (``None``) plus every deficiency — `min_separation`'s default."""

Lab: TypeAlias = tuple[float, float, float]
"""A CIELAB triple: lightness in ``[0, 100]``, then the two opponent axes."""

# Machado, Oliveira & Fernandes (2009), "A Physiologically-based Model for
# Simulation of Color Vision Deficiency", table 1 at severity 1.0. The matrices
# act on *linear* RGB, so `simulate` linearizes before applying one.
_MACHADO: dict[Deficiency, tuple[tuple[float, ...], ...]] = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}

# CIE standard illuminant D65, the white point sRGB is defined against.
_WHITE = (0.95047, 1.0, 1.08883)


def _linearize(channel: int) -> float:
    """Undo the sRGB transfer function on one 0-255 channel."""
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _encode(value: float) -> int:
    """Re-apply the sRGB transfer function, returning a 0-255 channel."""
    clamped = min(1.0, max(0.0, value))
    if clamped <= 0.0031308:
        encoded = 12.92 * clamped
    else:
        encoded = 1.055 * clamped ** (1 / 2.4) - 0.055
    return round(encoded * 255.0)


def simulate(color: ColorLike, deficiency: Deficiency | None) -> Color:
    """Render ``color`` as a viewer with ``deficiency`` sees it.

    Applies the Machado (2009) severity-1.0 matrix in linear RGB, then
    re-encodes to sRGB. Alpha is carried through untouched — a deficiency
    changes which colors are confusable, not how transparent they are.

    Args:
        color: The color to simulate (any :data:`ColorLike`).
        deficiency: Which dichromacy to simulate, or ``None`` for normal vision
            (which returns the color unchanged, so callers can loop over
            :data:`VISION` without special-casing).

    Returns:
        The simulated `Color`.

    Examples:
        >>> simulate("#ff0000", None)
        Color(r=255, g=0, b=0, a=255)

        A pure red loses most of its redness to a protanope, landing on a dark
        yellow-brown:

        >>> simulate("#ff0000", "protanopia")
        Color(r=109, g=95, b=0, a=255)

        Grays are unaffected, since every model preserves the achromatic axis:

        >>> simulate("#808080", "tritanopia")
        Color(r=128, g=128, b=128, a=255)

    """
    parsed = Color.parse(color)
    if deficiency is None:
        return parsed
    try:
        matrix = _MACHADO[deficiency]
    except KeyError:
        names = ", ".join(DEFICIENCIES)
        raise ValueError(
            f"unknown deficiency {deficiency!r}; choose one of: {names}"
        ) from None
    r, g, b = (_linearize(c) for c in parsed.rgb)
    red, green, blue = (
        _encode(row[0] * r + row[1] * g + row[2] * b) for row in matrix
    )
    return Color(red, green, blue, parsed.a)


def to_lab(color: ColorLike) -> Lab:
    """Convert ``color`` to CIELAB (D65), the space `delta_e` measures in.

    Args:
        color: The color to convert (any :data:`ColorLike`). Alpha is ignored:
            CIELAB has no opacity axis, and an annotation's alpha is a style
            choice rather than part of its identity.

    Returns:
        The ``(L*, a*, b*)`` triple.

    Examples:
        >>> round(to_lab("#ffffff")[0])
        100
        >>> round(to_lab("#000000")[0])
        0
        >>> round(to_lab("#808080")[0], 1)  # mid gray is *not* L* 50
        53.6

    """
    r, g, b = (_linearize(c) for c in Color.parse(color).rgb)
    xyz = (
        0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
        0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
        0.0193339 * r + 0.1191920 * g + 0.9503041 * b,
    )
    fx, fy, fz = (_lab_f(v / w) for v, w in zip(xyz, _WHITE, strict=True))
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def _lab_f(t: float) -> float:
    """Apply the CIELAB compression curve, linear near black to stay finite."""
    if t > 216 / 24389:
        return t ** (1 / 3)
    return t * (841 / 108) + 4 / 29


def delta_e(a: ColorLike, b: ColorLike) -> float:
    """Return the CIEDE2000 color difference between ``a`` and ``b``.

    CIEDE2000 rather than plain Euclidean CIELAB distance because the latter
    badly overstates differences among saturated blues — exactly the region
    every colorblind-safe palette crowds into, which would let a palette pass a
    separation check it does not deserve.

    Args:
        a: The first color (any :data:`ColorLike`).
        b: The second color.

    Returns:
        The distance, ``0.0`` for identical colors and around ``1.0`` at the
        threshold where a viewer can tell two adjacent patches apart.

    Examples:
        >>> delta_e("#ff0000", "#ff0000")
        0.0
        >>> round(delta_e("#000000", "#ffffff"), 1)
        100.0
        >>> round(delta_e("#4477aa", "#ee6677"), 1)  # two Tol-bright colors
        43.0

    """
    return _ciede2000(to_lab(a), to_lab(b))


def _ciede2000(first: Lab, second: Lab) -> float:
    """CIEDE2000 between two CIELAB triples, with the standard unit weights.

    Follows Sharma, Wu & Dalal (2005), whose published test data the unit tests
    check this against — the formula has enough branches and wrap-arounds to be
    worth pinning down that way.
    """
    l1, a1, b1 = first
    l2, a2, b2 = second

    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar7 = ((c1 + c2) / 2.0) ** 7
    # Scale a* outward for near-neutral colors, so the formula stops treating
    # a tiny chroma difference between two grays as a big hue rotation.
    g = 0.5 * (1.0 - math.sqrt(c_bar7 / (c_bar7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0 if (b1 or a1p) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0 if (b2 or a2p) else 0.0

    delta_l = l2 - l1
    delta_c = c2p - c1p
    if c1p * c2p == 0.0:
        delta_h = 0.0
        h_bar = h1p + h2p
    else:
        delta_h = (h2p - h1p + 180.0) % 360.0 - 180.0
        h_bar = (h1p + h2p) / 2.0
        if abs(h1p - h2p) > 180.0:
            h_bar += 180.0 if h1p + h2p < 360.0 else -180.0
    delta_hp = 2.0 * math.sqrt(c1p * c2p) * math.sin(math.radians(delta_h / 2))

    l_bar, c_bar = (l1 + l2) / 2.0, (c1p + c2p) / 2.0
    t = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar))
        + 0.32 * math.cos(math.radians(3.0 * h_bar + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar - 63.0))
    )
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / math.sqrt(
        20.0 + (l_bar - 50.0) ** 2
    )
    s_c = 1.0 + 0.045 * c_bar
    s_h = 1.0 + 0.015 * c_bar * t
    # The rotation term pulls the blue/violet region together, where hue
    # differences read as much smaller than the raw geometry suggests.
    rotation = (
        -2.0
        * math.sqrt(c_bar**7 / (c_bar**7 + 25.0**7))
        * math.sin(
            math.radians(60.0 * math.exp(-(((h_bar - 275.0) / 25.0) ** 2)))
        )
    )
    term_l, term_c = delta_l / s_l, delta_c / s_c
    term_h = delta_hp / s_h
    return math.sqrt(
        term_l**2 + term_c**2 + term_h**2 + rotation * term_c * term_h
    )


def min_separation(
    colors: Iterable[ColorLike],
    vision: Sequence[Deficiency | None] = VISION,
) -> float:
    """Return the smallest distance between any two ``colors``, worst vision first.

    The one number that says whether a palette works: every pair of colors is
    compared under every requested vision type, and the closest pairing wins.
    A palette whose worst case is comfortably above ~10 stays readable for
    every viewer; one in the low single digits has a pair that some viewers
    simply cannot separate.

    Args:
        colors: The palette to check. Fewer than two colors has no pair to
            measure and returns infinity.
        vision: Which vision types to simulate. Defaults to :data:`VISION`
            (normal plus all three deficiencies), i.e. the worst case; pass
            ``[None]`` to measure normal vision alone, or a single deficiency
            to isolate it.

    Returns:
        The minimum CIEDE2000 distance, or ``math.inf`` for fewer than two
        colors.

    Examples:
        >>> from luxonis_ml.utils.color.cvd import min_separation
        >>> round(min_separation(["#000000", "#ffffff"]))
        100

        A red/green pair looks fine to normal vision and falls apart under
        simulation — the whole reason this function defaults to the worst case:

        >>> pair = ["#d62728", "#2ca02c"]
        >>> round(min_separation(pair, [None]))
        72
        >>> round(min_separation(pair))
        5

    """
    labs = [
        [to_lab(simulate(color, deficiency)) for deficiency in vision]
        for color in colors
    ]
    worst = math.inf
    for i, first in enumerate(labs):
        for second in labs[i + 1 :]:
            for lab_a, lab_b in zip(first, second, strict=True):
                worst = min(worst, _ciede2000(lab_a, lab_b))
    return worst


def _vision_labs(color: Color) -> tuple[Lab, ...]:
    """Return ``color``'s CIELAB coordinates under every vision type.

    The unit `CVDDistinctColors` searches in: two colors are safely distinct
    only when they stay apart in *all* of these at once.
    """
    return tuple(to_lab(simulate(color, d)) for d in VISION)


def _worst_distance(first: Sequence[Lab], second: Sequence[Lab]) -> float:
    """Return the smallest CIEDE2000 distance across paired vision types."""
    return min(
        _ciede2000(lab_a, lab_b)
        for lab_a, lab_b in zip(first, second, strict=True)
    )
