"""Color gradients (colormaps) for scalar fields such as heatmaps.

A `Gradient` maps a scalar in ``[0, 1]`` to a `Color` by linearly interpolating
between ordered color stops. It is the color model for heatmaps, kept separate
from the class `Palette` because a heatmap colors a continuous magnitude, not a
set of discrete classes.

A handful of ready-made gradients are registered by name in :data:`GRADIENTS`
(perceptually-uniform ones like ``"viridis"``/``"turbo"`` plus classics like
``"jet"`` and ``"hot"``). Resolve a name or pass your own with
`resolve_gradient`; build a custom one with `Gradient.from_colors`.

Only `Gradient.colorize` needs NumPy, and it imports it when called, so this
module stays importable on a base ``luxonis-ml`` install that has no NumPy.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import Color, ColorLike

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True)
class Gradient:
    """A colormap: an ordered set of color stops interpolated in RGB.

    A scalar ``t`` in ``[0, 1]`` is mapped to a color by piecewise-linear
    interpolation between the surrounding stops. Values outside ``[0, 1]`` are
    clamped.

    Attributes:
        stops: Ascending ``(position, color)`` pairs with positions in ``[0, 1]``;
            the first should be at ``0.0`` and the last at ``1.0``.

    Examples:
        >>> g = Gradient.from_colors(["#000000", "#ff0000"])
        >>> g.color_at(0.0)
        Color(r=0, g=0, b=0, a=255)
        >>> g.color_at(0.5)
        Color(r=128, g=0, b=0, a=255)
        >>> g.color_at(1.0)
        Color(r=255, g=0, b=0, a=255)

    """

    stops: tuple[tuple[float, Color], ...]

    @classmethod
    def from_colors(
        cls,
        colors: "list[ColorLike]",
        *,
        positions: "list[float] | None" = None,
    ) -> "Gradient":
        """Build a gradient from colors, evenly spaced unless positions are given.

        Args:
            colors: Two or more colors (hex strings, tuples, or `Color`), from the
                low end of the scale to the high end.
            positions: Optional stop positions in ``[0, 1]``, one per color; when
                ``None`` the colors are spread evenly from ``0`` to ``1``.

        Returns:
            The `Gradient`.

        Raises:
            ValueError: If fewer than two colors are given, or ``positions`` is
                given with a different length than ``colors``, a position outside
                ``[0, 1]``, or a repeated position.

        """
        parsed = [Color.parse(c) for c in colors]
        if len(parsed) < 2:
            raise ValueError("a gradient needs at least two colors")
        if positions is None:
            last = len(parsed) - 1
            positions = [i / last for i in range(len(parsed))]
        elif len(positions) != len(parsed):
            raise ValueError("positions must have one entry per color")

        outside = [p for p in positions if not 0.0 <= p <= 1.0]
        if outside:
            raise ValueError(f"positions must lie in [0, 1], got {outside}")
        # Two stops at one position leave the color between them undefined.
        if len(set(positions)) != len(positions):
            raise ValueError(f"positions must be distinct, got {positions}")
        stops = sorted(
            ((float(p), c) for p, c in zip(positions, parsed, strict=True)),
            key=lambda stop: stop[0],
        )
        return cls(tuple(stops))

    def color_at(self, t: float) -> Color:
        """Return the color at scalar ``t`` in ``[0, 1]`` (clamped).

        All four channels are interpolated, so a gradient between translucent
        stops fades in alpha too.

        Args:
            t: The position along the gradient.

        Returns:
            The interpolated `Color`.

        """
        clamped = min(1.0, max(0.0, t))
        if clamped <= self.stops[0][0]:
            return self.stops[0][1]
        for (p0, low), (p1, high) in zip(
            self.stops, self.stops[1:], strict=False
        ):
            if clamped <= p1:
                # Slope first, then offset — the order ``np.interp`` uses, so a
                # scalar lookup rounds exactly like the same value in `colorize`.
                span, offset = p1 - p0, clamped - p0
                return Color(
                    round(low.r + (high.r - low.r) / span * offset),
                    round(low.g + (high.g - low.g) / span * offset),
                    round(low.b + (high.b - low.b) / span * offset),
                    round(low.a + (high.a - low.a) / span * offset),
                )
        return self.stops[-1][1]

    def colorize(self, field: "npt.ArrayLike") -> "np.ndarray":
        """Map a scalar field to RGB, interpolating each channel through the stops.

        Only the color channels are mapped: the result is opaque RGB, so any
        alpha on the stops is ignored. Use `color_at` when alpha matters.

        Args:
            field: An array of scalars, expected in ``[0, 1]`` (values are
                clamped). Any shape is accepted, as is anything
                ``numpy.asarray`` converts to a float array.

        Returns:
            A ``uint8`` array shaped ``(*field.shape, 3)`` in RGB order.

        """
        import numpy as np

        values = np.asarray(field, dtype=np.float64)
        flat = np.clip(values.ravel(), 0.0, 1.0)
        xp = [p for p, _ in self.stops]
        rgb = np.stack(
            [
                np.interp(flat, xp, [c.r for _, c in self.stops]),
                np.interp(flat, xp, [c.g for _, c in self.stops]),
                np.interp(flat, xp, [c.b for _, c in self.stops]),
            ],
            axis=-1,
        )
        return np.round(rgb).astype(np.uint8).reshape(*values.shape, 3)


# Preset colormaps. The perceptually-uniform families (viridis, magma, inferno,
# plasma, turbo) are approximated with a handful of anchor colors sampled from
# the matplotlib originals — close enough for visualization, without shipping a
# 256-entry lookup table. Classics (jet, hot, coolwarm, grayscale) round it out.
_PRESETS: dict[str, tuple[str, ...]] = {
    "viridis": (
        "#440154",
        "#414487",
        "#2a788e",
        "#22a884",
        "#7ad151",
        "#fde725",
    ),
    "magma": (
        "#000004",
        "#180f3e",
        "#451077",
        "#721f81",
        "#9f2f7f",
        "#cd4071",
        "#f1605d",
        "#fd9567",
        "#feca8d",
        "#fcfdbf",
    ),
    "inferno": (
        "#000004",
        "#320a5a",
        "#781c6d",
        "#bb3754",
        "#ed6925",
        "#fbb61a",
        "#fcffa4",
    ),
    "plasma": (
        "#0d0887",
        "#5402a3",
        "#8b0aa5",
        "#b83289",
        "#db5c68",
        "#f48849",
        "#febd2a",
        "#f0f921",
    ),
    "turbo": (
        "#30123b",
        "#4145ab",
        "#4675ed",
        "#39a2fc",
        "#1bcfd4",
        "#24eca6",
        "#61fc6c",
        "#a4fc3b",
        "#d1e834",
        "#f3c53a",
        "#fe9b2d",
        "#ec7014",
        "#cb4149",
        "#7a0403",
    ),
    "jet": ("#000080", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#800000"),
    "hot": ("#000000", "#ff0000", "#ffff00", "#ffffff"),
    "coolwarm": ("#3b4cc0", "#6f91f2", "#dddcdc", "#f6a385", "#b40426"),
    "rdbu": ("#67001f", "#d6604d", "#f7f7f7", "#4393c3", "#053061"),
    "seismic": ("#00004c", "#0000ff", "#ffffff", "#ff0000", "#800000"),
    "grayscale": ("#000000", "#ffffff"),
}

GRADIENTS: dict[str, Gradient] = {
    name: Gradient.from_colors(list(colors))
    for name, colors in _PRESETS.items()
}
"""The preset gradients, keyed by name. See `resolve_gradient`."""

DEFAULT_GRADIENT = "turbo"
"""Name of the gradient used for an unsigned field when none is given."""

DIVERGING_GRADIENTS = frozenset({"coolwarm", "rdbu", "seismic"})
"""Presets built around a neutral midpoint.

Each has an odd number of evenly spaced stops with a near-white one in the
middle, so centering a heatmap puts that neutral color exactly on the center
value. The sequential presets have no such anchor: centering them
still balances the range, but no particular color marks the middle.
"""

DEFAULT_DIVERGING_GRADIENT = "coolwarm"
"""Name of the gradient used for a signed field when none is given."""


def resolve_gradient(gradient: "Gradient | str") -> Gradient:
    """Resolve a gradient or a preset name to a `Gradient`.

    Args:
        gradient: A `Gradient`, or the name of a preset in :data:`GRADIENTS`.

    Returns:
        The `Gradient`.

    Raises:
        KeyError: If a name is given that is not a registered preset.

    Examples:
        >>> resolve_gradient("grayscale").color_at(1.0)
        Color(r=255, g=255, b=255, a=255)

    """
    if isinstance(gradient, Gradient):
        return gradient
    try:
        return GRADIENTS[gradient]
    except KeyError:
        names = ", ".join(sorted(GRADIENTS))
        raise KeyError(
            f"unknown gradient {gradient!r}; choose one of: {names}, "
            "or pass a Gradient"
        ) from None
