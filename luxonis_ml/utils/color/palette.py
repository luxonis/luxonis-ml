"""Distinct class colors from a pluggable, index-based generator.

A `Palette` hands each new class the next color from a **color generator**:
a callable ``int -> Color`` that maps a sequence position to a color. The default
generator, `GoldenRatioColors`, offsets each hue from the last by the golden
angle (~137.5°) so no two classes land on a near-identical hue — the failure mode a
fixed palette (or a hash into one) can't avoid once the class count approaches the
palette size. Class labels stay on this distinct-hue scheme on purpose; the
Luxonis brand colors are reserved for chrome (see
`luxonis_ml.utils.color.brand`), though `SequenceColors` lets a caller anchor a
palette to the brand :data:`BRAND_COLORS` when they want to.

Swapping the whole color scheme is a single argument — ``Palette(generator=...)`` —
so the strategy lives in exactly one place and callers (``BBox`` and friends) only
ever touch `Palette.color_for`.

Because the spacing guarantee only holds for *sequential* indices, colors are
assigned in order of first appearance and memoized: within a process, a class keeps
its color across every image (the module-level :data:`DEFAULT_PALETTE` is shared).
The trade-off is that a class's color depends on the order classes are first seen,
not on its name alone — stable within a run and across runs that request classes in
the same order, but not across arbitrary reorderings. Pin a color explicitly with
``BBox(color=...)`` or by pre-registering classes in a fixed order when a name must
map to one exact color forever.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from .base import Color, ColorLike
from .brand import AMBER, GREEN, MINT, ORANGE, PERIWINKLE, PURPLE, RED, SALMON

ColorGenerator = Callable[[int], Color]
"""A callable mapping a zero-based sequence index to a `Color`."""

# The Luxonis brand colors in a distinct-hue order: the four solid hues first,
# then their lighter "decoration" variants, interleaving hues so adjacent
# entries stay easy to tell apart. Reserved for callers that explicitly want a
# brand-anchored palette (via `SequenceColors`); the default palette does not
# use them (see the module docstring).
BRAND_COLORS: tuple[Color, ...] = (
    PURPLE,
    GREEN,
    ORANGE,
    RED,
    PERIWINKLE,
    MINT,
    AMBER,
    SALMON,
)
"""The Luxonis brand color sequence, for use with `SequenceColors`."""

# (sqrt(5) - 1) / 2 — the golden ratio's conjugate. Stepping the hue by this each
# time is the classic low-discrepancy way to spread points around a circle.
_GOLDEN_CONJUGATE = 0.6180339887498949

# A second, unrelated irrational step used to vary lightness, so that even the rare
# pair of hue-neighbors (at large class counts) still differ in brightness.
_LIGHTNESS_STEP = 0.7548776662466927


@dataclass(frozen=True)
class GoldenRatioColors:
    """Maps a sequence index to a distinct color via golden-ratio hue spacing.

    This is the default :data:`ColorGenerator`. Every knob that shapes the look
    lives here, so restyling the whole palette means constructing one of these (or
    any other ``int -> Color`` callable) and passing it to `Palette`.

    Attributes:
        hue0: Hue of index 0, in ``[0, 1]`` turns (default a blue).
        saturation: Fixed saturation of every color, in ``[0, 1]``.
        lightness: Base lightness of every color, in ``[0, 1]``.
        lightness_jitter: Peak lightness deviation from ``lightness``, applied via a
            second low-discrepancy sequence so brightness varies subtly (which keeps
            colors apart even when hues get crowded at high class counts).

    Examples:
        >>> gen = GoldenRatioColors()
        >>> gen(0) == gen(0)
        True
        >>> gen(0) == gen(1)
        False

    """

    hue0: float = 0.6
    saturation: float = 0.7
    lightness: float = 0.62
    lightness_jitter: float = 0.12

    def __call__(self, index: int) -> Color:
        """Generate the color for sequence position ``index``.

        Args:
            index: The zero-based position in the sequence.

        Returns:
            The generated `Color`.

        """
        hue = (self.hue0 + index * _GOLDEN_CONJUGATE) % 1.0
        jitter = 2.0 * ((index * _LIGHTNESS_STEP) % 1.0) - 1.0
        lightness = self.lightness + self.lightness_jitter * jitter
        return Color.from_hls(hue, lightness, self.saturation)


@dataclass(frozen=True)
class SequenceColors:
    """Hand out a fixed list of colors first, then fall back to a generator.

    Wraps a list of ``anchors`` (e.g. the Luxonis :data:`BRAND_COLORS`) so the
    first classes seen get those exact colors, in order. Once the anchors run
    out, index ``n`` continues through ``overflow`` (a `GoldenRatioColors`
    generator by default, queried with an index that restarts at the first
    post-anchor class), so any number of classes still receive distinct colors.

    Attributes:
        anchors: Colors handed out for indices ``0 .. len(anchors) - 1``.
        overflow: Generator used once the anchors are exhausted.

    Examples:
        >>> from luxonis_ml.vizlab.color import Color
        >>> gen = SequenceColors((Color(255, 0, 0), Color(0, 255, 0)))
        >>> gen(0)
        Color(r=255, g=0, b=0, a=255)
        >>> gen(1)
        Color(r=0, g=255, b=0, a=255)
        >>> gen(2) == gen(2)  # overflow is deterministic
        True

    """

    anchors: tuple[Color, ...]
    overflow: ColorGenerator = field(default_factory=GoldenRatioColors)

    def __call__(self, index: int) -> Color:
        """Return the color for sequence position ``index``.

        Args:
            index: The zero-based position in the sequence.

        Returns:
            The anchor color at ``index``, or the overflow generator's color for
            the position past the anchors.

        """
        if 0 <= index < len(self.anchors):
            return self.anchors[index]
        return self.overflow(index - len(self.anchors))


class Palette:
    """Assigns colors to classes in order of first use, from a color generator.

    Examples:
        A class keeps its color; different classes get different colors:

        >>> p = Palette()
        >>> p.color_for("car") == p.color_for("car")
        True
        >>> p.color_for("car") == p.color_for("bus")
        False
        >>> len(p)
        2

        Two fresh palettes agree as long as classes are requested in the same order
        (``"car"`` is index 0 in both) ...

        >>> Palette().color_for("car") == Palette().color_for("car")
        True

        ... but the color follows first-seen order, not the name, so reordering
        changes it:

        >>> Palette(["a", "b"]).color_for("b") == Palette(
        ...     ["b", "a"]
        ... ).color_for("b")
        False

    """

    def __init__(
        self,
        classes: Iterable[str] | None = None,
        *,
        generator: ColorGenerator | None = None,
        colors: Mapping[str, ColorLike] | None = None,
    ) -> None:
        """Create a palette, optionally pinning class order and exact colors.

        Args:
            classes: Class names to register up front, in the order that fixes their
                colors. Any class not listed is assigned on first use.
            generator: The ``int -> Color`` strategy; defaults to
                `GoldenRatioColors`.
            colors: Explicit ``{class_name: color}`` pins. A pinned class always
                gets exactly that color and never consumes a generator slot, so
                pins do not shift the sequence colors of the other classes.

        """
        self._generator: ColorGenerator = generator or GoldenRatioColors()
        self._colors: dict[str, Color] = {}
        self._pins: dict[str, Color] = {
            name: Color.parse(value) for name, value in (colors or {}).items()
        }
        if classes is not None:
            for name in classes:
                self.color_for(name)

    def __len__(self) -> int:
        """Return the number of classes assigned a color so far."""
        return len(self._colors)

    def at(self, index: int) -> Color:
        """Return the color for sequence position ``index`` (without registering it).

        Args:
            index: The zero-based position in the generator's sequence.

        Returns:
            The generated `Color`.

        """
        return self._generator(index)

    def color_for(self, key: str) -> Color:
        """Return the color assigned to ``key``, assigning a new one if unseen.

        The first time a key is seen it takes the next color in the sequence; every
        later call for that key returns the same color.

        Args:
            key: The label (or any string identity) to color.

        Returns:
            The assigned `Color`.

        """
        pinned = self._pins.get(key)
        if pinned is not None:
            return pinned
        color = self._colors.get(key)
        if color is None:
            color = self._generator(len(self._colors))
            self._colors[key] = color
        return color

    def pin(self, key: str, color: ColorLike) -> "Palette":
        """Pin ``key`` to exactly ``color`` and return ``self`` for chaining.

        A pinned class always gets this color from `color_for`, overriding any
        generated one, and never consumes a generator slot.

        Args:
            key: The class name (or string identity) to pin.
            color: The exact color for ``key`` (any :data:`ColorLike`).

        Returns:
            This palette, to allow fluent chaining.

        """
        self._pins[key] = Color.parse(color)
        return self

    def with_colors(self, colors: Mapping[str, ColorLike]) -> "Palette":
        """Return a copy of this palette with additional class-color pins.

        The copy shares the generator and inherits already-assigned colors and
        pins; ``colors`` are merged on top. The original is not mutated, so a
        shared palette can be specialized per render without side effects.

        Args:
            colors: Extra ``{class_name: color}`` pins to add.

        Returns:
            A new `Palette` with the merged pins.

        """
        clone = Palette(generator=self._generator)
        clone._colors = dict(self._colors)
        clone._pins = {
            **self._pins,
            **{name: Color.parse(value) for name, value in colors.items()},
        }
        return clone


DEFAULT_PALETTE = Palette()
"""The process-wide default `Palette`, shared so a class keeps its color.

Uses the distinct-hue `GoldenRatioColors` generator: label colors are chosen to
stay far apart, not to match the brand (which is reserved for chrome)."""
