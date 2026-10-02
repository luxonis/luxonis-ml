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
so the strategy lives in exactly one place and callers only ever touch
`Palette.color_for`. Besides the default, three strategies ship:

- A **named colorblind-safe set**: ``Palette(generator="okabe-ito")`` resolves one
  of the published qualitative palettes in :data:`PALETTES`, whose colors are
  chosen to stay apart for viewers with color-vision deficiency — something
  golden-ratio hue spacing does *not* guarantee, since two hues far apart on the
  wheel can collapse onto each other under simulation. Each set is short, so
  `CVDDistinctColors` takes over past its last color.
- `CVDDistinctColors` on its own (``Palette(generator="cvd")``): an unbounded
  generator that picks each color to maximize its distance from every color
  already handed out, measured under normal vision *and* all three dichromacies
  (see `luxonis_ml.utils.color.cvd`).
- `Palette.from_colormap`, which samples a categorical palette out of a
  continuous colormap from :data:`luxonis_ml.utils.color.gradient.GRADIENTS`.

Because the spacing guarantee only holds for *sequential* indices, colors are
assigned in order of first appearance and memoized: within a process, a class keeps
its color across every image (the module-level :data:`DEFAULT_PALETTE` is shared).
The trade-off is that a class's color depends on the order classes are first seen,
not on its name alone — stable within a run and across runs that request classes in
the same order, but not across arbitrary reorderings. Pin a color explicitly with
`Palette.pin` or by pre-registering classes in a fixed order when a name must map
to one exact color forever.
"""

import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, lru_cache

from .base import Color, ColorLike
from .brand import AMBER, GREEN, MINT, ORANGE, PERIWINKLE, PURPLE, RED, SALMON
from .cvd import Lab, _vision_labs, _worst_distance
from .gradient import Gradient, resolve_gradient

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
        >>> from luxonis_ml.utils.color import Color
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


# The published colorblind-safe qualitative palettes, as hex. Two deliberate
# departures from the sources, both because these color *annotations* rather
# than chart series: Okabe-Ito's black is dropped (a black outline vanishes
# against dark imagery and against the dark theme's navy), and Tol's grey is
# dropped from every set that has one (he reserves it for "bad data", and a grey
# box reads as unlabeled rather than as its own class). Pin them back with
# `Palette.pin` if you need a source-exact set.
_NAMED_PALETTES: dict[str, tuple[str, ...]] = {
    # Okabe & Ito, "Color Universal Design" (2008).
    "okabe-ito": (
        "#e69f00",  # orange
        "#56b4e9",  # sky blue
        "#009e73",  # bluish green
        "#f0e442",  # yellow
        "#0072b2",  # blue
        "#d55e00",  # vermilion
        "#cc79a7",  # reddish purple
    ),
    # Paul Tol, "Colour Schemes" (SRON/EPS-TN-2009-003).
    "tol-bright": (
        "#4477aa",
        "#ee6677",
        "#228833",
        "#ccbb44",
        "#66ccee",
        "#aa3377",
    ),
    "tol-high-contrast": ("#004488", "#ddaa33", "#bb5566"),
    "tol-vibrant": (
        "#ee7733",
        "#0077bb",
        "#33bbee",
        "#ee3377",
        "#cc3311",
        "#009988",
    ),
    "tol-muted": (
        "#cc6677",
        "#332288",
        "#ddcc77",
        "#117733",
        "#88ccee",
        "#882255",
        "#44aa99",
        "#999933",
        "#aa4499",
    ),
}

PALETTES: dict[str, tuple[Color, ...]] = {
    name: tuple(Color.parse(hex_code) for hex_code in colors)
    for name, colors in _NAMED_PALETTES.items()
}
"""The named colorblind-safe qualitative palettes, keyed by name.

Every set here is checked by the test suite to keep a minimum CIEDE2000
separation between all of its colors under normal vision and under simulated
protanopia, deuteranopia, and tritanopia — the property that makes them worth
shipping over a prettier arbitrary set. They are short by nature (three to nine
colors); `resolve_generator` extends each one with `CVDDistinctColors`, which
carries the *intent* past the published length but not the hand-tuning.

Pass a name wherever a :data:`ColorGenerator` is accepted::

    Palette(generator="okabe-ito")
    DARK_THEME.with_palette("tol-muted")
"""

# The pool `CVDDistinctColors` selects from. A regular HSL lattice rather than a
# continuous search: it keeps the choice deterministic and the per-color cost a
# fixed number of distance evaluations, and it bounds the result to colors that
# actually work as annotations — nothing so dark it reads as an outline shadow,
# nothing so pale it disappears on a bright photo, nothing so desaturated it
# reads as chrome.
_CANDIDATE_HUES = 36
_CANDIDATE_LIGHTNESS = (0.40, 0.52, 0.64, 0.76)
_CANDIDATE_SATURATION = (0.55, 0.85)


@lru_cache(maxsize=1)
def _candidates() -> tuple[tuple[Color, tuple[Lab, ...]], ...]:
    """Return the candidate colors, each with its CIELAB under every vision type.

    Built once on first use and cached: simulating the whole lattice costs a few
    milliseconds, which is not worth paying at import time for the majority of
    processes that never ask for a colorblind-safe palette.
    """
    pool = [
        Color.from_hls(hue / _CANDIDATE_HUES, lightness, saturation)
        for hue in range(_CANDIDATE_HUES)
        for lightness in _CANDIDATE_LIGHTNESS
        for saturation in _CANDIDATE_SATURATION
    ]
    return tuple((color, _vision_labs(color)) for color in pool)


class CVDDistinctColors:
    """Generates colors that stay distinct under simulated color blindness.

    A farthest-point search: each index takes the candidate color whose
    *smallest* CIEDE2000 distance to every color already handed out is the
    largest — where "distance" is itself the smallest distance across normal
    vision and simulated protanopia, deuteranopia, and tritanopia. A candidate
    that would collide with an existing class for any one of those viewers is
    therefore scored as if it collided for all of them, and loses.

    This is the answer to the problem every fixed colorblind-safe palette has:
    Okabe-Ito gives seven colors and COCO has eighty. What this generator
    guarantees, precisely:

    - **No repeats.** Picking a color already in use scores zero, so it is never
      picked while anything else is available.
    - **Best-available spacing, not a fixed floor.** The ``n``-th color is the
      most separable one the lattice still offers, and that separation falls
      monotonically as ``n`` grows: the measured worst-case CIEDE2000 is ~13 at
      eight colors, ~7 at twenty, and ~3 at eighty. So the colors stay
      *obviously* different for about a dozen classes, merely distinguishable
      out to forty, and past that the only honest claim left is that no two are
      the same. Repeats begin only when the 288-entry lattice runs out.

    Which is to say: this is a real guarantee for tens of classes, and for
    hundreds the answer is not a cleverer generator but a second visual channel
    — the label text drawn beside each annotation.

    Because index ``n``'s color depends on indices ``0 .. n-1``, the sequence is
    computed incrementally and memoized. It is still a pure function of
    ``avoid``, so two generators built the same way agree.

    Attributes:
        avoid: Colors that are already spoken for — typically the anchors of the
            named palette this generator is extending, so the first generated
            color does not land on top of one of them.

    Examples:
        >>> gen = CVDDistinctColors()
        >>> gen(0) == gen(0)  # memoized, so stable
        True
        >>> from luxonis_ml.utils.color.cvd import min_separation
        >>> colors = [gen(i) for i in range(8)]
        >>> round(min_separation(colors))  # worst pair, worst vision type
        13

        For comparison, the default `GoldenRatioColors` spreads eight hues
        beautifully for normal vision and puts two of them on top of each other
        for a deuteranope:

        >>> golden = [GoldenRatioColors()(i) for i in range(8)]
        >>> round(min_separation(golden, [None]))
        11
        >>> round(min_separation(golden))
        1

        Seeding it with colors to avoid steers it away from them:

        >>> from luxonis_ml.utils.color.cvd import delta_e
        >>> seeded = CVDDistinctColors(avoid=["#e69f00"])
        >>> delta_e(seeded(0), "#e69f00") > 40
        True

    """

    def __init__(self, avoid: Iterable[ColorLike] = ()) -> None:
        """Create a generator that avoids ``avoid`` and everything it yields.

        Args:
            avoid: Colors already in use, kept out of the generated sequence.

        """
        self.avoid = tuple(Color.parse(color) for color in avoid)
        self._chosen: list[Color] = []
        self._scores: list[float] | None = None
        # `DEFAULT_PALETTE`-style sharing means two threads can extend the
        # sequence at once; the search is a read-modify-write of `_scores`.
        self._lock = threading.Lock()

    def __call__(self, index: int) -> Color:
        """Return the color for sequence position ``index``.

        Args:
            index: The zero-based position in the sequence. Positions up to
                ``index`` are computed if they have not been already, so
                jumping straight to a high index costs the whole prefix.

        Returns:
            The generated `Color`.

        """
        with self._lock:
            while len(self._chosen) <= index:
                self._chosen.append(self._pick())
            return self._chosen[index]

    def _pick(self) -> Color:
        """Take the candidate currently farthest from everything already used."""
        candidates = _candidates()
        if self._scores is None:
            self._scores = [math.inf] * len(candidates)
            for color in self.avoid:
                self._exclude(_vision_labs(color))
        scores = self._scores
        best = max(range(len(candidates)), key=scores.__getitem__)
        color, labs = candidates[best]
        self._exclude(labs)
        return color

    def _exclude(self, labs: Sequence[Lab]) -> None:
        """Fold a newly used color into every candidate's running score."""
        scores = self._scores
        assert scores is not None
        for i, (_, candidate) in enumerate(_candidates()):
            distance = _worst_distance(candidate, labs)
            scores[i] = min(scores[i], distance)


# Bit-reversal positions in [0, 1]: 0, 1, 1/2, 1/4, 3/4, 1/8, 5/8, ... Every
# index has one fixed position, yet any prefix is spread evenly over the whole
# colormap — for prefix lengths of 2^k + 1 it is *exactly* the evenly spaced
# grid. That is what lets `ColormapColors` sample lazily, as classes appear,
# without ever having to know how many classes there will be.
def _radical_inverse(index: int) -> float:
    """Return the base-2 radical inverse of ``index`` (van der Corput)."""
    result, denominator = 0.0, 1.0
    while index:
        denominator *= 2.0
        index, bit = divmod(index, 2)
        result += bit / denominator
    return result


@dataclass(frozen=True)
class ColormapColors:
    """Draws categorical colors out of a continuous colormap.

    A colormap is a one-dimensional path through color space, so unlike a
    qualitative palette it cannot keep an unbounded number of classes apart:
    every class added crowds the ones already on the path. What it buys instead
    is *order* — with severity levels, size buckets, or depth bands the reader
    can see the ranking in the colors, which a qualitative palette deliberately
    hides.

    Prefer a perceptually uniform sequential map (``"viridis"``, ``"magma"``,
    ``"inferno"``): the ordering they encode is mostly lightness, which no
    color-vision deficiency touches, so they survive simulation and reproduce in
    grayscale. Past roughly eight classes reach for one of the qualitative
    :data:`PALETTES` instead — by then the ordering is too fine to read anyway,
    and the colors measure closer together than a hand-tuned set.

    Attributes:
        gradient: The colormap to sample.
        count: How many colors to divide the colormap into, or ``None`` to
            sample lazily and without limit. See `Palette.from_colormap` for
            what each mode costs.

    Examples:
        >>> from luxonis_ml.utils.color.gradient import GRADIENTS
        >>> gen = ColormapColors(GRADIENTS["grayscale"], count=3)
        >>> [color.r for color in (gen(0), gen(1), gen(2))]
        [0, 128, 255]
        >>> gen(3).r  # a fixed count wraps around
        0

        Unbounded, the same colormap is sampled at bit-reversal positions, so
        each index keeps its color no matter how many classes turn up later:

        >>> lazy = ColormapColors(GRADIENTS["grayscale"])
        >>> [lazy(i).r for i in range(5)]
        [0, 255, 128, 64, 191]

    """

    gradient: Gradient
    count: int | None = None

    def __call__(self, index: int) -> Color:
        """Return the color for sequence position ``index``.

        Args:
            index: The zero-based position in the sequence.

        Returns:
            The sampled `Color`.

        """
        if self.count is None:
            return self.gradient.color_at(_radical_inverse_position(index))
        if self.count == 1:
            return self.gradient.color_at(0.0)
        return self.gradient.color_at((index % self.count) / (self.count - 1))


def _radical_inverse_position(index: int) -> float:
    """Return the ``index``-th low-discrepancy position, endpoints first."""
    if index < 2:
        return float(index)
    return _radical_inverse(index - 1)


#: The name that selects `CVDDistinctColors` with nothing in front of it. Not a
#: :data:`PALETTES` entry because it is not a fixed list of colors — it is the
#: search those lists are extended with, run from the first color.
CVD_PALETTE = "cvd"


@cache
def _named_generator(name: str) -> ColorGenerator:
    """Build (once) the generator behind a palette name.

    Cached so every caller of a given name shares one generator, and therefore
    one memoized sequence: the farthest-point search — past the anchors of a
    :data:`PALETTES` set, or from the first color for :data:`CVD_PALETTE` — is
    paid for at most once per name per process.
    """
    if name == CVD_PALETTE:
        return CVDDistinctColors()
    try:
        anchors = PALETTES[name]
    except KeyError:
        names = ", ".join(sorted([*PALETTES, CVD_PALETTE]))
        raise KeyError(
            f"unknown palette {name!r}; choose one of: {names}, "
            "or pass a color generator"
        ) from None
    return SequenceColors(anchors, overflow=CVDDistinctColors(avoid=anchors))


def resolve_generator(generator: "ColorGenerator | str") -> ColorGenerator:
    """Resolve a color generator, or the name of a preset palette, to a generator.

    The counterpart to `luxonis_ml.utils.color.gradient.resolve_gradient`, so a
    palette is chosen the same way a colormap is: by name, anywhere a generator
    is accepted.

    Args:
        generator: A ``int -> Color`` callable, the name of a palette in
            :data:`PALETTES`, or :data:`CVD_PALETTE` for the unbounded
            `CVDDistinctColors` search on its own.

    Returns:
        The `ColorGenerator`. For a :data:`PALETTES` name, one that hands out
        the named colors in order and then continues with `CVDDistinctColors`.

    Raises:
        KeyError: If a name is given that is not a registered palette.

    Examples:
        >>> resolve_generator("okabe-ito")(0)
        Color(r=230, g=159, b=0, a=255)
        >>> resolve_generator("cvd")(0) == CVDDistinctColors()(0)
        True
        >>> resolve_generator("nope")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        KeyError: "unknown palette 'nope'; choose one of: ..."

    """
    if isinstance(generator, str):
        return _named_generator(generator)
    return generator


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
        generator: ColorGenerator | str | None = None,
        colors: Mapping[str, ColorLike] | None = None,
    ) -> None:
        """Create a palette, optionally pinning class order and exact colors.

        Args:
            classes: Class names to register up front, in the order that fixes their
                colors. Any class not listed is assigned on first use.
            generator: The ``int -> Color`` strategy, or the name of a
                colorblind-safe palette in :data:`PALETTES`; defaults to
                `GoldenRatioColors`.
            colors: Explicit ``{class_name: color}`` pins. A pinned class always
                gets exactly that color and never consumes a generator slot, so
                pins do not shift the sequence colors of the other classes.

        """
        self._generator: ColorGenerator = (
            GoldenRatioColors()
            if generator is None
            else resolve_generator(generator)
        )
        self._colors: dict[str, Color] = {}
        # Assigning a color is a read-modify-write of `_colors`, and
        # `DEFAULT_PALETTE` is shared process-wide, so serialize that step.
        self._lock = threading.Lock()
        self._pins: dict[str, Color] = {
            name: Color.parse(value) for name, value in (colors or {}).items()
        }
        if classes is not None:
            for name in classes:
                self.color_for(name)

    @classmethod
    def from_colormap(
        cls,
        gradient: "Gradient | str",
        count: int | None = None,
        *,
        classes: Iterable[str] | None = None,
    ) -> "Palette":
        """Build a categorical palette by sampling a continuous colormap.

        Reuses the colormaps registered in
        :data:`luxonis_ml.utils.color.gradient.GRADIENTS`, so ``"viridis"``
        means the same thing here as it does on a heatmap and there is one place
        to add a colormap.

        How many colors to take is the interesting question, and the answer
        depends on whether you know the class count up front:

        - ``count=None`` (the default) samples lazily at bit-reversal positions
          — ``0, 1, ½, ¼, ¾, ⅛, …``. Each index has one fixed position, so a
          class's color never changes when new classes turn up, yet any prefix
          is spread over the whole colormap rather than bunched at one end.
          There is no limit and no repetition; the colors simply crowd together
          as classes accumulate, since a colormap is a one-dimensional path.
        - ``count=n`` takes exactly ``n`` evenly spaced colors, the way
          ``matplotlib``'s ``resampled`` does, which is what you want when the
          class count is known and the colors have to match a legend built
          elsewhere. Class ``n`` and beyond wrap around and *repeat* colors, so
          only pin a count you are sure of.

        Args:
            gradient: A `Gradient`, or the name of a colormap in
                :data:`luxonis_ml.utils.color.gradient.GRADIENTS`.
            count: How many colors to divide the colormap into, or ``None`` for
                unbounded lazy sampling.
            classes: Class names to register up front, fixing their order.

        Returns:
            The `Palette`.

        Raises:
            KeyError: If a colormap name is given that is not registered.

        Examples:
            >>> palette = Palette.from_colormap("viridis", 4)
            >>> [palette.at(i).rgb for i in range(4)]
            [(68, 1, 84), (50, 103, 140), (63, 182, 115), (253, 231, 37)]

            Lazily, the first two classes take the ends of the colormap and each
            later one splits the widest remaining gap:

            >>> lazy = Palette.from_colormap("viridis")
            >>> lazy.color_for("near") == lazy.at(0)
            True
            >>> lazy.color_for("far") == lazy.at(1)
            True
            >>> lazy.at(0).rgb, lazy.at(1).rgb
            ((68, 1, 84), (253, 231, 37))

        """
        return cls(
            classes,
            generator=ColormapColors(resolve_gradient(gradient), count),
        )

    def __len__(self) -> int:
        """Return the number of classes assigned a color so far."""
        return len(self._colors)

    @property
    def generator(self) -> ColorGenerator:
        """The ``int -> Color`` strategy this palette assigns colors from.

        Readable so a *fresh* palette can be built on an existing color scheme
        — which is what a caller that pins class order
        (``Palette(classes, generator=...)``) needs when the scheme itself
        comes from somewhere else, such as a theme. Cloning the palette
        instead would carry over its already-assigned colors, and with them
        whatever order those classes happened to be seen in.

        Examples:
            >>> palette = Palette(generator="okabe-ito")
            >>> pinned = Palette(["car", "bus"], generator=palette.generator)
            >>> pinned.color_for("car") == palette.at(0)
            True

        """
        return self._generator

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
        later call for that key returns the same color. Two threads racing to
        register different keys get different colors: only the assignment is
        locked, so the common case of an already-colored key stays lock-free.

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
            with self._lock:
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
