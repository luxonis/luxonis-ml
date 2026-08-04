"""Color parsing and manipulation.

The `Color` class is the single color primitive shared across ``luxonis-ml`` —
the visualization layer re-exports it, and other submodules (e.g. augmentation
fill values) parse colors through it too, so no module reimplements color
handling.

Colors are normalized to an ``(r, g, b, a)`` tuple of 8-bit integers. `Color.parse`
accepts the shapes users actually have on hand — hex strings, CSS color names, a
single grayscale integer, RGB/RGBA tuples, or another `Color` — and exposes the
HSL-space operations the palette and style derivation rely on.
"""

import colorsys
from dataclasses import dataclass
from typing import TypeAlias

RGBA: TypeAlias = tuple[int, int, int, int]
"""An ``(r, g, b, a)`` tuple of 8-bit integers (0-255)."""

RGB: TypeAlias = tuple[int, int, int]
"""An ``(r, g, b)`` tuple of 8-bit integers (0-255)."""

#: A small dependency-free set of common color names, so the basics resolve
#: without Pillow. Anything else falls back to Pillow's full CSS set when it is
#: installed (see `Color._from_name`).
_NAMED_COLORS: dict[str, RGB] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "aqua": (0, 255, 255),
    "magenta": (255, 0, 255),
    "fuchsia": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "navy": (0, 0, 128),
    "purple": (128, 0, 128),
    "teal": (0, 128, 128),
    "orange": (255, 165, 0),
}


def _clamp8(value: float) -> int:
    """Clamp a number to the inclusive 0-255 integer range.

    Args:
        value: The value to clamp.

    Returns:
        The value rounded and clamped to ``[0, 255]``.

    """
    return max(0, min(255, round(value)))


@dataclass(frozen=True)
class Color:
    """An immutable RGBA color with HSL-space helpers.

    Attributes:
        r: Red channel, 0-255.
        g: Green channel, 0-255.
        b: Blue channel, 0-255.
        a: Alpha channel, 0-255 (255 is opaque).

    Examples:
        >>> Color.parse("#4c8dff")
        Color(r=76, g=141, b=255, a=255)
        >>> Color.parse("#abc")
        Color(r=170, g=187, b=204, a=255)
        >>> Color.parse("white")
        Color(r=255, g=255, b=255, a=255)
        >>> Color.parse(128)
        Color(r=128, g=128, b=128, a=255)
        >>> Color.parse(300)  # out-of-range channels are clamped
        Color(r=255, g=255, b=255, a=255)
        >>> Color.parse((255, 0, 0))
        Color(r=255, g=0, b=0, a=255)
        >>> Color.parse("white").rgb
        (255, 255, 255)
        >>> Color(255, 0, 0).with_alpha(0.5)
        Color(r=255, g=0, b=0, a=128)
        >>> Color(1, 2, 3, 4).rgba
        (1, 2, 3, 4)
        >>> Color(255, 255, 255).readable_text_color()
        Color(r=17, g=17, b=17, a=255)
        >>> Color(0, 0, 0).readable_text_color()
        Color(r=255, g=255, b=255, a=255)
        >>> c = Color(120, 120, 120)
        >>> c.lighten(0.5).hls[1] > c.hls[1]
        True
        >>> c.darken(0.5).hls[1] < c.hls[1]
        True
        >>> Color.parse("#12")
        Traceback (most recent call last):
            ...
        ValueError: invalid hex color '#12'

    """

    r: int
    g: int
    b: int
    a: int = 255

    @classmethod
    def parse(cls, value: object) -> "Color":
        """Coerce a color-like value into a `Color`.

        Args:
            value: A hex string (``"#rgb"``, ``"#rrggbb"``, ``"#rrggbbaa"``, with
                or without the leading ``#``), a CSS color name (``"red"``), a
                single grayscale integer (``128`` → gray), an ``(r, g, b)`` or
                ``(r, g, b, a)`` tuple, or an existing `Color`.

        Returns:
            The corresponding `Color`.

        Raises:
            ValueError: If the value cannot be interpreted as a color.

        """
        if isinstance(value, Color):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            channel = _clamp8(value)
            return cls(channel, channel, channel)
        if isinstance(value, str):
            return cls._from_str(value)
        if isinstance(value, tuple):
            if len(value) == 3:
                r, g, b = value
                return cls(_clamp8(r), _clamp8(g), _clamp8(b))
            if len(value) == 4:
                r, g, b, a = value
                return cls(_clamp8(r), _clamp8(g), _clamp8(b), _clamp8(a))
            raise ValueError(
                f"color tuple must have 3 or 4 elements, got {len(value)}"
            )
        raise ValueError(f"cannot parse color from {value!r}")

    @classmethod
    def _from_str(cls, text: str) -> "Color":
        """Parse a hex string or a color name.

        A leading ``#`` forces hex parsing (so a bad hex reports as such);
        otherwise hex is tried first, then a color name.
        """
        try:
            return cls._from_hex(text)
        except ValueError:
            if text.startswith("#"):
                raise
            return cls._from_name(text)

    @classmethod
    def _from_name(cls, text: str) -> "Color":
        """Resolve a CSS color name to a `Color`.

        Uses a small built-in table, then Pillow's full name set when it is
        installed.

        Args:
            text: The color name.

        Returns:
            The named `Color`.

        Raises:
            ValueError: If the name is not recognized.

        """
        key = text.strip().lower()
        if key in _NAMED_COLORS:
            r, g, b = _NAMED_COLORS[key]
            return cls(r, g, b)
        try:
            from PIL import ImageColor

            r, g, b = ImageColor.getrgb(text)[:3]
        except (ImportError, ValueError):
            raise ValueError(f"invalid color name {text!r}") from None
        return cls(r, g, b)

    @classmethod
    def _from_hex(cls, text: str) -> "Color":
        """Parse a hex color string.

        Args:
            text: A hex string with 3, 4, 6, or 8 digits, optional leading ``#``.

        Returns:
            The parsed `Color`.

        Raises:
            ValueError: If the string is not valid hex of a supported length.

        """
        s = text.lstrip("#")
        if len(s) in (3, 4):
            s = "".join(ch * 2 for ch in s)
        if len(s) not in (6, 8):
            raise ValueError(f"invalid hex color {text!r}")
        try:
            channels = [int(s[i : i + 2], 16) for i in range(0, len(s), 2)]
        except ValueError as exc:
            raise ValueError(f"invalid hex color {text!r}") from exc
        if len(channels) == 3:
            channels.append(255)
        r, g, b, a = channels
        return cls(r, g, b, a)

    @classmethod
    def from_hls(
        cls, hue: float, lightness: float, saturation: float, a: int = 255
    ) -> "Color":
        """Build a color from hue/lightness/saturation.

        Args:
            hue: Hue in ``[0, 1]``.
            lightness: Lightness in ``[0, 1]``.
            saturation: Saturation in ``[0, 1]``.
            a: Alpha channel, 0-255.

        Returns:
            The corresponding `Color`.

        """
        r, g, b = colorsys.hls_to_rgb(
            hue % 1.0,
            max(0.0, min(1.0, lightness)),
            max(0.0, min(1.0, saturation)),
        )
        return cls(_clamp8(r * 255), _clamp8(g * 255), _clamp8(b * 255), a)

    @property
    def rgb(self) -> RGB:
        """The color as an ``(r, g, b)`` tuple."""
        return (self.r, self.g, self.b)

    @property
    def rgba(self) -> RGBA:
        """The color as an ``(r, g, b, a)`` tuple."""
        return (self.r, self.g, self.b, self.a)

    @property
    def hls(self) -> tuple[float, float, float]:
        """The color as a ``(hue, lightness, saturation)`` tuple in ``[0, 1]``."""
        return colorsys.rgb_to_hls(self.r / 255, self.g / 255, self.b / 255)

    def with_alpha(self, a: float) -> "Color":
        """Return a copy with a new alpha.

        Args:
            a: Alpha as an int (0-255) or a float in ``[0, 1]``.

        Returns:
            A new `Color` with the requested alpha.

        """
        alpha = _clamp8(a * 255) if isinstance(a, float) else _clamp8(a)
        return Color(self.r, self.g, self.b, alpha)

    def lighten(self, amount: float) -> "Color":
        """Move the color toward white in HSL space.

        Args:
            amount: Fraction of the remaining lightness to add, in ``[0, 1]``.

        Returns:
            A lighter `Color`, alpha preserved.

        """
        hue, lightness, saturation = self.hls
        return Color.from_hls(
            hue, lightness + (1.0 - lightness) * amount, saturation, self.a
        )

    def darken(self, amount: float) -> "Color":
        """Move the color toward black in HSL space.

        Args:
            amount: Fraction of the current lightness to remove, in ``[0, 1]``.

        Returns:
            A darker `Color`, alpha preserved.

        """
        hue, lightness, saturation = self.hls
        return Color.from_hls(
            hue, lightness * (1.0 - amount), saturation, self.a
        )

    def saturate(self, amount: float) -> "Color":
        """Increase (or, with a negative amount, decrease) saturation.

        Args:
            amount: Fraction of the remaining saturation to add, in ``[-1, 1]``.

        Returns:
            A `Color` with adjusted saturation, alpha preserved.

        """
        hue, lightness, saturation = self.hls
        if amount >= 0:
            new = saturation + (1.0 - saturation) * amount
        else:
            new = saturation * (1.0 + amount)
        return Color.from_hls(hue, lightness, new, self.a)

    def shift_hue(self, turns: float) -> "Color":
        """Rotate the hue around the color wheel.

        Args:
            turns: Amount to rotate, in turns (``1.0`` is a full circle).

        Returns:
            A hue-rotated `Color`, alpha preserved.

        """
        hue, lightness, saturation = self.hls
        return Color.from_hls(hue + turns, lightness, saturation, self.a)

    @property
    def relative_luminance(self) -> float:
        """WCAG relative luminance in ``[0, 1]``, channels linearized first.

        Examples:
            >>> Color(255, 255, 255).relative_luminance
            1.0
            >>> Color(0, 0, 0).relative_luminance
            0.0

        """

        def channel(value: int) -> float:
            v = value / 255.0
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

        return (
            0.2126 * channel(self.r)
            + 0.7152 * channel(self.g)
            + 0.0722 * channel(self.b)
        )

    def contrast_ratio(self, other: "Color") -> float:
        """Return the WCAG contrast ratio against ``other``, from 1 to 21.

        Args:
            other: The color to compare against.

        Returns:
            The ratio; 4.5 is the AA threshold for body text.

        Examples:
            >>> round(Color(255, 255, 255).contrast_ratio(Color(0, 0, 0)), 1)
            21.0

        """
        pair = sorted((self.relative_luminance, other.relative_luminance))
        return (pair[1] + 0.05) / (pair[0] + 0.05)

    @property
    def is_light(self) -> bool:
        """Whether dark text contrasts better on this color than white does."""
        return self.contrast_ratio(_INK) >= self.contrast_ratio(_PAPER)

    def readable_text_color(self) -> "Color":
        """Return near-black or white, whichever contrasts more on this color.

        The winner is chosen by *measuring* both, not by testing brightness
        against a threshold. A threshold mispicks the middle of the range —
        a mid-tone orange or magenta is dark enough to pass a brightness test
        yet still contrasts nearly twice as well with black as with white — so
        exactly the saturated chip colors a palette generates were the ones
        getting unreadable text.

        Returns:
            Opaque `Color`: a soft near-black or white, whichever wins.

        Examples:
            A mid-tone orange, where a brightness threshold picks white:

            >>> Color(216, 132, 46).readable_text_color()
            Color(r=17, g=17, b=17, a=255)

        """
        return _INK if self.is_light else _PAPER


#: The two candidates `Color.readable_text_color` chooses between: a soft
#: near-black rather than pure black, which reads less harshly on a chip.
_INK = Color(17, 17, 17)
_PAPER = Color(255, 255, 255)

ColorLike: TypeAlias = str | int | tuple[int, ...] | Color
"""Anything `Color.parse` accepts: a hex string or color name, a grayscale int,
an RGB/RGBA tuple, or a `Color`."""
