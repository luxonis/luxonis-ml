"""Color parsing and manipulation.

Colors are normalized to an ``(r, g, b, a)`` tuple of 8-bit integers. The
`Color` class accepts the shapes users actually have on hand — hex
strings, RGB/RGBA tuples, or another `Color` — and exposes the HSL-space
operations that the palette and sub-label style derivation rely on.
"""

import colorsys
from dataclasses import dataclass
from typing import TypeAlias

RGBA: TypeAlias = tuple[int, int, int, int]
"""An ``(r, g, b, a)`` tuple of 8-bit integers (0-255)."""


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
        >>> Color.parse((255, 0, 0))
        Color(r=255, g=0, b=0, a=255)
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
                or without the leading ``#``), an ``(r, g, b)`` or ``(r, g, b, a)``
                tuple, or an existing `Color`.

        Returns:
            The corresponding `Color`.

        Raises:
            ValueError: If the value cannot be interpreted as a color.

        """
        if isinstance(value, Color):
            return value
        if isinstance(value, str):
            return cls._from_hex(value)
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
    def from_hls(cls, h: float, ll: float, s: float, a: int = 255) -> "Color":
        """Build a color from hue/lightness/saturation.

        Args:
            h: Hue in ``[0, 1]``.
            ll: Lightness in ``[0, 1]``.
            s: Saturation in ``[0, 1]``.
            a: Alpha channel, 0-255.

        Returns:
            The corresponding `Color`.

        """
        r, g, b = colorsys.hls_to_rgb(
            h % 1.0, max(0.0, min(1.0, ll)), max(0.0, min(1.0, s))
        )
        return cls(_clamp8(r * 255), _clamp8(g * 255), _clamp8(b * 255), a)

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
        h, ll, s = self.hls
        return Color.from_hls(h, ll + (1.0 - ll) * amount, s, self.a)

    def darken(self, amount: float) -> "Color":
        """Move the color toward black in HSL space.

        Args:
            amount: Fraction of the current lightness to remove, in ``[0, 1]``.

        Returns:
            A darker `Color`, alpha preserved.

        """
        h, ll, s = self.hls
        return Color.from_hls(h, ll * (1.0 - amount), s, self.a)

    def saturate(self, amount: float) -> "Color":
        """Increase (or, with a negative amount, decrease) saturation.

        Args:
            amount: Fraction of the remaining saturation to add, in ``[-1, 1]``.

        Returns:
            A `Color` with adjusted saturation, alpha preserved.

        """
        h, ll, s = self.hls
        new_s = s + (1.0 - s) * amount if amount >= 0 else s * (1.0 + amount)
        return Color.from_hls(h, ll, new_s, self.a)

    def shift_hue(self, turns: float) -> "Color":
        """Rotate the hue around the color wheel.

        Args:
            turns: Amount to rotate, in turns (``1.0`` is a full circle).

        Returns:
            A hue-rotated `Color`, alpha preserved.

        """
        h, ll, s = self.hls
        return Color.from_hls(h + turns, ll, s, self.a)

    def readable_text_color(self) -> "Color":
        """Return black or white, whichever is more readable on this color.

        Uses the standard relative-luminance threshold so label text placed on a
        chip of this color stays legible.

        Returns:
            Opaque black or white as a `Color`.

        """
        r, g, b = self.r / 255, self.g / 255, self.b / 255
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return Color(17, 17, 17) if luminance > 0.55 else Color(255, 255, 255)


ColorLike: TypeAlias = str | tuple[int, ...] | Color
"""Anything `Color.parse` accepts: a hex string, an RGB/RGBA tuple, or a Color."""
