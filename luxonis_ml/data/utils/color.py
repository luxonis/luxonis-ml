"""Small color helpers for the data stack.

`resolve_color` normalizes a color-like value (a CSS/Pillow color name, a single
grayscale integer, or an RGB tuple) into an RGB value. It is used for augmentation
fill values (see ``LetterboxResize``).
"""

from PIL import ImageColor

from luxonis_ml.typing import RGB, Color


def resolve_color(color: Color) -> RGB:
    r"""Resolve a color to RGB.

    Args:
        color: Color to resolve.

    Returns:
        The resolved RGB color.

    Raises:
        ValueError: If an integer channel value is outside
            :math:`\left[0, 255\right]`.

    Examples:
        >>> resolve_color(12)
        (12, 12, 12)
        >>> resolve_color((1, 2, 3))
        (1, 2, 3)
        >>> resolve_color(300)
        Traceback (most recent call last):
        ...
        ValueError: Color value 300 is out of range [0, 255]

    """

    def _check_range(val: int) -> None:
        if val < 0 or val > 255:
            raise ValueError(f"Color value {val} is out of range [0, 255]")

    if isinstance(color, str):
        # Preserve the historical float-[0, 1] return for named/hex colors.
        r, g, b = ImageColor.getrgb(color)[:3]
        return r / 255, g / 255, b / 255  # type: ignore
    if isinstance(color, int):
        _check_range(color)
        return color, color, color
    for c in color:
        _check_range(c)
    return color
