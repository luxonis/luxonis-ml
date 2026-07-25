"""Small color helpers for the data stack.

`resolve_color` normalizes a color-like value (a matplotlib color name, a single
grayscale integer, or an RGB tuple) into an RGB value. It is used for augmentation
fill values (see ``LetterboxResize``).
"""

import matplotlib.colors

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
        return matplotlib.colors.to_rgb(color)  # type: ignore
    if isinstance(color, int):
        _check_range(color)
        return color, color, color
    for c in color:
        _check_range(c)
    return color
