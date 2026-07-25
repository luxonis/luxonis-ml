"""Bundled-font management.

The library ships the Inter variable font so rendering never depends on system
fonts (which also sidesteps noisy fontconfig warnings). Typefaces are cloned to a
requested weight on demand and cached, and `FontManager` hands out sized
``skia.Font`` objects.
"""

import functools
from pathlib import Path

import skia

_FONTS_DIR = Path(__file__).parent / "fonts"
_UPRIGHT = _FONTS_DIR / "InterVariable.ttf"
_ITALIC = _FONTS_DIR / "InterVariable-Italic.ttf"

_WGHT_TAG = (ord("w") << 24) | (ord("g") << 16) | (ord("h") << 8) | ord("t")

# An empty, custom font manager: it holds no system fonts and never consults
# fontconfig, so loading the bundled font stays silent (Skia's default manager
# would otherwise scan ``/etc/fonts`` and emit noisy fontconfig warnings).
_FONT_MGR = skia.FontMgr.New_Custom_Empty()


@functools.lru_cache(maxsize=2)
def _base_typeface(italic: bool) -> skia.Typeface:
    """Load and cache the bundled Inter typeface.

    Args:
        italic: Whether to load the italic variant.

    Returns:
        The base (weight-400) ``skia.Typeface``.

    Raises:
        FileNotFoundError: If the bundled font file is missing.

    """
    path = _ITALIC if italic else _UPRIGHT
    tf = _FONT_MGR.makeFromFile(str(path), 0)
    if tf is None:  # pragma: no cover - the font is bundled with the package
        raise FileNotFoundError(f"bundled font not found: {path}")
    return tf


@functools.lru_cache(maxsize=32)
def _weighted_typeface(weight: int, italic: bool) -> skia.Typeface:
    """Return the Inter typeface cloned to a specific weight.

    Args:
        weight: OpenType weight (100-900), e.g. ``400`` regular, ``600`` semibold.
        italic: Whether to use the italic variant.

    Returns:
        The weight-adjusted ``skia.Typeface``.

    """
    base = _base_typeface(italic)
    coord = skia.FontArguments.VariationPosition.Coordinate(
        _WGHT_TAG, float(weight)
    )
    coords = skia.FontArguments.VariationPosition.Coordinates([coord])
    position = skia.FontArguments.VariationPosition(coords)
    args = skia.FontArguments()
    args.setVariationDesignPosition(position)
    return base.makeClone(args)


class FontManager:
    """Hands out sized, weighted ``skia.Font`` objects from the bundled font."""

    def font(
        self, size: float, *, weight: int = 400, italic: bool = False
    ) -> skia.Font:
        """Build a font at the given size and weight.

        Args:
            size: Text size in pixels.
            weight: OpenType weight (100-900).
            italic: Whether to use the italic variant.

        Returns:
            A configured ``skia.Font`` with subpixel anti-aliasing enabled.

        """
        font = skia.Font(_weighted_typeface(weight, italic), float(size))
        font.setSubpixel(True)
        font.setEdging(skia.Font.Edging.kAntiAlias)
        return font


DEFAULT_FONTS = FontManager()
"""The process-wide default `FontManager`."""
