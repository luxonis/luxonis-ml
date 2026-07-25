"""Bundled-font management.

The library ships two variable fonts so rendering never depends on system fonts
(which also sidesteps noisy fontconfig warnings): **Inter** for proportional
("sans") text and **JetBrains Mono** for monospace ("mono") text — the same pair
used across Luxonis products. Each family ships an upright and an italic file, and
is cloned to a requested weight on demand and cached. `FontManager` hands out sized
``skia.Font`` objects for any ``(weight, italic, mono)`` combination.
"""

import functools
from pathlib import Path

import skia

_FONTS_DIR = Path(__file__).parent / "fonts"

# (mono, italic) -> variable font file.
_FONT_FILES = {
    (False, False): _FONTS_DIR / "InterVariable.ttf",
    (False, True): _FONTS_DIR / "InterVariable-Italic.ttf",
    (True, False): _FONTS_DIR / "JetBrainsMonoVariable.ttf",
    (True, True): _FONTS_DIR / "JetBrainsMonoVariable-Italic.ttf",
}

_WGHT_TAG = (ord("w") << 24) | (ord("g") << 16) | (ord("h") << 8) | ord("t")

# An empty, custom font manager: it holds no system fonts and never consults
# fontconfig, so loading the bundled fonts stays silent (Skia's default manager
# would otherwise scan ``/etc/fonts`` and emit noisy fontconfig warnings).
_FONT_MGR = skia.FontMgr.New_Custom_Empty()


@functools.lru_cache(maxsize=4)
def _base_typeface(italic: bool, mono: bool) -> skia.Typeface:
    """Load and cache a bundled base typeface.

    Args:
        italic: Whether to load the italic variant.
        mono: Whether to load JetBrains Mono (monospace) instead of Inter.

    Returns:
        The base (weight-400) ``skia.Typeface``.

    Raises:
        FileNotFoundError: If the bundled font file is missing.

    """
    path = _FONT_FILES[(mono, italic)]
    tf = _FONT_MGR.makeFromFile(str(path), 0)
    if tf is None:  # pragma: no cover - the fonts are bundled with the package
        raise FileNotFoundError(f"bundled font not found: {path}")
    return tf


@functools.lru_cache(maxsize=64)
def _weighted_typeface(weight: int, italic: bool, mono: bool) -> skia.Typeface:
    """Return a bundled typeface cloned to a specific weight.

    Args:
        weight: OpenType weight (100-900), e.g. ``400`` regular, ``600`` semibold.
        italic: Whether to use the italic variant.
        mono: Whether to use JetBrains Mono (monospace) instead of Inter.

    Returns:
        The weight-adjusted ``skia.Typeface``.

    """
    base = _base_typeface(italic, mono)
    coord = skia.FontArguments.VariationPosition.Coordinate(
        _WGHT_TAG, float(weight)
    )
    coords = skia.FontArguments.VariationPosition.Coordinates([coord])
    position = skia.FontArguments.VariationPosition(coords)
    args = skia.FontArguments()
    args.setVariationDesignPosition(position)
    return base.makeClone(args)


class FontManager:
    """Hands out sized, weighted ``skia.Font`` objects from the bundled fonts."""

    def font(
        self,
        size: float,
        *,
        weight: int = 400,
        italic: bool = False,
        mono: bool = False,
    ) -> skia.Font:
        """Build a font at the given size, weight, and family.

        Args:
            size: Text size in pixels.
            weight: OpenType weight (100-900).
            italic: Whether to use the italic variant.
            mono: Whether to use JetBrains Mono (monospace) instead of Inter.

        Returns:
            A configured ``skia.Font`` with subpixel anti-aliasing enabled.

        """
        font = skia.Font(_weighted_typeface(weight, italic, mono), float(size))
        font.setSubpixel(True)
        font.setEdging(skia.Font.Edging.kAntiAlias)
        return font


DEFAULT_FONTS = FontManager()
"""The process-wide default `FontManager`."""
