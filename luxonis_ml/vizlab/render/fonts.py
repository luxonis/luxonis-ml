"""Bundled-font management.

The library ships two variable fonts so rendering never depends on system fonts
(which also sidesteps noisy fontconfig warnings): **Inter** for proportional
("sans") text and **JetBrains Mono** for monospace ("mono") text — the same pair
used across Luxonis products. Each family ships an upright and an italic file, and
is cloned to a requested weight on demand and cached. `FontManager` hands out sized
``skia.Font`` objects for any ``(weight, italic, mono)`` combination.

The two families cover Latin, Greek and Cyrillic text. A character they lack,
such as a CJK ideograph in a class name, comes from a system font instead:
`FontManager.runs` splits text into runs that each have their glyphs in one
font. The system fonts are consulted only for such a character, so text that
the bundled fonts cover still never depends on them.

Right-to-left text (Hebrew, Arabic) needs more than glyphs: its characters are
stored in reading order, which runs right to left, and Arabic letters change
shape to join. `needs_shaping` tells such text apart, and
`FontManager.paragraph` lays it out with Skia's paragraph shaper, which does
both. All other text keeps the plain glyph-by-glyph path.
"""

import functools
import os
import sys
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import skia

#: The bundled ``*.ttf`` files, declared as package data of ``luxonis_ml.vizlab``
#: in ``pyproject.toml``. Anchored to the package rather than to this module's
#: own location, so moving this module never breaks font loading.
_FONTS_DIR = Path(str(resources.files("luxonis_ml.vizlab"))) / "fonts"

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


@contextmanager
def _quiet_stderr() -> Iterator[None]:
    """Silence the C-level standard error, where fontconfig writes warnings."""
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


@functools.cache
def _system_fonts() -> skia.FontMgr:
    """Open the system font manager once, without its fontconfig warnings.

    The warnings are about the system's own font configuration, which a
    caller of vizlab can do nothing about.
    """
    with _quiet_stderr():
        return skia.FontMgr.RefDefault()


@functools.lru_cache(maxsize=256)
def _fallback_typeface(
    char: str, weight: int, italic: bool
) -> skia.Typeface | None:
    """Find a system typeface that draws ``char``, or ``None``."""
    slant = (
        skia.FontStyle.kItalic_Slant
        if italic
        else skia.FontStyle.kUpright_Slant
    )
    style = skia.FontStyle(weight, skia.FontStyle.kNormal_Width, slant)
    return _system_fonts().matchFamilyStyleCharacter("", style, [], ord(char))


#: The layout width of a shaped paragraph: wide enough that it never wraps.
_ONE_LINE = 1e6

#: The bidi classes of right-to-left characters: Hebrew (``R``) and Arabic,
#: Syriac or Thaana (``AL``).
_RTL_CLASSES = frozenset({"R", "AL"})


def needs_shaping(text: str) -> bool:
    """Tell whether ``text`` holds a right-to-left character.

    Drawn a glyph at a time, such text reads backwards, and Arabic letters
    do not join. `FontManager.paragraph` lays it out correctly.

    Examples:
        >>> needs_shaping("car 7"), needs_shaping("car שלום")
        (False, True)

    """
    return any(
        unicodedata.bidirectional(char) in _RTL_CLASSES for char in text
    )


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
        return _configured(
            skia.Font(_weighted_typeface(weight, italic, mono), float(size))
        )

    def runs(
        self,
        text: str,
        size: float,
        *,
        weight: int = 400,
        italic: bool = False,
        mono: bool = False,
    ) -> list[tuple[skia.Font, str]]:
        """Split text into runs that each have their glyphs in one font.

        Text that the bundled font covers comes back as one run. A character
        it lacks gets a system font that has it; when no installed font has
        it either, it stays with the bundled font and draws as a box.

        Args:
            text: The text to split.
            size: Text size in pixels.
            weight: OpenType weight (100-900).
            italic: Whether to use the italic variant.
            mono: Whether to use JetBrains Mono (monospace) instead of Inter.

        Returns:
            The ``(font, text)`` runs, in reading order.

        Examples:
            >>> runs = DEFAULT_FONTS.runs("car", 12)
            >>> [chunk for _, chunk in runs]
            ['car']

        """
        font = self.font(size, weight=weight, italic=italic, mono=mono)
        glyphs = font.textToGlyphs(text)
        if all(glyphs) or len(glyphs) != len(text):
            return [(font, text)]
        runs: list[tuple[skia.Font, str]] = []
        for char, glyph in zip(text, glyphs, strict=True):
            typeface = (
                None if glyph else _fallback_typeface(char, weight, italic)
            )
            run_font = (
                font
                if typeface is None
                else _configured(skia.Font(typeface, float(size)))
            )
            last = runs[-1][0] if runs else None
            if last is not None and _same_typeface(last, run_font):
                runs[-1] = (last, runs[-1][1] + char)
            else:
                runs.append((run_font, char))
        return runs

    def paragraph(
        self,
        text: str,
        size: float,
        paint: skia.Paint,
        *,
        weight: int = 400,
        italic: bool = False,
        mono: bool = False,
    ) -> skia.textlayout_Paragraph:
        """Shape ``text`` into a one-line paragraph, in visual order.

        Skia's paragraph shaper reorders right-to-left runs and joins Arabic
        letters. It draws with the same fonts as `runs`: the bundled font
        for what it covers, a system font for the rest.

        Args:
            text: The text to shape.
            size: Text size in pixels.
            paint: The paint to draw the glyphs with.
            weight: OpenType weight (100-900).
            italic: Whether to use the italic variant.
            mono: Whether to use JetBrains Mono (monospace) instead of Inter.

        Returns:
            The laid-out paragraph. Its ``LongestLine`` is its width; paint
            it at the origin's y minus its ``AlphabeticBaseline``.

        """
        provider = skia.textlayout_TypefaceFontProvider()
        bundled = _weighted_typeface(weight, italic, mono)
        typefaces = [bundled]
        for font, _ in self.runs(
            text, size, weight=weight, italic=italic, mono=mono
        ):
            typeface = font.getTypeface()
            if all(typeface.uniqueID() != t.uniqueID() for t in typefaces):
                typefaces.append(typeface)
        # Each typeface gets its own alias, so the shaper tries them in
        # this order and the bundled font always comes first.
        families = [f"vizlab-{index}" for index in range(len(typefaces))]
        for typeface, family in zip(typefaces, families, strict=True):
            provider.registerTypeface(typeface, family)
        collection = skia.textlayout_FontCollection()
        collection.setDefaultFontManager(provider)
        style = skia.textlayout_TextStyle()
        style.setFontFamilies(families)
        style.setFontSize(float(size))
        style.setForegroundPaint(paint)
        paragraph_style = skia.textlayout_ParagraphStyle()
        paragraph_style.setTextStyle(style)
        builder = skia.textlayout_ParagraphBuilder.make(
            paragraph_style, collection, skia.Unicode()
        )
        builder.pushStyle(style)
        builder.addText(text)
        paragraph = builder.Build()
        paragraph.layout(_ONE_LINE)
        return paragraph


def _configured(font: skia.Font) -> skia.Font:
    """Turn on the subpixel positioning and anti-aliasing every font uses."""
    font.setSubpixel(True)
    font.setEdging(skia.Font.Edging.kAntiAlias)
    return font


def _same_typeface(a: skia.Font, b: skia.Font) -> bool:
    """Report whether two fonts draw with the same typeface."""
    return a.getTypeface().uniqueID() == b.getTypeface().uniqueID()


DEFAULT_FONTS = FontManager()
"""The process-wide default `FontManager`."""
