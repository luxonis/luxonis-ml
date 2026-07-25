"""Image-level classification tags rendered as a stack of corner chips.

`Classification` shows one or more class tags (multi-label) or a single tag
(binary) as rounded chips stacked in a corner of the image. Each chip is colored by
its class name from the palette, so a class reads the same here as on a box.
"""

from collections.abc import Iterable

from luxonis_ml.vizlab.style import Palette, Style

from .base import RenderContext
from .chip import compose_label
from .overlay import Cell, CornerStack, chip_cell


def _split(tag: str | tuple[str, float]) -> tuple[str, float | None]:
    """Split a tag into ``(name, score)``.

    Args:
        tag: A class name, or a ``(name, score)`` pair.

    Returns:
        The name and optional score.

    Examples:
        >>> _split("cat")
        ('cat', None)
        >>> _split(("cat", 0.9))
        ('cat', 0.9)

    """
    if isinstance(tag, str):
        return tag, None
    name, score = tag
    return name, score


class Classification(CornerStack):
    """Render image-level classes as colored corner chips.

    Each tag becomes one chip. Multiple `Classification` overlays anchored to
    the same corner are offset into one visual stack instead of covering one
    another.

    Attributes:
        tags: Class names, or ``(name, score)`` pairs, one chip each.

    See `CornerStack` for ``corner``/``margin``/``gap`` and
    `Annotation` for ``style``/``palette``.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import Classification, Image
        >>> tags = Classification(tags=["indoor", ("cat", 0.92)])
        >>> Image(np.zeros((40, 80, 3), np.uint8)).add(tags).render().shape
        (40, 80, 4)

    """

    tags: list[str | tuple[str, float]] = []

    @classmethod
    def from_ldf(
        cls,
        class_names: Iterable[str | tuple[str, float]],
        *,
        palette: Palette | None = None,
    ) -> "Classification":
        """Build a classification overlay from LDF class names.

        In LDF, any detection whose ``class_name`` is set (with no geometry)
        contributes an image-level classification target; collect those names
        and pass them here as one corner-stacked overlay.

        Args:
            class_names: Class names, or ``(name, score)`` pairs, one chip each.
            palette: Palette used to color each chip by name.

        Returns:
            The equivalent `Classification` overlay.

        """
        return cls(tags=list(class_names), palette=palette)

    def _cells(self, ctx: RenderContext, style: Style) -> list[Cell]:
        canvas = ctx.canvas
        palette = self.resolved_palette(ctx)
        cells: list[Cell] = []
        for tag in self.tags:
            name, score = _split(tag)
            text = compose_label(name, score, None)
            if not text:
                continue
            cells.append(
                chip_cell(canvas, text, palette.color_for(name), style)
            )
        return cells
