"""The `Tooltip`: hover metadata carried by a spatial annotation.

A `Tooltip` is plain data — a title, a list of key/value rows, and an optional
tint color — attached to an annotation (see `Annotation.tooltip`). It says *what*
to show when the annotation is hovered; the interactive viewer decides *how* to
draw it (see `luxonis_ml.vizlab.viewer`). Keeping it rendering-free means the
data layer can attach hover content without importing any windowing code.
"""

from dataclasses import dataclass

from .color import Color


@dataclass(frozen=True)
class Tooltip:
    """Hover content for one annotation.

    Attributes:
        title: Optional heading (e.g. a class name), drawn in the tint color.
        rows: Ordered ``(key, value)`` string pairs shown beneath the title.
        tint: Optional title color; ``None`` uses the default card heading color.

    Examples:
        >>> Tooltip(title="car", rows=(("id", "7"),)).is_empty
        False
        >>> Tooltip().is_empty
        True

    """

    title: str | None = None
    rows: tuple[tuple[str, str], ...] = ()
    tint: Color | None = None

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to show (no title and no rows)."""
        return not self.title and not self.rows
