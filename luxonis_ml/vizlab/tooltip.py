"""The `Tooltip`: hover metadata carried by a spatial annotation.

A `Tooltip` is plain data — a title, a list of key/value rows, and an optional
tint color — attached to an annotation (see `Annotation.tooltip`). It says *what*
to show when the annotation is hovered; the interactive viewer decides *how* to
draw it (see `luxonis_ml.vizlab.viewer`). Keeping it rendering-free means the
data layer can attach hover content without importing any windowing code.

Its strings are inline markup, like every other string vizlab draws.
"""

from dataclasses import dataclass

from .color import Color


@dataclass(frozen=True)
class Tooltip:
    """Hover content for one annotation.

    The title and both halves of every row are drawn as inline markup (see
    `luxonis_ml.vizlab.render.markup`), so a caller can emphasize part of a
    value. Text that did not come from the caller — dataset metadata, a file
    path — should be passed through `luxonis_ml.vizlab.render.markup.escape`
    first so it renders verbatim; the LDF adapter already does this for the
    tooltips it builds.

    Attributes:
        title: Optional heading (e.g. a class name), drawn in the tint color.
        rows: Ordered ``(key, value)`` string pairs shown beneath the title.
        tint: Optional title color; ``None`` uses the default card heading color.

    Examples:
        >>> Tooltip(title="car", rows=(("id", "7"),)).is_empty
        False
        >>> Tooltip().is_empty
        True
        >>> Tooltip(title="<b>car</b>", rows=(("id", "7"),)).title
        '<b>car</b>'

    """

    title: str | None = None
    rows: tuple[tuple[str, str], ...] = ()
    tint: Color | None = None

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to show (no title and no rows)."""
        return not self.title and not self.rows
