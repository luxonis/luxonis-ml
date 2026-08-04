"""The `Tooltip`: hover metadata carried by a spatial annotation.

A `Tooltip` is plain data — a title, a list of key/value rows, optional JSON-like
``data``, and an optional tint color — attached to an annotation (see
`Annotation.tooltip`). It says *what* to show when the annotation is hovered; the
interactive viewer decides *how* to draw it (see `luxonis_ml.vizlab.viewer`).
Keeping it rendering-free means the data layer can attach hover content without
importing any windowing code.

Its strings are inline markup, like every other string vizlab draws.
"""

from dataclasses import dataclass

from luxonis_ml.typing import ParamValue

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
        data: Optional JSON-like value (a mapping, sequence, or scalar, nested
            arbitrarily) shown after ``rows`` as an indented key/value tree —
            the same shape the metadata panel draws, so a nested blob reads the
            same wherever it appears. See `resolved_rows`.
        tint: Optional title color; ``None`` uses the default card heading color.

    Examples:
        >>> Tooltip(title="car", rows=(("id", "7"),)).is_empty
        False
        >>> Tooltip().is_empty
        True
        >>> Tooltip(title="<b>car</b>", rows=(("id", "7"),)).title
        '<b>car</b>'
        >>> Tooltip(data={"p": 0.5}).is_empty
        False

    """

    title: str | None = None
    rows: tuple[tuple[str, str], ...] = ()
    tint: Color | None = None
    data: "ParamValue | None" = None

    def __post_init__(self) -> None:
        """Normalize ``rows`` to tuples, whatever sequence was handed in.

        A list of rows is the obvious thing to write and nothing rejects it —
        this is a plain dataclass, so the annotation is not enforced — but it
        used to fail much later and somewhere else: `resolved_rows` concatenates
        with a tuple, which raises ``TypeError`` only once something actually
        draws the tooltip. Coercing here keeps the field's declared type honest
        and keeps `Tooltip` hashable.
        """
        object.__setattr__(
            self, "rows", tuple((str(k), str(v)) for k, v in self.rows)
        )

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to show (no title, rows, or data)."""
        return not self.title and not self.rows and self.data is None

    @property
    def resolved_rows(self) -> tuple[tuple[str, str], ...]:
        """The rows to draw: ``rows``, then ``data`` flattened into rows.

        JSON-like ``data`` is flattened by the metadata panel's own formatter,
        so a nested value becomes indented ``key: value`` rows (and a list
        becomes bullets) exactly as it would in a side panel. Renderers draw
        this instead of ``rows`` so both kinds of content share one layout.

        Examples:
            >>> Tooltip(data={"scale": [0.9, 1.1]}).resolved_rows
            (('scale:', ''), ('  • ', '0.9'), ('  • ', '1.1'))
            >>> Tooltip(rows=(("id", "7"),), data={"p": 0.5}).resolved_rows
            (('id', '7'), ('p: ', '0.5'))

        """
        if self.data is None:
            return self.rows
        # Imported here: the panel module pulls in the whole scene graph, and a
        # tooltip is plain data that must stay importable without it.
        from luxonis_ml.vizlab.layout.panel import format_tree

        return self.rows + tuple(
            (f"{'  ' * depth}{prefix}", body)
            for depth, prefix, _is_key, body in format_tree(self.data)
        )
