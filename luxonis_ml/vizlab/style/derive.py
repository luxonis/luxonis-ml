"""Derive a nested annotation's look from its parent's.

Nesting (a ``driver`` box inside a ``car`` box) reads best when the child is
visibly *related* to the parent rather than independently colored. These helpers
take the parent's already-resolved color and style and step them one level: the
color lightens and shifts hue slightly, the stroke thins and the fill softens.
Because each level derives from the previous one's resolved values, the effect
compounds naturally with depth without any level needing to know how deep it is.
"""

from luxonis_ml.vizlab.color import Color

from .style import Style


def derive_child_color(parent: Color) -> Color:
    """Return a child color one derivation step lighter than the parent's.

    Args:
        parent: The parent's resolved color.

    Returns:
        A lighter, slightly hue-shifted `Color`.

    Examples:
        >>> parent = Color(50, 110, 190)
        >>> derive_child_color(parent).hls[1] > parent.hls[1]
        True

    """
    return parent.lighten(0.30).shift_hue(0.02)


def derive_child_style(parent: Style) -> Style:
    """Return a child style one derivation step thinner/softer than the parent's.

    Args:
        parent: The parent's resolved style.

    Returns:
        A `Style` with a thinner, dashed stroke and softer fill.

    Examples:
        >>> derive_child_style(Style(stroke_width=4.0)).stroke_width < 4.0
        True
        >>> derive_child_style(Style()).dash is not None
        True

    """
    stroke_width = max(1.25, parent.stroke_width * 0.72)
    return parent.merge(
        stroke_width=stroke_width,
        fill_alpha=parent.fill_alpha * 0.85,
        # A dashed outline signals "sub-label"; scale the pattern to the stroke.
        dash=parent.dash or (stroke_width * 2.4, stroke_width * 2.0),
    )
