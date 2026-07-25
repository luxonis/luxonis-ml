"""The `Style` dataclass: the resolved look of a single annotation.

A `Style` bundles the handful of knobs that decide how an annotation is
drawn — stroke width, fill opacity, corner radius, label typography, and whether
a soft shadow is used. Annotations resolve to a style at render time; anything
the caller does not override falls back to :data:`DEFAULT_STYLE`.
"""

from dataclasses import dataclass, replace
from enum import Enum


class LabelPlacement(Enum):
    """Where an annotation's label chip is anchored relative to its box."""

    TOP = "top"
    """Above the box's top edge, dropping inside when there is no room above."""

    INSIDE = "inside"
    """Always inside the box, at the top-left corner."""


@dataclass(frozen=True)
class Style:
    """The resolved visual style for one annotation.

    Attributes:
        stroke_width: Outline width in pixels.
        fill_alpha: Opacity of the translucent fill, in ``[0, 1]`` (``0`` disables
            the fill).
        corner_radius: Box corner radius in pixels.
        font_size: Label text size in pixels.
        font_weight: Label OpenType weight (100-900).
        label_pad_x: Horizontal padding inside the label chip, in pixels.
        label_pad_y: Vertical padding inside the label chip, in pixels.
        label_radius: Label chip corner radius in pixels.
        label_placement: Where the label chip is anchored.
        label_alpha: Opacity of the label chip in ``[0, 1]`` (``1`` is opaque). Lower
            it to fade labels, e.g. over a blended/mixup background.
        dash: ``(on, off)`` dash lengths for the outline, or ``None`` for a solid
            stroke. Nested sub-labels default to a dashed outline (see
            `vizlab.style.derive_child_style`).
        keypoint_radius: Radius of a keypoint joint, in pixels.
        keypoint_outline_width: White outline width around each joint, in pixels.
        mask_alpha: Fill opacity for masks, in ``[0, 1]``.
        shadow: Whether shapes cast a soft drop shadow.

    Examples:
        >>> Style().stroke_width
        3.0
        >>> Style().merge(stroke_width=5.0).stroke_width
        5.0
        >>> Style().merge(stroke_width=None) == Style()
        True
        >>> Style().merge(fill_alpha=0.3).stroke_width
        3.0
        >>> Style().dash is None  # top-level boxes are solid by default
        True

    """

    stroke_width: float = 3.0
    fill_alpha: float = 0.16
    corner_radius: float = 9.0
    font_size: float = 15.0
    font_weight: int = 600
    label_pad_x: float = 7.0
    label_pad_y: float = 4.0
    label_radius: float = 6.0
    label_placement: LabelPlacement = LabelPlacement.TOP
    label_alpha: float = 1.0
    dash: tuple[float, float] | None = None
    keypoint_radius: float = 5.0
    keypoint_outline_width: float = 1.5
    mask_alpha: float = 0.45
    shadow: bool = True

    def merge(self, **overrides: object) -> "Style":
        """Return a copy with the given fields replaced.

        Args:
            **overrides: Field values to override; ``None`` values are ignored so
                callers can pass through optional overrides unconditionally.

        Returns:
            A new `Style` with the non-``None`` overrides applied.

        """
        clean = {
            key: value for key, value in overrides.items() if value is not None
        }
        return replace(self, **clean)  # type: ignore[arg-type]


DEFAULT_STYLE = Style()
"""The process-wide default `Style`."""
