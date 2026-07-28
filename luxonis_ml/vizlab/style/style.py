"""The `Style` dataclass: the resolved look of a single annotation.

A `Style` bundles the handful of knobs that decide how an annotation is
drawn — stroke width, fill opacity, corner radius, label typography, and whether
a soft shadow is used. Annotations resolve to a style at render time; anything
the caller does not override falls back to :data:`DEFAULT_STYLE`.
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal

FontFamily = Literal["sans", "mono"]
"""Which bundled family a label uses: proportional Inter or monospace JetBrains Mono."""


class LabelPlacement(Enum):
    """Where an annotation's label chip is anchored relative to its box."""

    TOP = "top"
    """Above the box's top edge, dropping inside when there is no room above."""

    INSIDE = "inside"
    """Always inside the box, at the top-left corner."""


class MaskOutline(Enum):
    """How much work a mask's contour outline is worth.

    A speed/quality knob for mask and semantic-segmentation outlines. It only
    affects masks; and the smoothing it governs only runs when a mask is
    *upscaled*, so downscaled views render the same under every setting.
    """

    SMOOTH = "smooth"
    """Traced outline, corner-rounded when upscaled (the crisp default)."""

    CRISP = "crisp"
    """Traced outline without the corner-rounding pass — faster, but the
    pixel staircase shows through on an upscaled mask."""

    NONE = "none"
    """No outline at all; masks are drawn as translucent fills only (fastest)."""


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
        font_family: ``"sans"`` (Inter) or ``"mono"`` (JetBrains Mono).
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
        mask_outline: Detail level of mask/semantic contours — ``SMOOTH`` (default,
            corner-rounded when upscaled), ``CRISP`` (traced but unsmoothed, faster),
            or ``NONE`` (fill only, fastest). Only masks are affected.
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
    font_size: float = 16.0
    font_weight: int = 600
    font_family: FontFamily = "sans"
    label_pad_x: float = 7.0
    label_pad_y: float = 4.0
    label_radius: float = 6.0
    label_placement: LabelPlacement = LabelPlacement.TOP
    label_alpha: float = 1.0
    dash: tuple[float, float] | None = None
    keypoint_radius: float = 5.0
    keypoint_outline_width: float = 1.5
    mask_alpha: float = 0.45
    mask_outline: MaskOutline = MaskOutline.SMOOTH
    shadow: bool = True

    @property
    def mono(self) -> bool:
        """Whether labels render with the monospace family."""
        return self.font_family == "mono"

    def merge(self, **overrides) -> "Style":
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
        return replace(self, **clean)

    def scaled(self, factor: float) -> "Style":
        """Return a copy with every pixel dimension multiplied by ``factor``.

        Scales the metrics that should track the canvas resolution — strokes,
        typography, chip padding/radius, keypoint joints, and the dash pattern —
        so labels stay proportionate (and readable) on large images and small
        ones alike. Opacities, weight, placement, and shadow are unchanged.

        Args:
            factor: Multiplier for the pixel dimensions (``1.0`` is a no-op).

        Returns:
            The scaled `Style` (``self`` when ``factor`` is ``1.0``).

        """
        if factor == 1.0:
            return self
        return replace(
            self,
            stroke_width=self.stroke_width * factor,
            corner_radius=self.corner_radius * factor,
            font_size=self.font_size * factor,
            label_pad_x=self.label_pad_x * factor,
            label_pad_y=self.label_pad_y * factor,
            label_radius=self.label_radius * factor,
            keypoint_radius=self.keypoint_radius * factor,
            keypoint_outline_width=self.keypoint_outline_width * factor,
            dash=None
            if self.dash is None
            else (self.dash[0] * factor, self.dash[1] * factor),
        )

    @contextmanager
    def as_default(self) -> "Generator[Style, None, None]":
        """Use this style as the scope's default within a ``with`` block.

        A `ContextVar`-scoped replacement for the theme's style: every annotation
        rendered inside the block that does not override its own style falls back
        to this one instead of the active theme's. Nesting and threads are safe.

        Yields:
            This style.

        Examples:
            >>> bold = Style(stroke_width=6.0)
            >>> with bold.as_default():
            ...     current_default_style() is bold
            True

        """
        token = _AMBIENT_STYLE.set(self)
        try:
            yield self
        finally:
            _AMBIENT_STYLE.reset(token)

    @classmethod
    @contextmanager
    def override(
        cls, **overrides: Any
    ) -> "Generator[Mapping[str, Any], None, None]":
        """Layer style ``overrides`` over the default within a ``with`` block.

        Unlike `as_default` (which replaces the whole style), this merges only the
        named fields over whatever the effective default is — the active theme's
        style, or an enclosing `as_default`/`override` scope — leaving every other
        field untouched. Override values are in display pixels (not scaled).

        Args:
            **overrides: `Style` field values to layer for the scope.

        Yields:
            The merged override mapping in effect for the scope.

        Examples:
            >>> with Style.override(stroke_width=6.0):
            ...     current_style_overrides()["stroke_width"]
            6.0

        """
        merged = {**(_AMBIENT_OVERRIDES.get() or {}), **overrides}
        token = _AMBIENT_OVERRIDES.set(merged)
        try:
            yield merged
        finally:
            _AMBIENT_OVERRIDES.reset(token)


DEFAULT_STYLE = Style()
"""The process-wide default `Style`."""

#: Scoped default style set by `Style.as_default` (``None`` outside any scope).
_AMBIENT_STYLE: ContextVar[Style | None] = ContextVar(
    "vizlab_ambient_style", default=None
)
#: Scoped field overrides accumulated by `Style.override` (``None`` when unset).
_AMBIENT_OVERRIDES: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "vizlab_ambient_style_overrides", default=None
)


def current_default_style() -> Style | None:
    """Return the scope's default style from `Style.as_default`, or ``None``."""
    return _AMBIENT_STYLE.get()


def current_style_overrides() -> Mapping[str, Any]:
    """Return the scope's accumulated `Style.override` fields (possibly empty)."""
    return _AMBIENT_OVERRIDES.get() or {}
