"""`RenderOptions`: one bundle of render-wide look and behavior.

`RenderOptions` gathers everything a render falls back to that is not set on an
individual annotation: the `Theme` (style + palette + background), the default
`Gradient` for heatmaps, and the behavior of the LDF adapter (skeletons, keypoint
labels, metadata handling). Pass it explicitly (``Image(options=...)``,
``visualize_record(..., options=...)``), or install one for a scope with
`default_options` / `set_default_options` — a `ContextVar`, so it is thread-safe
and test-isolated rather than a mutable module global.

The palette lives inside ``theme`` (there is no separate ``palette`` field), so a
class keeps one source of truth for its color; pin classes with
`Theme.with_class_colors` or `Theme.with_palette`. A colorblind-safe palette is
selected by name through the same door the default `Gradient` is::

    RenderOptions(
        theme=DARK_THEME.with_palette("okabe-ito"), gradient="viridis"
    )
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

from .gradient import DEFAULT_GRADIENT, Gradient
from .style import DARK_THEME, Theme

KeypointLabelMode = Literal["none", "numbers", "names", "full"]
"""How keypoints are labeled: nothing, index, name, or ``index:name``."""

SkeletonDef = tuple[list[str], list[tuple[int, int]]]
"""A keypoint skeleton as ``(labels, edges)`` — ``get_skeletons()``'s shape."""

ArrayView = Literal["off", "tile", "overlay"]
"""How an array label is shown: not at all, in its own tile, or over the photo."""

ArrayKind = Literal[
    "scalar",
    "signed",
    "flow",
    "normals",
    "image",
    "scores",
    "confidence",
    "class_confidence",
]
"""What an array label *is*, which decides how its values become colors.

Shape alone cannot settle this — a ``(3, H, W)`` array is equally plausibly
surface normals, an RGB picture, or three-class scores — and one stack has
more than one honest reading (``scores`` and ``confidence`` share it), so the
kind is resolved
from an explicit setting, then a reserved task name, then the shape. See
`luxonis_ml.vizlab.adapters.arrays.resolve_array_kind`.
"""

ArrayKinds: TypeAlias = tuple[tuple[str, ArrayKind], ...]
"""Explicit ``(task_name, kind)`` overrides, scoped per task."""

RenderOptionValue: TypeAlias = (
    Theme
    | Gradient
    | str
    | Mapping[str, SkeletonDef]
    | ArrayKinds
    | bool
    | float
    | None
)
"""A value accepted by one of `RenderOptions`' configurable fields."""


@dataclass(frozen=True)
class RenderOptions:
    """Render-wide defaults for look and LDF-adapter behavior.

    Attributes:
        theme: The look bundle (style + palette + background) an annotation falls
            back to. Pin class colors via `Theme.with_class_colors`.
        gradient: Default colormap for heatmaps that do not set their own.
        skeletons: Keypoint skeletons keyed by task name, in LDF's ``(labels,
            edges)`` shape (pass ``LuxonisDataset.get_skeletons()`` directly).
        keypoint_label_mode: How to label keypoints.
        draw_skeletons: Whether to draw skeleton limbs between keypoints.
        hover_metadata: When ``True``, a boxed detection's metadata is attached as
            a hover `Tooltip` instead of crowding the frame.
        antialias: Whether shape fills and strokes are anti-aliased. ``False`` is a
            render-wide speed trade for dense scenes — jagged shape edges, but
            faster; text stays anti-aliased so labels stay legible.
        array_view: Whether, and how, to draw an array label whose shape is
            understandable as a dense field. Off by default — most arrays are
            not pictures. ``"tile"`` gives the field its own panel beside the
            image; ``"overlay"`` blends it onto the image itself.
        array_vmin: Pin the low end of every array field's value range. ``None``
            scales each field to its own minimum, which makes two datasets look
            alike even when their values differ — pin both ends to compare them.
        array_vmax: Pin the high end of every array field's value range.
        array_ignore_value: A "no data" sentinel in array fields, drawn
            transparent and left out of the automatic range. See
            `Heatmap.ignore_value`.
        array_overlay_source: Which image source ``"overlay"`` draws onto.
            ``None`` uses the first, since a field describes the reference view.
        array_colorbar: Whether to draw a `ColorBar` keying each array field.
        array_kinds: Explicit ``(task_name, kind)`` pairs pinning how a given
            array is read, overriding both its reserved name and its shape. Kept
            per task because one sample may carry several arrays of different
            kinds — a disparity map beside an optical flow field, say.
        array_center: The value a signed field's neutral color sits on. ``None``
            centers on ``0.0`` when the kind is ``"signed"``. See
            `Heatmap.center`.

    Examples:
        >>> from luxonis_ml.vizlab import RenderOptions, DARK_THEME
        >>> opts = RenderOptions(hover_metadata=True)
        >>> opts.replace(draw_skeletons=True).draw_skeletons
        True
        >>> opts.theme is DARK_THEME
        True

    """

    theme: Theme = DARK_THEME
    gradient: Gradient | str = DEFAULT_GRADIENT
    skeletons: Mapping[str, SkeletonDef] = field(default_factory=dict)
    keypoint_label_mode: KeypointLabelMode = "numbers"
    draw_skeletons: bool = False
    hover_metadata: bool = False
    antialias: bool = True
    array_view: ArrayView = "off"
    array_vmin: float | None = None
    array_vmax: float | None = None
    array_ignore_value: float | None = None
    array_overlay_source: str | None = None
    array_colorbar: bool = True
    array_kinds: ArrayKinds = ()
    array_center: float | None = None

    def replace(self, **changes: RenderOptionValue) -> "RenderOptions":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)


#: The fallback used when no scope has installed options.
_DEFAULT_OPTIONS = RenderOptions()
_CURRENT: ContextVar[RenderOptions | None] = ContextVar(
    "vizlab_render_options", default=None
)


def current_options() -> RenderOptions:
    """Return the `RenderOptions` in effect for the current scope."""
    options = _CURRENT.get()
    return options if options is not None else _DEFAULT_OPTIONS


@contextmanager
def default_options(
    options: RenderOptions,
) -> Generator[RenderOptions, None, None]:
    """Install ``options`` as the default within a ``with`` block.

    Args:
        options: The options to make current for the scope.

    Yields:
        The installed options.

    Examples:
        >>> from luxonis_ml.vizlab import RenderOptions, default_options
        >>> with default_options(RenderOptions(hover_metadata=True)):
        ...     current_options().hover_metadata
        True

    """
    token = _CURRENT.set(options)
    try:
        yield options
    finally:
        _CURRENT.reset(token)


def set_default_options(options: RenderOptions) -> None:
    """Install ``options`` as the default for the rest of this context.

    Unscoped counterpart to `default_options`, for a "set once at the top of a
    script/notebook" workflow. Still a `ContextVar` under the hood (not a mutable
    module global), so separate threads and tests stay isolated.

    Args:
        options: The options to make current.

    """
    _CURRENT.set(options)
