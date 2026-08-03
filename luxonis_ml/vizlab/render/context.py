"""Resolved ambient state for one complete render."""

from collections.abc import Collection, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from luxonis_ml.vizlab.style import (
    Style,
    StyleValue,
    current_default_style,
    current_style_overrides,
)

if TYPE_CHECKING:
    from luxonis_ml.vizlab.annotations.layout import Placement

#: Every layer an annotation can belong to, in draw order. An annotation
#: declares its own through `Annotation.LAYER`, so a custom one joins the
#: vocabulary by setting that rather than by being listed here.
LAYERS: tuple[str, ...] = (
    "field",
    "mask",
    "polyline",
    "box",
    "keypoint",
    "relation",
    "overlay",
)


@dataclass(frozen=True)
class LayerMask:
    """Which parts of a scene a render pass is allowed to draw.

    Rendering the same scene several times under different masks is how a
    layered export is built: one pass for the picture and its chrome, then one
    per layer that draws nothing else. Because the mask travels in the
    `RenderEnvironment` it reaches annotations nested inside composites, which
    a scene-level filter cannot — a `luxonis_ml.vizlab.Composite` holds a paint
    closure, not a list of children.

    Attributes:
        kinds: Layers allowed to draw, or ``None`` for all of them.
        classes: Class names allowed to draw, or ``None`` for all of them. An
            unlabeled annotation draws only when this is ``None``.
        labels: Whether label chips draw. Separate from ``kinds`` because chips
            are their own pass over every annotation, not a layer of their own.
        chrome: Whether non-annotation content draws — the base image of every
            scene, and a composite's borders, panels, titles and legends.

    """

    kinds: "frozenset[str] | None" = None
    classes: "frozenset[str] | None" = None
    labels: bool = True
    chrome: bool = True

    def draws(self, layer: str, label: "str | None" = None) -> bool:
        """Return whether an annotation may draw under this mask.

        Args:
            layer: The layer name to test (see `Annotation.LAYER`).
            label: The annotation's class name, if it has one.

        Returns:
            Whether the annotation should draw its own marks.

        """
        if self.kinds is not None and layer not in self.kinds:
            return False
        return self.allows(label)

    def allows(self, label: "str | None") -> bool:
        """Return whether ``label``'s class may draw, ignoring its layer.

        The label-chip pass is gated on this alone: a chip belongs to its
        class, not to the layer of the shape underneath it, so isolating one
        class keeps its chips while isolating one layer does not move them.

        Args:
            label: The annotation's class name, if it has one.

        Returns:
            Whether that class may draw.

        """
        return self.classes is None or label in self.classes


#: The mask in effect outside any `layer_scope`: everything draws.
EVERYTHING = LayerMask()

_LAYERS: ContextVar[LayerMask] = ContextVar(
    "vizlab_layers", default=EVERYTHING
)


def current_layers() -> LayerMask:
    """Return the `LayerMask` in effect for the current context."""
    return _LAYERS.get()


@contextmanager
def layer_scope(
    kinds: "Collection[str] | None" = None,
    *,
    classes: "Collection[str] | None" = None,
    labels: bool = True,
    chrome: bool = True,
) -> "Generator[LayerMask, None, None]":
    """Restrict what renders inside a ``with`` block.

    A `ContextVar` scope, like `luxonis_ml.vizlab.Style.override`, so it needs
    no argument threading and reaches every annotation the block draws however
    deeply it is nested.

    Args:
        kinds: Layers allowed to draw, or ``None`` for all of them.
        classes: Class names allowed to draw, or ``None`` for all of them.
        labels: Whether label chips draw.
        chrome: Whether base images and layout chrome draw.

    Yields:
        The `LayerMask` in effect for the block.

    Examples:
        Draw only the boxes of a composed scene, over transparency:

        >>> from luxonis_ml.vizlab.render.context import layer_scope
        >>> with layer_scope({"box"}, chrome=False) as mask:
        ...     mask.draws("box"), mask.draws("mask"), mask.chrome
        (True, False, False)

    """
    mask = LayerMask(
        kinds=None if kinds is None else frozenset(kinds),
        classes=None if classes is None else frozenset(classes),
        labels=labels,
        chrome=chrome,
    )
    token = _LAYERS.set(mask)
    try:
        yield mask
    finally:
        _LAYERS.reset(token)


#: Collector filled during a render when `layer_inventory` is active.
_INVENTORY: ContextVar["dict[tuple[str, str | None], None] | None"] = (
    ContextVar("vizlab_layer_inventory", default=None)
)


def record_layer(layer: str, label: "str | None") -> None:
    """Note that ``layer``/``label`` drew, when an inventory is being taken."""
    found = _INVENTORY.get()
    if found is not None:
        found.setdefault((layer, label), None)


@contextmanager
def layer_inventory() -> (
    "Generator[dict[tuple[str, str | None], None], None, None]"
):
    """Collect the ``(layer, class)`` pairs a render actually draws.

    A `luxonis_ml.vizlab.Composite` holds a paint closure rather than a list of
    children, so what a composed scene contains cannot be read off it — it can
    only be observed while it draws. Wrapping a pass that would happen anyway
    makes that free.

    Yields:
        The mapping being filled, keyed in draw order and complete once the
        block exits. A mapping rather than a set so the order an export emits
        its fragments in is the order they were painted.

    """
    found: dict[tuple[str, str | None], None] = {}
    token = _INVENTORY.set(found)
    try:
        yield found
    finally:
        _INVENTORY.reset(token)


#: Chip placements resolved by the first pass of a multi-pass render.
_PLAN: ContextVar["dict[tuple[int, int], Placement] | None"] = ContextVar(
    "vizlab_label_plan", default=None
)


def current_label_plan() -> "dict[tuple[int, int], Placement] | None":
    """Return the chip placements to reuse, or ``None`` outside a `label_plan`."""
    return _PLAN.get()


@contextmanager
def label_plan() -> "Generator[dict[tuple[int, int], Placement], None, None]":
    """Resolve every label chip's position once and reuse it for the block.

    Chip layout is collision-aware, so where one chip lands depends on every
    chip placed before it. A layered export therefore has to place *all* of them
    on every pass — including the ones that pass will not paint — or the chips
    that remain would drift as layers are split out. Placing is the most
    expensive part of a pass, and it produces the same answer every time.

    Inside this block the first pass records what it placed and the rest look it
    up, keyed by `LabelLayout.occurrence` so the answer follows the chip rather
    than the order it was reached in.

    Yields:
        The mapping being filled, usable after the block to inspect placements.

    """
    plan: dict[tuple[int, int], Placement] = {}
    token = _PLAN.set(plan)
    try:
        yield plan
    finally:
        _PLAN.reset(token)


@dataclass(frozen=True)
class RenderEnvironment:
    """A snapshot of scoped style state at the render boundary.

    Rendering resolves ambient style once and threads this immutable value
    through the whole scene. An annotation therefore never observes a different
    scope halfway through a render, and cache keys depend on the same state that
    drawing consumes.
    """

    default_style: Style | None
    style_overrides: Mapping[str, StyleValue]
    layers: LayerMask = EVERYTHING

    @classmethod
    def current(cls) -> "RenderEnvironment":
        """Capture the style state in effect for the current context."""
        return cls(
            default_style=current_default_style(),
            style_overrides=dict(current_style_overrides()),
            layers=current_layers(),
        )
