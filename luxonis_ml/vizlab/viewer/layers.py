"""Interactive layer state: which annotation kinds a `Viewer` currently shows.

`LayerState` is the small mutable record a `Viewer` keeps while a window is open —
whether masks, keypoints, and label chips are drawn, which class (if any) is
isolated, and how filled shapes are. It owns the keybindings (`handle`) and, given
a scene's annotations, produces the filtered/adjusted copy to render
(`apply_layers`). The viewer stays generic: it mutates this state on a keypress
and asks its caller to re-render; the *meaning* of the toggles lives here, next to
the annotations they act on.

Nothing here imports a windowing backend, so the state and its transform are pure
and testable without opening a window.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from luxonis_ml.vizlab.annotations import (
    Annotation,
    BBox,
    Classification,
    Keypoints,
    Mask,
    SemanticMask,
)
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.style import Palette, derive_child_color
from luxonis_ml.vizlab.tooltip import Tooltip

#: Fill opacity a first ``[``/``]`` press starts from (before it had been nudged).
_FILL_START = 0.3
#: Step per ``[``/``]`` press.
_FILL_STEP = 0.1


def _is_mask(annotation: Annotation) -> bool:
    """Whether ``annotation`` is a mask layer (instance or semantic)."""
    return isinstance(annotation, (Mask, SemanticMask))


@dataclass
class LayerState:
    """Which annotation kinds are shown, plus a class focus and fill opacity.

    A `Viewer` keeps one of these per interactive session and applies it to every
    window's scene. `handle` maps a keypress to a mutation; `apply_layers` renders
    the state onto a list of annotations.

    Attributes:
        masks: Whether instance/semantic masks are drawn.
        keypoints: Whether keypoints (and their skeletons) are drawn.
        boxes: Whether bounding-box rectangles are drawn. Hiding them keeps any
            keypoints/masks nested inside a box (they render in place, in the
            box's color); a box with nothing inside simply disappears.
        labels: Whether label chips are drawn (shapes stay; only the text chips
            are suppressed, and colors are preserved).
        fill_alpha: Fill/mask opacity override in ``[0, 1]``, or ``None`` to keep
            each annotation's own (themed) opacity.
        focus: A single class name to isolate — every detection of another class
            is hidden — or ``None`` to show all classes.
        classes: The class names ``c`` cycles focus through (kept in sync with the
            classes actually present via `update_classes`).

    """

    masks: bool = True
    keypoints: bool = True
    boxes: bool = True
    labels: bool = True
    fill_alpha: float | None = None
    focus: str | None = None
    classes: tuple[str, ...] = field(default_factory=tuple)

    def is_default(self) -> bool:
        """Whether nothing is toggled (so `apply_layers` can skip its work)."""
        return (
            self.masks
            and self.keypoints
            and self.boxes
            and self.labels
            and self.fill_alpha is None
            and self.focus is None
        )

    def update_classes(self, classes: Sequence[str]) -> None:
        """Set the classes ``c`` cycles through, dropping a now-absent focus.

        Called with the classes present in the current view so cycling only
        offers what is on screen; if the active focus is no longer among them it
        resets to "all" so the frame never goes blank after navigating.

        Args:
            classes: The class names present in what is being shown.

        """
        self.classes = tuple(classes)
        if self.focus is not None and self.focus not in self.classes:
            self.focus = None

    def handle(self, key: str) -> bool:
        """Apply the keypress ``key``; return whether it was a control key.

        A ``True`` return tells the viewer this key changed the view (re-render and
        do not forward it to the caller); ``False`` leaves it for the caller (e.g.
        ``q`` to quit, any other key to advance).

        Args:
            key: The pressed key as a one-character string.

        Returns:
            ``True`` if ``key`` is one of the layer controls, else ``False``.

        """
        lowered = key.lower()
        if lowered == "m":
            self.masks = not self.masks
        elif lowered == "k":
            self.keypoints = not self.keypoints
        elif lowered == "b":
            self.boxes = not self.boxes
        elif lowered == "l":
            self.labels = not self.labels
        elif lowered == "c":
            self._cycle_focus()
        elif key == "[":
            self._nudge(-_FILL_STEP)
        elif key == "]":
            self._nudge(_FILL_STEP)
        else:
            return False
        return True

    def _cycle_focus(self) -> None:
        """Step focus ``all -> class 0 -> class 1 -> ... -> all``."""
        if not self.classes:
            return
        order: list[str | None] = [None, *self.classes]
        current = self.focus if self.focus in order else None
        self.focus = order[(order.index(current) + 1) % len(order)]

    def _nudge(self, delta: float) -> None:
        """Move the fill opacity by ``delta``, clamped to ``[0, 1]``."""
        base = _FILL_START if self.fill_alpha is None else self.fill_alpha
        self.fill_alpha = max(0.0, min(1.0, round(base + delta, 2)))

    def hud(self) -> Tooltip:
        """Return a compact card describing the controls and current state."""
        fill = "auto" if self.fill_alpha is None else f"{self.fill_alpha:.1f}"
        return Tooltip(
            title="Controls",
            rows=(
                ("m", "masks " + ("on" if self.masks else "off")),
                ("k", "keypoints " + ("on" if self.keypoints else "off")),
                ("b", "boxes " + ("on" if self.boxes else "off")),
                ("l", "labels " + ("on" if self.labels else "off")),
                ("c", "class " + (self.focus or "all")),
                ("[ ]", "fill " + fill),
            ),
        )

    def apply_layers(
        self, annotations: Sequence[Annotation], palette: Palette
    ) -> list[Annotation]:
        """Return ``annotations`` filtered and adjusted for the current state.

        Hidden masks/keypoints (and detections of a non-focused class) are dropped;
        with labels off, chip text is removed while each shape keeps its color; a
        fill override is layered onto every shape. Inputs are never mutated — a
        pruned copy is returned (the same list when nothing is toggled).

        Args:
            annotations: The scene's top-level annotations.
            palette: The palette used to bake a color before a label is stripped,
                so hiding labels never changes an annotation's color.

        Returns:
            A new list of annotations to render (or the input unchanged when
            `is_default`).

        """
        if self.is_default():
            return list(annotations)
        out: list[Annotation] = []
        for annotation in annotations:
            if self.focus is not None and not self._in_focus(annotation):
                continue
            out.extend(self._transform(annotation, palette, None))
        return out

    def _in_focus(self, annotation: Annotation) -> bool:
        """Whether a top-level annotation belongs to the focused class.

        A labeled detection of another class is out; an unlabeled or scene-level
        annotation (no class of its own) is kept so focusing never blanks the frame.
        """
        return annotation.label is None or annotation.label == self.focus

    def _transform(
        self,
        annotation: Annotation,
        palette: Palette,
        parent_color: Color | None,
    ) -> list[Annotation]:
        """Copy ``annotation`` with the toggles applied.

        Returns a list because hiding a box replaces it with its (kept) children,
        promoted in place — so one input annotation can yield zero (fully hidden),
        one, or several outputs.
        """
        if _is_mask(annotation) and not self.masks:
            return []
        if isinstance(annotation, Keypoints) and not self.keypoints:
            return []
        own_color = _resolved_color(annotation, palette, parent_color)
        children: list[Annotation] = []
        for child in annotation.children:
            children.extend(self._transform(child, palette, own_color))
        if isinstance(annotation, BBox) and not self.boxes:
            # Drop the rectangle but keep what was inside it. A child that was
            # deriving its color from this (now-gone) box gets that color baked on
            # so it looks exactly as it did nested.
            for child in children:
                if child.color is None:
                    child.color = derive_child_color(own_color)
            return children
        clone = annotation.model_copy()
        clone.children = children
        if self.fill_alpha is not None:
            clone.style_overrides = {
                **clone.style_overrides,
                "fill_alpha": self.fill_alpha,
                "mask_alpha": self.fill_alpha,
            }
        if not self.labels:
            _strip_label(clone, palette)
        return [clone]


def _resolved_color(
    annotation: Annotation, palette: Palette, parent_color: Color | None
) -> Color:
    """Return the color ``annotation`` renders with (`Annotation.resolve_color`).

    Used to bake a color onto a child when its box is hidden, so the child keeps
    the color it had while nested.
    """
    if annotation.color is not None:
        return Color.parse(annotation.color)
    if annotation.label is not None:
        return palette.color_for(annotation.label)
    if parent_color is not None:
        return derive_child_color(parent_color)
    return palette.color_for(f"{type(annotation).__name__}@{id(annotation):x}")


def _strip_label(annotation: Annotation, palette: Palette) -> None:
    """Remove an annotation's chip text in place, keeping its color.

    A labeled annotation's palette color is baked onto it first, so dropping the
    label leaves the shape exactly the same color it had.
    """
    if annotation.color is None and annotation.label is not None:
        annotation.color = palette.color_for(annotation.label)
    annotation.label = None
    annotation.score = None
    annotation.payload = None
    if isinstance(annotation, Classification):
        annotation.tags = []
