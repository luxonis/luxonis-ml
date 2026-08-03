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
    ArrayField,
    BBox,
    Classification,
    ColorBar,
    FlowWheel,
    Heatmap,
    Keypoints,
    Mask,
    SemanticMask,
)
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.style import Palette, derive_child_color

#: Decluttering only touches scenes with at least this many detection boxes;
#: below it a scene is not busy enough for a tiny box to read as noise.
_DECLUTTER_MIN_SCENE = 20
#: A detection counts as "small" when even its longer normalized side is under
#: this (~4% of the frame, roughly a 30-40px box on a fitted view).
_DECLUTTER_SMALL_SIDE = 0.04
#: Radius (in normalized image space) that counts as "all around" a detection.
_DECLUTTER_RADIUS = 0.15
#: Number of other detections within that radius that make a spot crowded.
_DECLUTTER_CROWD = 5


@dataclass(frozen=True)
class Control:
    """One row of the controls HUD: a key, what it does, and its current value.

    Attributes:
        key: The key(s) that trigger it (e.g. ``"m"`` or ``"[ ]"``).
        name: The layer/action name (e.g. ``"masks"``).
        value: The current value shown (e.g. ``"on"``, ``"off"``, a class, or a
            fill fraction).
        active: ``True`` if engaged (drawn brightly), ``False`` if off (dimmed),
            or ``None`` for a neutral value that is neither (e.g. ``class all``).

    """

    key: str
    name: str
    value: str
    active: bool | None


def _is_mask(annotation: Annotation) -> bool:
    """Whether ``annotation`` is a mask layer (instance or semantic)."""
    return isinstance(annotation, (Mask, SemanticMask))


def _is_field(annotation: Annotation) -> bool:
    """Whether ``annotation`` is a dense array field, or the key that reads it."""
    return isinstance(annotation, (Heatmap, ArrayField, ColorBar, FlowWheel))


@dataclass
class LayerState:
    """Which annotation kinds are shown, plus per-class visibility.

    A `Viewer` keeps one of these per interactive session and applies it to every
    window's scene. `handle` maps a keypress to a mutation, `toggle_class` a
    legend click; `apply_layers` renders the state onto a list of annotations.

    Attributes:
        masks: Whether instance/semantic masks are drawn.
        keypoints: Whether keypoints (and their skeletons) are drawn.
        boxes: Whether bounding-box rectangles are drawn. Hiding them keeps any
            keypoints/masks nested inside a box (they render in place, in the
            box's color); a box with nothing inside simply disappears.
        labels: Whether label chips are drawn (shapes stay; only the text chips
            are suppressed, and colors are preserved).
        declutter: Whether tiny detections crowded by many others are dropped in
            busy scenes, so a dense pile of unreadable specks does not bury the
            frame (see `_declutter`). On by default; toggle off to see every
            detection.
        hidden: Class names currently switched off — their detections are not
            drawn (but stay in the legend, shown disabled). Toggled per class via
            `toggle_class` (a legend click); the ``c`` key cycles an isolate-one
            sequence over them (see `handle`).
        classes: The class names present in the current view, in order (kept in
            sync via `update_classes`); drives the legend and the ``c`` cycle.
        arrays: Whether dense scalar fields (a `Heatmap` and its `ColorBar`) are
            drawn. Toggled with ``a``.
        has_arrays: Whether the current sample has any array field at all. This
            is *content*, not a toggle: it only decides whether the ``a`` control
            is worth listing, so the panel stays clean for the vast majority of
            datasets that have no arrays.

    """

    masks: bool = True
    keypoints: bool = True
    boxes: bool = True
    labels: bool = True
    declutter: bool = True
    hidden: set[str] = field(default_factory=set)
    classes: tuple[str, ...] = field(default_factory=tuple)
    arrays: bool = True
    has_arrays: bool = False
    #: Cursor for the ``c`` isolate cycle: ``None`` = all shown, else the index
    #: in ``classes`` of the one class kept. A legend click clears it (the manual
    #: visibility no longer matches the cycle), so the next ``c`` restarts it.
    _focus: int | None = None

    def copy(self) -> "LayerState":
        """Return an independent snapshot suitable for render-ahead work."""
        return LayerState(
            masks=self.masks,
            keypoints=self.keypoints,
            boxes=self.boxes,
            labels=self.labels,
            declutter=self.declutter,
            hidden=set(self.hidden),
            classes=self.classes,
            arrays=self.arrays,
            has_arrays=self.has_arrays,
            _focus=self._focus,
        )

    def is_default(self) -> bool:
        """Whether nothing is toggled (so `apply_layers` can skip its work)."""
        return (
            self.masks
            and self.keypoints
            and self.boxes
            and self.labels
            and self.arrays
            and not self.hidden
        )

    def update_classes(self, classes: Sequence[str]) -> None:
        """Set the classes present in the current view (order preserved).

        Drives the legend and the ``c`` isolate cycle. The hidden set persists
        across samples (a class switched off stays off when it reappears), but a
        stale isolate cursor is reset so the cycle stays consistent.

        Args:
            classes: The class names present in what is being shown.

        """
        self.classes = tuple(classes)
        if self._focus is not None and self._focus >= len(self.classes):
            self._focus = None

    def toggle_class(self, name: str) -> None:
        """Show/hide one class (a legend click), leaving the ``c`` cycle to reset.

        Args:
            name: The class to flip on/off.

        """
        if name in self.hidden:
            self.hidden.discard(name)
        else:
            self.hidden.add(name)
        # Manual visibility no longer matches the isolate cycle, so the next
        # ``c`` press resets rather than blindly advancing (see `_cycle_class`).
        self._focus = None

    def toggle_all_classes(self) -> None:
        """Show every class if any is hidden, else hide them all (a master toggle).

        The switch beside the legend heading: one click clears every hidden
        class, the next hides the whole set. Either way the ``c`` isolate cursor
        is reset so its cycle restarts cleanly.
        """
        self.hidden = set() if self.hidden else set(self.classes)
        self._focus = None

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
        elif lowered == "a":
            # Consumed even when the sample has no fields, so the key never
            # silently changes meaning between samples (``c`` behaves the same
            # way when there are no classes).
            self.arrays = not self.arrays
        elif lowered == "d":
            self.declutter = not self.declutter
        elif lowered == "c":
            self._cycle_class()
        else:
            return False
        return True

    def _implied_hidden(self) -> set[str]:
        """Return the hidden set the isolate cursor currently stands for."""
        if self._focus is None or self._focus >= len(self.classes):
            return set()
        keep = self.classes[self._focus]
        return {name for name in self.classes if name != keep}

    def _cycle_class(self) -> None:
        """Step the isolate cycle ``all -> class 0 -> ... -> all``.

        If the visibility was changed by legend clicks (so it no longer matches
        the cursor), the first ``c`` press resets to "all shown" and restarts the
        cycle from the beginning rather than advancing from an arbitrary state.
        """
        if not self.classes:
            return
        if self.hidden != self._implied_hidden():
            self._focus = None
            self.hidden = set()
            return
        order: list[int | None] = [None, *range(len(self.classes))]
        index = order.index(self._focus) if self._focus in order else 0
        self._focus = order[(index + 1) % len(order)]
        self.hidden = self._implied_hidden()

    def controls(self) -> list[Control]:
        """Describe the current controls, for the panel's controls section.

        Returns one `Control` per key: the layer toggles carry their on/off state,
        and the class control shows ``all`` or ``N off``. It deliberately does not
        name the isolated class — the legend already shows which classes are on,
        and a variable-length class name would make the panel width jump.
        """

        def toggle(key: str, name: str, on: bool) -> Control:
            return Control(key, name, "on" if on else "off", on)

        if not self.hidden:
            class_value, class_active = "all", None
        else:
            class_value, class_active = f"{len(self.hidden)} off", True
        controls = [
            toggle("m", "masks", self.masks),
            toggle("k", "keypoints", self.keypoints),
            toggle("b", "boxes", self.boxes),
            toggle("l", "labels", self.labels),
            toggle("d", "declutter", self.declutter),
        ]
        # Listed only when there is something to toggle, like the class row.
        if self.has_arrays:
            controls.append(toggle("a", "arrays", self.arrays))
        controls.append(Control("c", "class", class_value, class_active))
        # Not a layer toggle, and deliberately not in `handle`: saving writes
        # the frame rather than changing it, so the viewer owns the action (see
        # `luxonis_ml.vizlab.viewer.Viewer.save`). It is listed here because
        # this is the one description of the key map, feeding both the HUD and
        # the side panel — including the panel row that makes it clickable.
        controls.append(Control("s", "save", "png", None))
        return controls

    def apply_layers(
        self, annotations: Sequence[Annotation], palette: Palette
    ) -> list[Annotation]:
        """Return ``annotations`` filtered and adjusted for the current state.

        Tiny crowded detections are dropped first (when `declutter` is on); then
        annotations of a hidden class (nested ones included) and hidden
        masks/keypoints are dropped, a hidden class's ids are blanked from
        semantic masks, and with labels off chip text is removed while each
        shape keeps its color. Inputs are never mutated — a pruned copy is
        returned (the same annotations, in a new list, when nothing changes
        them).

        Args:
            annotations: The scene's top-level annotations.
            palette: The palette used to bake a color before a label is stripped,
                so hiding labels never changes an annotation's color.

        Returns:
            A new list of annotations to render.

        """
        kept = list(annotations)
        if self.declutter:
            kept = _declutter(kept)
        if self.is_default():
            return kept
        out: list[Annotation] = []
        for annotation in kept:
            out.extend(self._transform(annotation, palette, None))
        return out

    def _transform(
        self,
        annotation: Annotation,
        palette: Palette,
        parent_color: Color | None,
    ) -> list[Annotation]:
        """Copy ``annotation`` with the toggles applied.

        Returns a list because hiding a box replaces it with its (kept) children,
        promoted in place — so one input annotation can yield zero (fully hidden),
        one, or several outputs. An annotation of a hidden class yields nothing,
        at any nesting depth.
        """
        if annotation.label is not None and annotation.label in self.hidden:
            return []
        if _is_mask(annotation) and not self.masks:
            return []
        if _is_field(annotation) and not self.arrays:
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
        if isinstance(clone, SemanticMask):
            _hide_mask_classes(clone, self.hidden)
        if not self.labels:
            _strip_label(clone, palette)
        return [clone]


def _hide_mask_classes(mask: SemanticMask, hidden: set[str]) -> None:
    """Extend ``mask.ignore_index`` with the ids of classes in ``hidden``.

    A semantic mask carries its classes in its own id map rather than in
    ``label``, so hiding a class means teaching the (already copied) mask to
    treat those ids as background.
    """
    if not hidden or mask.labels is None:
        return
    hidden_ids = [
        int(class_id)
        for class_id in mask._ids()
        if mask._name(int(class_id)) in hidden
    ]
    if not hidden_ids:
        return
    ignored = mask.ignore_index
    existing = [ignored] if isinstance(ignored, int) else list(ignored)
    mask.ignore_index = [*existing, *hidden_ids]


def _declutter(annotations: list[Annotation]) -> list[Annotation]:
    """Drop tiny detection boxes that sit in a crowd of other detections.

    In a busy scene a detection far smaller than the frame and ringed by many
    others is unreadable noise — its box and chip only clutter the pile. Such a
    box (and whatever it nests) is removed; every larger or more isolated
    detection, and every non-box annotation, is kept. The size test is in
    normalized image space, so it scales with the frame rather than the display.
    Sparse scenes (fewer than `_DECLUTTER_MIN_SCENE` boxes) are returned as-is.

    Args:
        annotations: The scene's top-level annotations.

    Returns:
        ``annotations`` unchanged when nothing qualifies, else a new list with
        the crowded tiny boxes removed. Never mutates the input or its members.

    """
    boxes = [a for a in annotations if isinstance(a, BBox)]
    if len(boxes) < _DECLUTTER_MIN_SCENE:
        return annotations
    small = [i for i, a in enumerate(boxes) if _is_small(a)]
    centers = [(a.x + a.w / 2.0, a.y + a.h / 2.0) for a in boxes]
    radius_sq = _DECLUTTER_RADIUS**2
    drop = {
        id(boxes[index])
        for index in small
        if _is_crowded(index, centers, radius_sq)
    }
    if not drop:
        return annotations
    return [a for a in annotations if id(a) not in drop]


def _is_crowded(
    index: int,
    centers: list[tuple[float, float]],
    radius_sq: float,
) -> bool:
    """Whether ``centers[index]`` has enough neighbors within the radius."""
    cx, cy = centers[index]
    neighbors = 0
    for other, (nx, ny) in enumerate(centers):
        if other == index:
            continue
        if (nx - cx) ** 2 + (ny - cy) ** 2 > radius_sq:
            continue
        neighbors += 1
        if neighbors >= _DECLUTTER_CROWD:
            return True
    return False


def _is_small(box: BBox) -> bool:
    """Whether a box is small enough (even its longer side) to read as a speck."""
    return max(box.w, box.h) < _DECLUTTER_SMALL_SIDE


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
    return palette.color_for(annotation.unlabeled_color_key())


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
