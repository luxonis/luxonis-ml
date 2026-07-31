"""Decide what an LDF array label *is*, and build the picture that reads it.

An "array" label is LDF's escape hatch for per-sample tensor data — a disparity
map, an optical flow field, a segmentation score stack, an embedding. It stores
a bare ``.npy`` path and nothing else, so nothing in the data says what the
numbers mean. This module supplies the missing half: it works out how a label
should be read, and hands the values to the matching annotation in
`luxonis_ml.vizlab.annotations.fields`. Labels it cannot place it simply
declines, drawing nothing.

Three things make that harder than it sounds.

**The stored layout is not the payload.** `LuxonisLoader` hands over what
`ArrayAnnotation.combine_to_numpy` builds: ``(N, n_classes, *array_shape)``,
one-hot-sparse along the class axis — row ``i`` holds data only at class slot
``classes[i]`` and zeros everywhere else. Slicing naively gives an all-zero
field. `array_payload` unpacks it, and deliberately stops *before* reducing the
channel axis, since the channel count is what settles everything downstream.

**Two encodings reach the same shape.** Channels can live inside one ``.npy``
(``(2, H, W)`` under a single class), or across several array annotations, one
per class. Both unwrap to ``(2, H, W)``, but only the second has *names* for its
channels, because they are the task's own class names. `array_payload` prefers
that reading whenever the annotations occupy distinct class slots.

**Shape alone is ambiguous.** A ``(3, H, W)`` block is equally plausibly surface
normals, an RGB picture, or three-class scores. `resolve_array_kind` settles it
from the most deliberate signal available: an explicit pin, then a reserved task
name (`RESERVED_TASK_NAMES`), then the shape. Nothing left over is drawn, since
a wrong picture here looks exactly as authoritative as a right one.

.. image:: TODO-HOST/array_kinds.png
   :alt: One scene drawn as a scalar field, a signed error map, optical flow,
         surface normals, a plain image, and class scores.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from luxonis_ml.vizlab.annotations import (
    Annotation,
    ArrayField,
    ArrayImage,
    ColorBar,
    Corner,
    FlowField,
    FlowWheel,
    Legend,
    NormalMap,
    ScalarField,
    SegmentationScores,
)
from luxonis_ml.vizlab.options import ArrayKind, ArrayKinds, RenderOptions

#: Fractional aspect-ratio disagreement tolerated when overlaying a field on an
#: image. Generous enough for a field that was downsampled with rounding, tight
#: enough to reject one that simply describes a differently-shaped picture.
_ASPECT_TOLERANCE = 0.15

#: A field must be at least this many pixels on both axes to be worth drawing.
_MIN_SIDE = 2


@dataclass(frozen=True)
class ArrayPayload:
    """An array label unwrapped to its channel axis, named where possible.

    Attributes:
        data: Either a bare ``(H, W)`` field or a ``(C, H, W)`` stack, as
            ``float32``. The channel count is the signal every viewing decision
            keys off, so it is deliberately *not* reduced further.
        channel_names: One name per channel, present only when the channels came
            from the LDF class axis and so carry real names. ``None`` whenever
            the channels lived inside a single ``.npy``, which stores no names.

    """

    data: np.ndarray
    channel_names: "list[str] | None" = None

    @property
    def channels(self) -> int:
        """Channels in the payload; ``0`` for a bare 2-D field."""
        return 0 if self.data.ndim == 2 else self.data.shape[0]

    @property
    def field_shape(self) -> "tuple[int, int]":
        """The spatial ``(height, width)`` the payload describes."""
        return self.data.shape[-2], self.data.shape[-1]


def _slot_names(
    class_names: "Sequence[str] | None", count: int
) -> "list[str] | None":
    """Name each class slot, or decline when the list does not line up."""
    if class_names is None or len(class_names) != count:
        return None
    return [str(name) for name in class_names]


def _unwrap_loader_axes(
    arr: np.ndarray, class_names: "Sequence[str] | None"
) -> "tuple[np.ndarray, list[str] | None]":
    """Strip the loader's ``(N, n_classes)`` prefix off a stored array.

    ``combine_to_numpy`` always returns ``(N_annotations, n_classes, *shape)``,
    one-hot-sparse: row ``i`` holds data only in class slot ``classes[i]``. Two
    different encodings arrive in that shape and have to be told apart.

    When *several* annotations each occupy a *distinct* class slot, the class
    axis is the channel axis and its slots have real names — one ``.npy`` per
    channel. Summing over the instance axis recovers that stack in class order.

    Otherwise the channels (if any) live inside each ``.npy`` and the class axis
    carries nothing, so the instance axis is summed away instead and several
    annotations of the same class are unioned rather than double-counted.

    Summing rather than maxing matters in both directions: unselected slots are
    exactly zero, so a sum recovers the populated slice including its negative
    values, which a max would silently floor to ``0.0``.
    """
    populated = arr.any(axis=tuple(range(2, arr.ndim)))
    slots = [np.flatnonzero(row) for row in populated]
    filled = [int(slot[0]) for slot in slots if len(slot) == 1]
    # A row with *no* populated slot carries no information about which class it
    # holds, but it also contributes nothing to the sum — a segmentation stack
    # whose background plane is all zeros is still a per-class stack. Only rows
    # that do say something have to agree, and at least one must.
    one_slot_each = all(len(slot) <= 1 for slot in slots)
    distinct = len(set(filled)) == len(filled)
    if len(slots) > 1 and filled and one_slot_each and distinct:
        stack = arr.sum(axis=0)
        return stack, _slot_names(class_names, stack.shape[0])
    per_instance = arr.sum(axis=1)
    if per_instance.shape[0] > 1:
        return np.nanmax(per_instance, axis=0), None
    return per_instance[0], None


#: Trailing-axis sizes small enough to read as channels rather than image rows.
_CHANNEL_LAST_SIZES = (2, 3, 4)


def _channels_first(arr: np.ndarray) -> np.ndarray:
    """Move a trailing channel axis to the front, when there plainly is one.

    A ``.npy`` written by a vision model is ``(C, H, W)``, but one saved straight
    from an image library is ``(H, W, C)``. Only a lopsided shape is
    reinterpreted: a leading axis too big to be channels, beside a trailing axis
    exactly the right size for them. A genuinely ambiguous shape such as
    ``(3, 8, 3)`` keeps its channels-first reading, which is what LDF and torch
    produce.
    """
    leading, trailing = arr.shape[0], arr.shape[-1]
    if leading > max(_CHANNEL_LAST_SIZES) and trailing in _CHANNEL_LAST_SIZES:
        return np.moveaxis(arr, -1, 0)
    return arr


def array_payload(
    values: np.ndarray, *, class_names: "Sequence[str] | None" = None
) -> "ArrayPayload | None":
    """Unwrap a stored array label without collapsing its channels.

    Args:
        values: The raw label, in any shape LDF produces — the loader's
            ``(N, n_classes, *shape)`` or a bare on-disk array from
            `ArrayAnnotation.to_numpy`.
        class_names: The task's class names, in class-id order. Supply them and
            a per-channel encoding comes back named.

    Returns:
        The payload, or ``None`` when the array cannot be a picture at all (an
        embedding vector, a scalar, an empty array, or a stack too deeply nested
        to interpret).

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab.adapters.arrays import array_payload
        >>> one_file = np.zeros((1, 3, 2, 4, 5))  # a (2, 4, 5) payload
        >>> one_file[0, 1] = 7.0
        >>> payload = array_payload(one_file)
        >>> payload.data.shape, payload.channel_names
        ((2, 4, 5), None)

        Two annotations on distinct class slots are the *named* encoding:

        >>> per_channel = np.zeros((2, 2, 4, 5))
        >>> per_channel[0, 0] = 1.0  # slot 0
        >>> per_channel[1, 1] = 2.0  # slot 1
        >>> payload = array_payload(per_channel, class_names=["u", "v"])
        >>> payload.data.shape, payload.channel_names
        ((2, 4, 5), ['u', 'v'])

    """
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    arr = arr.astype(np.float32, copy=False)
    names: list[str] | None = None
    if arr.ndim >= 4:
        arr, names = _unwrap_loader_axes(arr, class_names)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
        names = None
    if arr.ndim not in (2, 3):
        return None
    if names is None and arr.ndim == 3:
        arr = _channels_first(arr)
    if min(arr.shape[-2:]) < _MIN_SIDE:
        return None
    if names is not None and len(names) != arr.shape[0]:
        names = None
    return ArrayPayload(arr, names)


def array_field(
    values: np.ndarray, *, class_names: "Sequence[str] | None" = None
) -> "np.ndarray | None":
    """Reduce a stored array label to the single 2-D field it represents.

    The scalar view: whatever channels survive `array_payload` are unioned into
    one field. Reach for `array_payload` instead when the channel count is the
    thing you need.

    Args:
        values: The raw label, in any of the shapes LDF produces.
        class_names: The task's class names, in class-id order.

    Returns:
        A 2-D ``float32`` field, or ``None`` when the array is not a field at
        all (an embedding vector, a scalar, an empty or all-invalid array).

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab.adapters.arrays import array_field
        >>> loader_shaped = np.zeros((1, 3, 4, 5))
        >>> loader_shaped[0, 1] = 7.0  # only class slot 1 is populated
        >>> field = array_field(loader_shaped)
        >>> field.shape, float(field.max())
        ((4, 5), 7.0)
        >>> array_field(np.zeros(512)) is None  # an embedding is not a picture
        True

    """
    payload = array_payload(values, class_names=class_names)
    if payload is None:
        return None
    field = payload.data
    if field.ndim > 2:
        field = np.nanmax(field, axis=0)
    if not np.isfinite(field).any():
        return None
    return field


#: Task names that declare what their array is. Matched on exact equality after
#: normalization, so ``left_disparity`` deliberately does *not* match — a
#: forgiving rule would have to guess which of several matching words wins.
RESERVED_TASK_NAMES: "dict[str, ArrayKind]" = {
    "disparity": "scalar",
    "depth": "scalar",
    "anomaly": "scalar",
    "confidence": "confidence",
    "uncertainty": "scalar",
    "error": "signed",
    "residual": "signed",
    "diff": "signed",
    "flow": "flow",
    "flow_field": "flow",
    "optical_flow": "flow",
    "normal": "normals",
    "normals": "normals",
    "surface_normals": "normals",
    "rgb": "image",
    "image": "image",
    "segmentation": "scores",
    "logits": "scores",
    "class_scores": "scores",
    "scores": "scores",
}

#: How far a per-pixel vector's length may drift from 1 and still read as a unit
#: normal. Loose enough for a field stored as quantized bytes and decoded back.
_UNIT_TOLERANCE = 0.1


def normalize_task_name(task_name: str) -> str:
    """Fold a task name to the form `RESERVED_TASK_NAMES` is keyed by.

    Examples:
        >>> from luxonis_ml.vizlab.adapters.arrays import normalize_task_name
        >>> normalize_task_name("Flow-Field")
        'flow_field'

    """
    return re.sub(r"[\s\-]+", "_", task_name.strip().lower())


def reserved_array_kind(task_name: str) -> "ArrayKind | None":
    """Read the kind a task name declares, if it declares one.

    Examples:
        >>> from luxonis_ml.vizlab.adapters.arrays import reserved_array_kind
        >>> reserved_array_kind("flow-field")
        'flow'
        >>> reserved_array_kind("left_disparity") is None  # exact match only
        True

    """
    return RESERVED_TASK_NAMES.get(normalize_task_name(task_name))


def _looks_like_normals(data: np.ndarray) -> bool:
    """Whether a 3-channel stack holds unit vectors rather than pixels."""
    finite = np.isfinite(data).all(axis=0)
    if not finite.any():
        return False
    sample = data[:, finite]
    if sample.min() < -1.0 - _UNIT_TOLERANCE:
        return False
    if sample.max() > 1.0 + _UNIT_TOLERANCE:
        return False
    lengths = np.linalg.norm(sample, axis=0)
    return bool(abs(float(np.median(lengths)) - 1.0) <= _UNIT_TOLERANCE)


def infer_array_kind(payload: ArrayPayload) -> "ArrayKind | None":
    """Guess a kind from the payload's shape and values.

    The last resort, used only when nothing declared the kind. It stays
    conservative: a channel count it cannot account for declines rather than
    picking the nearest plausible view, since a wrong picture here looks exactly
    as authoritative as a right one.

    Args:
        payload: The unwrapped array.

    Returns:
        The inferred kind, or ``None`` when the shape settles nothing.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab.adapters.arrays import (
        ...     ArrayPayload,
        ...     infer_array_kind,
        ... )
        >>> infer_array_kind(ArrayPayload(np.zeros((2, 4, 5))))
        'flow'
        >>> residual = np.linspace(-3.0, 3.0, 20).reshape(4, 5)
        >>> infer_array_kind(ArrayPayload(residual))
        'signed'
        >>> unnamed = ArrayPayload(np.zeros((21, 4, 5)))
        >>> infer_array_kind(unnamed) is None  # scores need channel names
        True

    """
    data = payload.data
    channels = payload.channels
    if channels == 0:
        finite = data[np.isfinite(data)]
        if finite.size and float(finite.min()) < 0.0 < float(finite.max()):
            return "signed"
        return "scalar"
    if channels == 2:
        return "flow"
    if channels == 3:
        return "normals" if _looks_like_normals(data) else "image"
    if payload.channel_names is not None:
        # Named channels came off the LDF class axis, so the stack really is one
        # plane per class -- the only case where a deep stack is self-evidently
        # class scores rather than an arbitrary tensor.
        return "scores"
    return None


def explicit_array_kind(
    task_name: str, kinds: "ArrayKinds"
) -> "ArrayKind | None":
    """Find the kind pinned for a task, comparing names leniently."""
    wanted = normalize_task_name(task_name)
    for name, kind in kinds:
        if normalize_task_name(name) == wanted:
            return kind
    return None


def resolve_array_kind(
    task_name: str,
    payload: ArrayPayload,
    *,
    kinds: "ArrayKinds" = (),
) -> "ArrayKind | None":
    """Decide how one array label should be read.

    Precedence runs from most to least deliberate: an explicit pin, then the
    reserved vocabulary the dataset opted into by naming its task, then the
    shape. Anything still unsettled declines.

    Args:
        task_name: The label's complete task path.
        payload: The unwrapped array.
        kinds: Explicit ``(task_name, kind)`` pins, as `RenderOptions.array_kinds`.

    Returns:
        The resolved kind, or ``None`` when the array should not be drawn.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab.adapters.arrays import (
        ...     ArrayPayload,
        ...     resolve_array_kind,
        ... )
        >>> payload = ArrayPayload(np.zeros((2, 4, 5)))
        >>> resolve_array_kind("motion", payload)  # shape alone
        'flow'
        >>> resolve_array_kind("depth", ArrayPayload(np.zeros((4, 5))))
        'scalar'
        >>> resolve_array_kind(  # a pin beats both
        ...     "flow", payload, kinds=(("flow", "image"),)
        ... )
        'image'

    """
    return (
        explicit_array_kind(task_name, kinds)
        or reserved_array_kind(task_name)
        or infer_array_kind(payload)
    )


def is_image_compatible(
    field: np.ndarray,
    image_shape: "tuple[int, int]",
    *,
    tolerance: float = _ASPECT_TOLERANCE,
) -> bool:
    """Whether a field can honestly be blended over an image.

    An exact pixel match always can. So can a lower-resolution field of the same
    proportions, since upsampling it only guesses between real samples. A field
    of *different* proportions cannot: stretching it would put its values over
    pixels they never described.

    Args:
        field: A 2-D field, as returned by `array_field`.
        image_shape: The target image's ``(height, width)``.
        tolerance: Fractional aspect-ratio disagreement to allow.

    Returns:
        Whether the field may be drawn over that image.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab.adapters.arrays import is_image_compatible
        >>> is_image_compatible(np.zeros((270, 480)), (540, 960))  # half-res
        True
        >>> is_image_compatible(
        ...     np.zeros((480, 640)), (540, 960)
        ... )  # 4:3 on 16:9
        False

    """
    height, width = field.shape[:2]
    target_h, target_w = image_shape
    if (height, width) == (target_h, target_w):
        return True
    if min(height, width, target_h, target_w) <= 0:
        return False
    ratio = (width / height) / (target_w / target_h)
    return abs(ratio - 1.0) <= tolerance


#: Opacity for a field blended over a photo. Translucent on purpose: the point
#: of an overlay is to check the field against the image content underneath, so
#: it has to stay visible. A standalone tile has nothing to show through and is
#: drawn opaque.
OVERLAY_ALPHA = 0.6

#: Corner the generated keys sit in, matching the other image-level chrome.
_KEY_CORNER = Corner.BOTTOM_LEFT


@dataclass(frozen=True)
class ArrayDrawing:
    """Everything needed to draw one array label.

    Attributes:
        task_name: The label's complete task path.
        kind: How the label was read.
        field: The annotation drawing the values.
        key: The annotation saying what the colors mean — a `ColorBar` for a
            scalar field, a `FlowWheel` for flow, a `Legend` for class scores.
            ``None`` when keys are switched off or the kind needs none.

    """

    task_name: str
    kind: "ArrayKind"
    field: ArrayField
    key: "Annotation | None" = None

    def annotations(self) -> "list[Annotation]":
        """Return the annotations to add, field first so the key draws over it."""
        return [self.field] if self.key is None else [self.field, self.key]


def _scalar_field(
    payload: ArrayPayload,
    kind: "ArrayKind",
    options: RenderOptions,
    alpha: float,
) -> ScalarField:
    """Build the gradient-read field, centered when the values are signed."""
    center = options.array_center
    if center is None and kind == "signed":
        center = 0.0
    return ScalarField(
        values=payload.data,
        alpha=alpha,
        vmin=options.array_vmin,
        vmax=options.array_vmax,
        center=center,
        ignore_value=options.array_ignore_value,
    )


def _present_classes(
    scores: SegmentationScores, names: "list[str] | None"
) -> "list[str]":
    """Name the classes the argmax actually awarded a pixel to."""
    if names is None:
        return []
    ignored = (
        {scores.ignore_index}
        if isinstance(scores.ignore_index, int)
        else set(scores.ignore_index)
    )
    won = sorted({int(index) for index in np.unique(scores.labels())})
    return [names[i] for i in won if i not in ignored and i < len(names)]


def array_annotation(
    values: np.ndarray,
    *,
    task_name: str,
    options: RenderOptions,
    image_shape: "tuple[int, int] | None" = None,
    class_names: "Sequence[str] | None" = None,
) -> "ArrayDrawing | None":
    """Build the annotations that draw one array label.

    Args:
        values: The raw array label.
        task_name: Its complete task path, which may declare the kind.
        options: Supplies the pins, sentinel, and key setting.
        image_shape: When given, the field must be compatible with this
            ``(height, width)`` or ``None`` is returned. Pass it for overlays;
            leave it unset for a standalone tile.
        class_names: The task's class names, in class-id order.

    Returns:
        An `ArrayDrawing`, or ``None`` when the label is not a picture, does not
        fit ``image_shape``, or nothing settled how to read it.

    """
    payload = array_payload(values, class_names=class_names)
    if payload is None:
        return None
    if image_shape is not None and not is_image_compatible(
        np.empty(payload.field_shape, dtype=bool), image_shape
    ):
        return None
    kind = resolve_array_kind(task_name, payload, kinds=options.array_kinds)
    if kind is None:
        return None
    alpha = 1.0 if image_shape is None else OVERLAY_ALPHA
    key: Annotation | None = None

    if kind in ("scalar", "signed"):
        field: ArrayField = _scalar_field(payload, kind, options, alpha)
        if options.array_colorbar:
            key = ColorBar.for_heatmap(
                field,  # pyright: ignore[reportArgumentType]
                title=task_name,
                corner=_KEY_CORNER,
            )
    elif kind == "flow":
        field = FlowField(values=payload.data, alpha=alpha)
        if options.array_colorbar:
            key = FlowWheel.for_field(
                field,  # pyright: ignore[reportArgumentType]
                title=task_name,
                corner=_KEY_CORNER,
            )
    elif kind == "normals":
        field = NormalMap(values=payload.data, alpha=alpha)
    elif kind == "image":
        field = ArrayImage(values=payload.data, alpha=alpha)
    elif kind == "confidence":
        scores = SegmentationScores(
            values=payload.data, names=payload.channel_names, alpha=alpha
        )
        # Passed only when actually set: `confidence` defaults the range to
        # what a winning probability can occupy, and an explicit `None` would
        # count as a value and suppress that.
        pins: dict[str, object] = {}
        if options.array_vmin is not None:
            pins["vmin"] = options.array_vmin
        if options.array_vmax is not None:
            pins["vmax"] = options.array_vmax
        field = scores.confidence(**pins)
        if options.array_colorbar:
            key = ColorBar.for_heatmap(
                field,  # pyright: ignore[reportArgumentType]
                title=task_name,
                corner=_KEY_CORNER,
            )
    else:
        field = SegmentationScores(
            values=payload.data,
            names=payload.channel_names,
            alpha=alpha,
            weight_by_confidence=kind == "class_confidence",
        )
        entries = _present_classes(field, payload.channel_names)
        if options.array_colorbar and entries:
            key = Legend(entries=entries, title=task_name, corner=_KEY_CORNER)

    field.label = task_name
    return ArrayDrawing(task_name, kind, field, key)


def array_annotations(
    arrays: "Mapping[str, np.ndarray]",
    *,
    options: RenderOptions,
    image_shape: "tuple[int, int] | None" = None,
    class_names: "Mapping[str, Sequence[str]] | None" = None,
) -> "list[ArrayDrawing]":
    """Build the drawings for every array label that can be drawn.

    The single entry point a caller needs: it applies the render options
    uniformly and silently drops the labels nothing could make a picture of.

    Args:
        arrays: Array labels keyed by task name.
        options: Render options, supplying pins, sentinel, and kind overrides.
        image_shape: Pass the target ``(height, width)`` to require image
            compatibility, and to draw at `OVERLAY_ALPHA` so the image shows
            through. Leave unset for opaque standalone tiles.
        class_names: Class names per task, in class-id order.

    Returns:
        One `ArrayDrawing` per drawable label, in input order.

    """
    built: list[ArrayDrawing] = []
    for task_name, values in arrays.items():
        drawing = array_annotation(
            values,
            task_name=task_name,
            options=options,
            image_shape=image_shape,
            class_names=None
            if class_names is None
            else class_names.get(task_name),
        )
        if drawing is not None:
            built.append(drawing)
    return built
