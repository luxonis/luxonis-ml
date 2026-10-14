"""Composing whole dataset samples — images, arrays, a record — into frames.

`SampleComposer` is the rendering half of dataset inspection: given one
sample's image sources, array labels, `DatasetRecord`, and metadata
panel, it builds the tiles, sizes them for a screen or a file, and frames
them with the side panel. The CLI (or any other caller) keeps only the data
side: loading samples, filtering them, and deciding what to do with the
finished frames.
"""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Literal

import numpy as np

from luxonis_ml.vizlab.annotations import ColorKey
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.interaction import Frame
from luxonis_ml.vizlab.layout.compose import (
    combine,
    key_strip_height,
    order_by_position,
    reference_source,
    with_key_strip,
)
from luxonis_ml.vizlab.layout.panel import Controls, Swatches
from luxonis_ml.vizlab.options import RenderOptions
from luxonis_ml.vizlab.scene.image import Image, Renderable
from luxonis_ml.vizlab.style import Palette

from .arrays import array_annotations
from .instances import (
    ColorBy,
    record_to_colored_annotations,
    spatial_instances,
)
from .ldf import metadata_annotations, metadata_panel, visualize_record

if TYPE_CHECKING:
    from luxonis_ml.ldf import DatasetRecord
    from luxonis_ml.vizlab.layout.panel import PanelData
    from luxonis_ml.vizlab.viewer.layers import LayerState


@dataclass(frozen=True)
class SampleComposer:
    """Build display- or file-ready frames from prepared dataset samples.

    One composer carries a whole inspection session's presentation policy —
    theme, palettes, coloring mode, legend, sizing — so a caller hands it
    only the per-sample data. `frame` sizes for the configured screen and
    keeps the interactive controls; `render` sizes for a file and leaves
    them out. Both draw the same pixels for the same sample otherwise.

    Attributes:
        options: The render options; their theme carries the class palette
            and the background every composite is mounted on.
        identity_palette: Colors for instance and task identities. A separate
            sequence from the class palette, so the first instance or task
            gets the first generated color regardless of how many class names
            the class palette registered.
        color_by: What a detection's color identifies. ``"instance"`` falls
            back per sample (see `fallback_color_by`).
        array_class_names: Class names per task, so an array whose channels
            ride the LDF class axis comes back with those channels named
            rather than numbered.
        legend: Whether the side panel carries a color legend.
        blend_all: Blend every record onto one tile even where per-record
            tiles would be drawn (multi-task class coloring).
        panel: Whether frames get the side panel (controls, legend,
            metadata) at all.
        scale: Display scale; ``"auto"`` fits `frame` to the screen and
            keeps `render` at the source resolution.
        screen: The ``(width, height)`` to fit interactive frames into, or
            ``None`` when there is no screen.
        reserve_class: The class name the legend reserves width for, keeping
            the panel a stable width as the per-sample class set changes.
        reserve_task: As ``reserve_class``, for the task legend.
        class_order: The dataset's classes. The legend lists a sample's
            classes in this order, so a class keeps its place from sample to
            sample; a class it lacks follows, in the sample's order.
        stable_grid: Keep a tile for an array that a sample lacks, so the
            grid of a sequence keeps its arrangement from frame to frame.
            Off by default, because the empty tiles take room.
        panel_width: Horizontal room reserved for the side panel when
            fitting a frame to the screen.

    """

    options: RenderOptions
    identity_palette: Palette = field(default_factory=Palette)
    color_by: ColorBy = "class"
    array_class_names: "Mapping[str, list[str]] | None" = None
    legend: bool = True
    blend_all: bool = False
    panel: bool = True
    scale: 'float | Literal["auto"]' = "auto"
    screen: "tuple[int, int] | None" = None
    reserve_class: str = ""
    reserve_task: str = ""
    class_order: Sequence[str] = ()
    stable_grid: bool = False
    panel_width: float = 400.0

    def fallback_color_by(self, record: "DatasetRecord") -> ColorBy:
        """Resolve this sample's coloring mode.

        Instance coloring needs spatial annotations to color; a sample with
        none falls back to class colors. The caller can compare the result
        with `color_by` to tell the user about the fallback.

        Args:
            record: The sample's record.

        Returns:
            `color_by`, or ``"class"`` when instance coloring has nothing to
            color.

        """
        if self.color_by != "instance" or spatial_instances(record):
            return self.color_by
        return "class"

    def sidebar(
        self,
        panel: "Mapping[str, PanelData]",
        layers: "LayerState",
        *,
        color_by: ColorBy,
        task_names: Sequence[str] = (),
        controls: bool = True,
    ) -> "dict[str, PanelData]":
        """Prepend the CONTROLS and CLASSES sections to a sample's metadata panel.

        The interactive controls and the class-color legend live in the side
        panel (not floated over the image): controls come from ``layers`` (so they
        reflect the current toggles and refresh on every re-render), and the class
        swatches are keyed to the classes present in the sample
        (``layers.classes``) with stable full-dataset palette colors.

        Args:
            panel: The sample's metadata panel.
            layers: The current layer toggles.
            color_by: What a color identifies in this sample, which picks the
                legend.
            task_names: The tasks the legend lists in ``"task"`` coloring.
            controls: Whether to add the controls. Turn them off where they do
                not apply, as in the non-interactive saved renders.

        Returns:
            The controls and the legend, followed by ``panel``.

        """
        out: dict[str, PanelData] = {}
        if controls:
            out["controls"] = Controls(
                tuple(
                    (c.key, c.name, c.value, c.active)
                    for c in layers.controls()
                )
            )
        rank = {name: index for index, name in enumerate(self.class_order)}
        names = sorted(
            layers.classes, key=lambda name: rank.get(name, len(rank))
        )
        class_palette = self.options.theme.palette
        if self.legend and color_by == "class" and names:
            out["classes"] = Swatches(
                tuple((class_palette.color_for(name), name) for name in names),
                disabled=frozenset(layers.hidden),
                # Hold the legend (and panel) width to the dataset's longest class
                # name so it stays put as the per-sample class set changes.
                reserve=self.reserve_class,
            )
        elif self.legend and color_by == "task" and task_names:
            out["tasks"] = Swatches(
                tuple(
                    (self.identity_palette.color_for(name), name)
                    for name in task_names
                ),
                reserve=self.reserve_task,
                interactive=False,
            )
        out.update(panel)
        return out

    def _overlay_arrays(
        self,
        viz: Image,
        arrays: "Mapping[str, np.ndarray]",
        layers: "LayerState",
    ) -> None:
        """Blend any array field onto ``viz``, in the ``overlay`` array view.

        The caller decides *which* source gets them; see `tiles`.
        """
        if (
            self.options.array_view != "overlay"
            or not arrays
            or not layers.arrays
        ):
            return
        # In front of the detections so the field paints beneath them.
        viz.annotations[:0] = [
            annotation
            for drawing in array_annotations(
                arrays,
                options=self.options,
                image_shape=(viz.height, viz.width),
                class_names=self.array_class_names,
                loader_layout=True,
            )
            for annotation in drawing.annotations()
        ]

    def _array_tiles(
        self,
        arrays: "Mapping[str, np.ndarray]",
        layers: "LayerState",
        image_shape: "tuple[int, int] | None" = None,
    ) -> "tuple[list[tuple[Image, ColorKey | None]], list[str]]":
        """Render each array field as its own tile, in the ``tile`` array view.

        A field is often stored at a fraction of the image resolution. Each
        tile is scaled to the height of ``image_shape``, the image tiles'
        ``(height, width)``, so a field is as large as the photo beside it.
        The field keeps its own aspect ratio. Its color key is returned beside
        it rather than drawn on it, for `tiles` to put in a strip under the
        field.

        With `stable_grid`, an array the sample lacks (the loader gives it
        as an empty array) keeps its slot as a hatched, photo-sized tile.
        """
        if (
            self.options.array_view != "tile"
            or not arrays
            or not layers.arrays
        ):
            return [], []
        tiles: list[tuple[Image, ColorKey | None]] = []
        titles: list[str] = []
        background = self.options.theme.background
        drawings = {
            drawing.task_name: drawing
            for drawing in array_annotations(
                arrays,
                options=self.options,
                class_names=self.array_class_names,
                loader_layout=True,
            )
        }
        for task_name, values in arrays.items():
            if values.size == 0:
                if not self.stable_grid:
                    continue
                rows, columns = image_shape or _MISSING_TILE
                canvas = _hatched(rows, columns, background)
                tiles.append((Image(canvas, options=self.options), None))
                titles.append(f"{task_name} \u00b7 not in this frame")
                continue
            drawing = drawings.get(task_name)
            array_field = None if drawing is None else drawing.field.field()
            if drawing is None or array_field is None:
                continue
            rows, columns = array_field.shape[-2:]
            if image_shape is not None:
                columns = max(1, round(columns * image_shape[0] / rows))
                rows = image_shape[0]
            # A nodata pixel shows a hatch rather than black or the frame, so
            # it reads as missing instead of as a low value or as no tile.
            canvas = _hatched(rows, columns, background)
            key = drawing.key
            key = key if isinstance(key, ColorKey) else None
            if key is not None and drawing.field.has_gaps():
                key.title = f"{key.title or drawing.task_name}  \u25a8 no data"
            tile = Image(canvas, options=self.options).add(drawing.field)
            tiles.append((tile, key))
            titles.append(_field_title(drawing.task_name, drawing.kind))
        return tiles, titles

    def _blend(
        self,
        image: np.ndarray,
        record: "DatasetRecord",
        layers: "LayerState",
        *,
        color_by: ColorBy,
        arrays: "Mapping[str, np.ndarray] | None" = None,
    ) -> Image:
        """Draw the annotations of every task onto one image, layer toggles applied.

        Shared by the interactive single/blended view and the headless save path.
        Detections from all tasks are blended together (a redundant classification
        chip is dropped next to boxes/keypoints/masks), the current ``layers``
        filter what is shown, and box-less metadata is added as hover-free cards
        unless the side panel shows it. The image is returned unsized; the
        caller sets any display size.
        """
        viz = Image(image, options=self.options)
        self._overlay_arrays(viz, arrays or {}, layers)
        detections = record_to_colored_annotations(
            record,
            color_by=color_by,
            options=self.options,
            identity_palette=self.identity_palette,
        )
        for annotation in layers.apply_layers(
            detections, self.options.theme.palette
        ):
            viz.add(annotation)
        if color_by != "instance" and not self.panel:
            for overlay in metadata_annotations(
                chain.from_iterable(record.annotation.values()),
                lone_object_card=True,
            ):
                viz.add(overlay)
        return viz

    def _source_tiles(
        self,
        source_name: str,
        image: np.ndarray,
        arrays: "Mapping[str, np.ndarray]",
        record: "DatasetRecord",
        layers: "LayerState",
        color_by: ColorBy,
        array_tasks: "AbstractSet[str]" = frozenset(),
    ) -> "tuple[list[Renderable], list[str]]":
        """Build the tiles one image source contributes, with their titles.

        Usually one — the source with every task's detections blended onto it.
        A multi-task dataset in the default class-color mode instead gets one
        tile per task, so the tasks stay legible side by side.
        """
        # The fields come from the array labels (see `tiles`), so an array task
        # gets no tile of its own here. A sample without an array of such a
        # task lists it with no detections, which alone looks like a negative
        # image task, so the names of the array labels decide.
        tasks = [
            task
            for task, detections in record.annotation.items()
            if task not in array_tasks
            and (not detections or any(d.array is None for d in detections))
        ]
        # The viewer's interactive layer toggles (masks/keypoints/labels, a class
        # focus) filter what is drawn without disturbing the metadata cards,
        # legend, or panel — `_blend` applies them to the detections.
        if color_by != "class" or self.blend_all or len(tasks) <= 1:
            return [
                self._blend(
                    image, record, layers, color_by=color_by, arrays=arrays
                )
            ], [source_name]
        tiles: list[Renderable] = []
        options = self.options.replace(array_view="off")
        for task_name in tasks:
            # Without its metadata, a task tile gets no panel: it is a plain
            # image whose annotations the layer toggles filter.
            task_record = record.model_copy(
                update={
                    "annotation": {task_name: record.annotation[task_name]},
                    "sample_metadata": {},
                }
            )
            tile = visualize_record(
                task_record, image, options=options, cards=not self.panel
            )
            if isinstance(tile, Image):
                tile.annotations[:] = layers.apply_layers(
                    tile.annotations, self.options.theme.palette
                )
                self._overlay_arrays(tile, arrays, layers)
            tiles.append(tile)
        return tiles, [f"{source_name} · {task}" for task in tasks]

    def tiles(
        self,
        images: "Mapping[str, np.ndarray]",
        arrays: "Mapping[str, np.ndarray]",
        record: "DatasetRecord",
        layers: "LayerState",
        color_by: ColorBy,
    ) -> "tuple[list[Renderable], list[str]]":
        """Build every tile of one sample: its sources, then any array fields.

        Args:
            images: The sample's image sources, keyed by source name.
            arrays: The sample's array labels from the loader, keyed by task
                name.
            record: The sample's record.
            layers: The current layer toggles.
            color_by: What a color identifies in this sample.

        Returns:
            The tiles and their titles. The titles are empty when there is only
            one tile, so the grid draws no title band.

        """
        tiles: list[Renderable] = []
        titles: list[str] = []
        # Source names carry their own placement: a rig storing `right` before
        # `left` should still be drawn left-to-right.
        images = order_by_position(images)
        # The labels and any array field describe the reference view, so they
        # land on exactly one source; the others are shown as they are.
        reference = reference_source(images, self.options.array_overlay_source)
        for source_name, source_image in images.items():
            if source_name != reference:
                tiles.append(
                    Image(source_image.astype(np.uint8), options=self.options)
                )
                titles.append(source_name)
                continue
            source, names = self._source_tiles(
                source_name,
                source_image.astype(np.uint8),
                arrays,
                record,
                layers,
                color_by,
                array_tasks=frozenset(arrays),
            )
            tiles.extend(source)
            titles.extend(names)
        field_tiles, field_titles = self._array_tiles(
            arrays,
            layers,
            image_shape=next(
                (
                    (int(image.shape[0]), int(image.shape[1]))
                    for image in images.values()
                ),
                None,
            ),
        )
        keyed = [(tile, key) for tile, key in field_tiles if key is not None]
        if keyed:
            # Each key sits in a strip under its field, so it covers no data.
            # Every tile gets a strip of one height, so the rows stay aligned
            # and each image starts at the top of its cell.
            strip_h = max(key_strip_height(tile, key) for tile, key in keyed)
            tiles = [with_key_strip(tile, height=strip_h) for tile in tiles]
            tiles.extend(
                with_key_strip(tile, key, height=strip_h)
                for tile, key in field_tiles
            )
        else:
            tiles.extend(tile for tile, _ in field_tiles)
        titles.extend(field_titles)
        # A lone tile needs no title band; several are only distinguishable with
        # one. Titles are dropped rather than blanked so the band collapses.
        return tiles, ([] if len(tiles) == 1 else titles)

    def _save_size(self, width: int, height: int) -> "tuple[int, int] | None":
        """File render size: the configured scale, or native when ``auto``.

        There is no screen to fit to when saving, so ``auto`` keeps the source
        resolution rather than scaling.
        """
        if isinstance(self.scale, str):  # "auto"
            return None
        return (
            max(1, round(width * self.scale)),
            max(1, round(height * self.scale)),
        )

    def _display_size(
        self, width: int, height: int, reserve: float
    ) -> "tuple[int, int] | None":
        """Display size for one source image, or ``None`` to keep native.

        An explicit scale multiplies the source directly; ``auto`` fits it to
        90% of the screen (leaving room for the panel), scaling a small image
        *up* as well as a large one down so it is never shown tiny. ``None``
        means render at the source size.
        """
        if self.scale != "auto":
            scale = self.scale
        elif self.screen is not None:
            avail_w = max(1.0, 0.9 * self.screen[0] - reserve)
            avail_h = max(1.0, 0.9 * self.screen[1])
            scale = min(avail_w / width, avail_h / height)
        else:
            return None
        if scale == 1.0:
            return None
        return (max(1, round(width * scale)), max(1, round(height * scale)))

    def _save_grid(
        self, tiles: "list[Renderable]", titles: "list[str]"
    ) -> Renderable:
        """Grid a sample's tiles for a file, at whatever size the scale implies.

        The headless twin of `_compose`: with no screen to fit to there is
        nothing to shrink toward, so tiles keep their (optionally scaled) size.
        """
        sized = [
            tile.render_at(self._save_size(tile.width, tile.height))
            for tile in tiles
        ]
        named: Mapping[str, Renderable] | Renderable = (
            dict(zip(_unique_titles(titles), sized, strict=True))
            if titles
            else sized[0]
        )
        return combine(named, bg=self.options.theme.background)

    def _compose(
        self, tiles: "list[Renderable]", titles: "list[str]", reserve: float
    ) -> Frame:
        """Hand a sample's tiles to `combine`, which picks the layout and sizing.

        The arrangement is not decided here: `combine` chooses the column count
        that shows the tiles largest inside the screen budget, orders positional
        source names, and drops the grid chrome when there is only one tile.
        """
        if self.scale != "auto":
            tiles = [
                tile.copy().render_at(
                    (
                        max(1, round(tile.width * self.scale)),
                        max(1, round(tile.height * self.scale)),
                    )
                )
                for tile in tiles
            ]
        named: Mapping[str, Renderable] | Renderable = (
            dict(zip(_unique_titles(titles), tiles, strict=True))
            if titles
            else tiles[0]
        )
        if self.scale == "auto" and self.screen is not None:
            return combine(
                named,
                target=(
                    round(0.9 * self.screen[0]),
                    round(0.9 * self.screen[1]),
                ),
                reserve=reserve,
                bg=self.options.theme.background,
                allow_upscale=True,
            ).frame()
        return combine(named, bg=self.options.theme.background).frame()

    def frame(
        self,
        images: "Mapping[str, np.ndarray]",
        arrays: "Mapping[str, np.ndarray]",
        record: "DatasetRecord",
        panel: "Mapping[str, PanelData]",
        layers: "LayerState",
        color_by: ColorBy,
    ) -> Frame:
        """Compose one sample into a display-ready `Frame`.

        The interactive form: tiles are fitted to the screen, and unless the
        panel is off it is always attached — the controls make it non-empty.
        The panel (controls + classes + metadata) reframes the image as a
        rounded surface at a margin offset, so `Frame.with_panel` shifts the
        hover map to match.

        Args:
            images: The sample's image sources, keyed by source name.
            arrays: The sample's array labels from the loader, keyed by task
                name.
            record: The sample's record.
            panel: The sample's metadata panel.
            layers: The current layer toggles.
            color_by: What a color identifies in this sample.

        Returns:
            The composed frame.

        """
        reserve = self.panel_width if self.panel else 0.0
        frame = self._compose(
            *self.tiles(images, arrays, record, layers, color_by), reserve
        )
        if not self.panel:
            return frame
        return frame.with_panel(
            self.sidebar(
                _with_object_metadata(record, panel),
                layers,
                color_by=color_by,
                task_names=tuple(record.annotation),
            )
        )

    def render(
        self,
        images: "Mapping[str, np.ndarray]",
        arrays: "Mapping[str, np.ndarray]",
        record: "DatasetRecord",
        panel: "Mapping[str, PanelData]",
        layers: "LayerState",
        color_by: ColorBy,
    ) -> "Renderable | None":
        """Compose one sample for a file, or ``None`` when it draws nothing.

        Fully headless (no viewer, no screen): the sample's image sources —
        and any array field — are tiled into one render, framed with the
        metadata panel (without the interactive controls) unless the panel is
        off. Draws the same pixels whether the caller writes stills or a clip.

        Args:
            images: The sample's image sources, keyed by source name.
            arrays: The sample's array labels from the loader, keyed by task
                name.
            record: The sample's record.
            panel: The sample's metadata panel.
            layers: The current layer toggles.
            color_by: What a color identifies in this sample.

        Returns:
            The render, or ``None`` when the sample has no tiles.

        """
        tiles, titles = self.tiles(images, arrays, record, layers, color_by)
        if not tiles:
            return None
        if len(tiles) == 1:
            viz: Renderable = tiles[0]
            viz.render_at(self._save_size(viz.width, viz.height))
        else:
            viz = self._save_grid(tiles, titles)
        if self.panel:
            viz = viz.with_panel(
                self.sidebar(
                    _with_object_metadata(record, panel),
                    layers,
                    color_by=color_by,
                    task_names=tuple(record.annotation),
                    controls=False,
                ),
            )
        return viz


def _with_object_metadata(
    record: "DatasetRecord", panel: "Mapping[str, PanelData]"
) -> "dict[str, PanelData]":
    """Put the record's hover-less metadata first in its panel.

    With a side panel, that metadata goes there instead of on a card over the
    image (see `metadata_panel`).
    """
    detections = chain.from_iterable(record.annotation.values())
    return {**metadata_panel(detections, lone_object_card=True), **panel}


def _field_title(task_name: str, kind: str) -> str:
    """Title an array tile with its task and how it is read, saying a word once.

    Examples:
        >>> _field_title("depth", "scalar")
        'depth · scalar'
        >>> _field_title("normals", "normals")
        'normals'

    """
    return task_name if task_name == kind else f"{task_name} \u00b7 {kind}"


#: The ``(height, width)`` of the tile that holds a missing array's slot when
#: the sample has no image to take the size from.
_MISSING_TILE = (240, 320)

#: How far the hatch stripes and the tile border stand off the background.
_HATCH_STRIPE = 22
_HATCH_BORDER = 48


def _hatched(rows: int, columns: int, background: Color) -> np.ndarray:
    """Return an array tile's canvas: a diagonal hatch with a 1 px border.

    The field draws over it, so the hatch shows only where the field has no
    data. The stripes and the border are lighter than a dark background and
    darker than a light one.

    Examples:
        >>> tile = _hatched(4, 6, Color(20, 20, 20))
        >>> tile[0, 0].tolist(), tile[2, 4].tolist()
        ([68, 68, 68], [20, 20, 20])

    """
    base = np.array((background.r, background.g, background.b), np.int16)
    sign = 1 if base.mean() < 128 else -1
    canvas = np.empty((rows, columns, 3), np.int16)
    canvas[:] = base
    ys, xs = np.mgrid[0:rows, 0:columns]
    canvas[(xs + ys) % 12 < 3] = base + sign * _HATCH_STRIPE
    border = base + sign * _HATCH_BORDER
    canvas[[0, -1], :] = border
    canvas[:, [0, -1]] = border
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _unique_titles(titles: Sequence[str]) -> "list[str]":
    """Disambiguate repeated tile titles so none is lost as a mapping key."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for title in titles:
        count = seen.get(title, 0)
        seen[title] = count + 1
        unique.append(title if not count else f"{title} ({count + 1})")
    return unique
