"""Composing whole dataset samples — images, arrays, records — into frames.

`SampleComposer` is the rendering half of dataset inspection: given one
sample's image sources, array labels, `DatasetRecord` trees, and metadata
panel, it builds the tiles, sizes them for a screen or a file, and frames
them with the side panel. The CLI (or any other caller) keeps only the data
side: loading samples, filtering them, and deciding what to do with the
finished frames.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from luxonis_ml.vizlab.interaction import Frame
from luxonis_ml.vizlab.layout.compose import combine, order_by_position
from luxonis_ml.vizlab.layout.panel import Controls, Swatches
from luxonis_ml.vizlab.options import RenderOptions
from luxonis_ml.vizlab.scene.image import Image, Renderable
from luxonis_ml.vizlab.style import Palette

from .arrays import array_annotations
from .instances import (
    ColorBy,
    records_to_colored_annotations,
    spatial_instances,
)
from .ldf import metadata_annotations, visualize_record

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
    panel_width: float = 400.0

    def fallback_color_by(
        self, records: "Mapping[str, DatasetRecord]"
    ) -> ColorBy:
        """Resolve this sample's coloring mode.

        Instance coloring needs spatial annotations to color; a sample with
        none falls back to class colors. The caller can compare the result
        with `color_by` to tell the user about the fallback.
        """
        if self.color_by != "instance" or spatial_instances(records.values()):
            return self.color_by
        return "class"

    def sidebar(
        self,
        panel: "Mapping[str, PanelData]",
        layers: "LayerState",
        *,
        task_names: Sequence[str] = (),
        controls: bool = True,
    ) -> "dict[str, PanelData]":
        """Prepend the CONTROLS and CLASSES sections to a sample's metadata panel.

        The interactive controls and the class-color legend live in the side
        panel (not floated over the image): controls come from ``layers`` (so they
        reflect the current toggles and refresh on every re-render), and the class
        swatches are keyed to the classes present in the sample (``layers.classes``)
        with stable full-dataset palette colors. ``controls=False`` omits the
        controls where they do not apply — currently the non-interactive saved
        renders.
        """
        out: dict[str, PanelData] = {}
        if controls:
            out["controls"] = Controls(
                tuple(
                    (c.key, c.name, c.value, c.active)
                    for c in layers.controls()
                )
            )
        names = layers.classes
        class_palette = self.options.theme.palette
        if self.legend and self.color_by == "class" and names:
            out["classes"] = Swatches(
                tuple((class_palette.color_for(name), name) for name in names),
                disabled=frozenset(layers.hidden),
                # Hold the legend (and panel) width to the dataset's longest class
                # name so it stays put as the per-sample class set changes.
                reserve=self.reserve_class,
            )
        elif self.legend and self.color_by == "task" and task_names:
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
        for drawing in array_annotations(
            arrays,
            options=self.options,
            image_shape=(viz.height, viz.width),
            class_names=self.array_class_names,
        ):
            # Added before the detections so the field paints beneath them.
            for annotation in drawing.annotations():
                viz.add(annotation)

    def _array_tiles(
        self, arrays: "Mapping[str, np.ndarray]", layers: "LayerState"
    ) -> "tuple[list[Renderable], list[str]]":
        """Render each array field as its own tile, in the ``tile`` array view."""
        if (
            self.options.array_view != "tile"
            or not arrays
            or not layers.arrays
        ):
            return [], []
        tiles: list[Renderable] = []
        titles: list[str] = []
        background = self.options.theme.background
        for drawing in array_annotations(
            arrays, options=self.options, class_names=self.array_class_names
        ):
            array_field = drawing.field.field()
            if array_field is None:
                continue
            height, width = array_field.shape[-2:]
            # On the theme background rather than black, so a nodata pixel reads
            # as absent instead of as a low value.
            canvas = np.full(
                (height, width, 3),
                (background.r, background.g, background.b),
                np.uint8,
            )
            tile = Image(canvas, options=self.options)
            for annotation in drawing.annotations():
                tile.add(annotation)
            tiles.append(tile)
            titles.append(f"{drawing.task_name} {drawing.kind}")
        return tiles, titles

    def _blend(
        self,
        image: np.ndarray,
        records: "Mapping[str, DatasetRecord]",
        layers: "LayerState",
        *,
        color_by: ColorBy,
        arrays: "Mapping[str, np.ndarray] | None" = None,
    ) -> Image:
        """Draw every record's annotations onto one image, layer toggles applied.

        Shared by the interactive single/blended view and the headless save path.
        Detections from all tasks are blended together (a redundant classification
        chip is dropped next to boxes/keypoints/masks), the current ``layers``
        filter what is shown, and box-less metadata is added as hover-free cards.
        The image is returned unsized; the caller sets any display size.
        """
        viz = Image(image, options=self.options)
        self._overlay_arrays(viz, arrays or {}, layers)
        detections = records_to_colored_annotations(
            list(records.values()),
            color_by=color_by,
            options=self.options,
            identity_palette=self.identity_palette,
        )
        for annotation in layers.apply_layers(
            detections, self.options.theme.palette
        ):
            viz.add(annotation)
        if color_by != "instance":
            for overlay in metadata_annotations(
                [d for r in records.values() for d in r._annotations()],
                lone_object_card=True,
            ):
                viz.add(overlay)
        return viz

    def _source_tiles(
        self,
        source_name: str,
        image: np.ndarray,
        arrays: "Mapping[str, np.ndarray]",
        records: "Mapping[str, DatasetRecord]",
        layers: "LayerState",
        color_by: ColorBy,
    ) -> "tuple[list[Renderable], list[str]]":
        """Build the tiles one image source contributes, with their titles.

        Usually one — the source with every task's detections blended onto it.
        A multi-task dataset in the default class-color mode instead gets one
        tile per record, so the tasks stay legible side by side.
        """
        # The viewer's interactive layer toggles (masks/keypoints/labels, a class
        # focus) filter what is drawn without disturbing the metadata cards,
        # legend, or panel — `_blend` applies them to the detections.
        if color_by != "class" or self.blend_all or len(records) <= 1:
            return [
                self._blend(
                    image, records, layers, color_by=color_by, arrays=arrays
                )
            ], [source_name]
        tiles: list[Renderable] = []
        for record in records.values():
            tile = visualize_record(record, image, options=self.options)
            # Grid tiles carry no per-record panel here, so they are plain images
            # whose annotations the layer toggles filter.
            if isinstance(tile, Image):
                tile.annotations[:] = layers.apply_layers(
                    tile.annotations, self.options.theme.palette
                )
            tiles.append(tile)
        return tiles, [f"{source_name} · {task}" for task in records]

    def tiles(
        self,
        images: "Mapping[str, np.ndarray]",
        arrays: "Mapping[str, np.ndarray]",
        records: "Mapping[str, DatasetRecord]",
        layers: "LayerState",
        color_by: ColorBy,
    ) -> "tuple[list[Renderable], list[str]]":
        """Build every tile of one sample: its sources, then any array fields."""
        tiles: list[Renderable] = []
        titles: list[str] = []
        # Source names carry their own placement: a rig storing `right` before
        # `left` should still be drawn left-to-right.
        images = order_by_position(images)
        # A field describes the reference view, so an overlay lands on exactly
        # one source; the others are built without it.
        overlay_source = self.options.array_overlay_source or next(
            iter(images), ""
        )
        for source_name, source_image in images.items():
            source, names = self._source_tiles(
                source_name,
                source_image.astype(np.uint8),
                arrays if source_name == overlay_source else {},
                records,
                layers,
                color_by,
            )
            tiles.extend(source)
            titles.extend(names)
        field_tiles, field_titles = self._array_tiles(arrays, layers)
        tiles.extend(field_tiles)
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
        records: "Mapping[str, DatasetRecord]",
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
        """
        reserve = self.panel_width if self.panel else 0.0
        frame = self._compose(
            *self.tiles(images, arrays, records, layers, color_by), reserve
        )
        if not self.panel:
            return frame
        return frame.with_panel(
            self.sidebar(panel, layers, task_names=tuple(records))
        )

    def render(
        self,
        images: "Mapping[str, np.ndarray]",
        arrays: "Mapping[str, np.ndarray]",
        records: "Mapping[str, DatasetRecord]",
        panel: "Mapping[str, PanelData]",
        layers: "LayerState",
        color_by: ColorBy,
    ) -> "Renderable | None":
        """Compose one sample for a file, or ``None`` when it draws nothing.

        Fully headless (no viewer, no screen): the sample's image sources —
        and any array field — are tiled into one render, framed with the
        metadata panel (without the interactive controls) unless the panel is
        off. Draws the same pixels whether the caller writes stills or a clip.
        """
        tiles, titles = self.tiles(images, arrays, records, layers, color_by)
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
                    panel,
                    layers,
                    task_names=tuple(records),
                    controls=False,
                ),
            )
        return viz


def _unique_titles(titles: Sequence[str]) -> "list[str]":
    """Disambiguate repeated tile titles so none is lost as a mapping key."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for title in titles:
        count = seen.get(title, 0)
        seen[title] = count + 1
        unique.append(title if not count else f"{title} ({count + 1})")
    return unique
