"""Composing whole compared samples into display- or file-ready frames.

`ComparisonComposer` is the presentation half of dataset comparison: it
matches one paired sample's records, draws every image source in the chosen
layout, tiles the sources, and attaches the metrics panel. The caller keeps
the data side — pairing samples across loaders and deciding what to do with
the finished frames.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from luxonis_ml.vizlab.interaction import Frame
from luxonis_ml.vizlab.layout.compose import combine
from luxonis_ml.vizlab.options import RenderOptions
from luxonis_ml.vizlab.scene.image import Image, Renderable

from .match import ComparisonResult, Verdict, match_detections
from .render import compare

if TYPE_CHECKING:
    from luxonis_ml.ldf import DatasetRecord
    from luxonis_ml.vizlab.annotations import Legend
    from luxonis_ml.vizlab.layout.panel import PanelData


@dataclass(frozen=True)
class ComparisonComposer:
    """Match and draw paired dataset samples, one policy for a whole session.

    The composer carries the matching thresholds and the presentation choices
    — layout, verdict filter, legend, panel, sizing — so a caller hands it
    only each sample's images and both sides' records. `match` scores a pair
    without drawing (the summary path); `frame` scores and draws it.

    Attributes:
        options: The render options; their theme carries the class palette
            and the background composites are mounted on.
        iou_threshold: Overlap threshold for a localized match.
        score_threshold: Confidence cutoff for predictions.
        class_aware: Whether only same-class boxes are true positives (and
            mismatched labels surface as class errors).
        show: The comparison layout every frame is drawn in.
        verdicts: Draw only these verdicts, or ``None`` for all of them. The
            metrics panel reflects every match either way.
        legend: A class-color legend overlaid on each frame, or ``None``.
        per_class: Add a per-class precision/recall breakdown to the panel.
        panel: Whether frames get the metrics side panel at all.
        scale: Display scale; ``"auto"`` fits the configured screen and falls
            back to the source resolution without one.
        screen: The ``(width, height)`` to fit frames into, or ``None`` when
            there is no screen.
        panel_width: Horizontal room reserved for the metrics panel when
            fitting a frame to the screen.

    """

    options: RenderOptions
    iou_threshold: float = 0.5
    score_threshold: float = 0.25
    class_aware: bool = True
    show: Literal["overlay", "side_by_side", "triptych"] = "overlay"
    verdicts: "frozenset[Verdict] | set[Verdict] | None" = None
    legend: "Legend | None" = None
    per_class: bool = False
    panel: bool = True
    scale: 'float | Literal["auto"]' = "auto"
    screen: "tuple[int, int] | None" = None
    panel_width: float = 400.0

    def match(
        self,
        gt_records: "Mapping[str, DatasetRecord]",
        pred_records: "Mapping[str, DatasetRecord]",
    ) -> ComparisonResult:
        """Match one paired sample's detections, without drawing anything."""
        gt_dets = [d for r in gt_records.values() for d in r._annotations()]
        pred_dets = [
            d for r in pred_records.values() for d in r._annotations()
        ]
        return match_detections(
            gt_dets,
            pred_dets,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            class_aware=self.class_aware,
        )

    def _display_size(
        self, width: int, height: int
    ) -> "tuple[int, int] | None":
        """Fit ``width`` x ``height`` to the screen (or apply the scale).

        There is no screen to fit to when saving, so ``auto`` falls through to
        ``None`` and keeps the source resolution.
        """
        reserve = self.panel_width if self.panel else 0.0
        if self.scale != "auto":
            scale = self.scale
        elif self.screen is not None:
            avail_w = max(1.0, 0.9 * self.screen[0] - reserve)
            avail_h = max(1.0, 0.9 * self.screen[1])
            scale = min(avail_w / width, avail_h / height, 1.0)
        else:
            return None
        if scale == 1.0:
            return None
        return (max(1, round(width * scale)), max(1, round(height * scale)))

    def _scene(
        self, image: np.ndarray, result: ComparisonResult
    ) -> Renderable:
        """Draw one image source's verdict scene, without the metrics panel."""
        viz = compare(
            image,
            result=result,
            options=self.options,
            show=self.show,
            panel=False,
            verdicts=self.verdicts,
        )
        # panel=False, so viz is the plain comparison scene (an image for the
        # overlay layout, a grid composite for side-by-side / triptych).
        viz.render_at(self._display_size(viz.width, viz.height))
        frame = viz.frame()
        display = frame.image
        if self.legend is not None:
            # Overlay the class legend; bake a grid composite to an image first
            # so it carries a mutable annotation list to `add` onto.
            if not isinstance(display, Image):
                display = (
                    Image(frame.render())
                    .with_hitmap(frame.hitmap)
                    .with_pickmap(frame.pickmap)
                )
            display.add(self.legend.model_copy())
        return display

    def frame(
        self,
        images: "Mapping[str, np.ndarray]",
        gt_records: "Mapping[str, DatasetRecord]",
        pred_records: "Mapping[str, DatasetRecord]",
    ) -> Frame:
        """Match GT vs predictions and compose every source into one `Frame`.

        A multi-source sample is tiled rather than given a window per source,
        so the metrics panel is attached once and the sources stay side by side.
        """
        result = self.match(gt_records, pred_records)
        scenes = [
            self._scene(image.astype(np.uint8), result)
            for image in images.values()
        ]
        # One source or several, vizlab decides the arrangement -- and orders
        # positional source names, so a stereo pair stays left-then-right.
        composed: Renderable = combine(
            dict(zip(images, scenes, strict=True))
            if len(scenes) > 1
            else scenes[0],
            bg=self.options.theme.background,
        )
        if not self.panel:
            return composed.frame()
        metrics: dict[str, PanelData] = dict(result.summary())
        if self.per_class and len(result.per_class) > 1:
            metrics["by class"] = result.per_class_panel()
        return composed.with_panel(metrics, title="Comparison").frame()
