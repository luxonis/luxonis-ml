"""Render a `ComparisonReport`'s confusion matrix as a labelled heat map."""

from typing import TYPE_CHECKING

import numpy as np

from luxonis_ml.vizlab.annotations import BBox
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.geometry import XY
from luxonis_ml.vizlab.options import RenderOptions

from .match import FP_COLOR, NONE_LABEL, TP_COLOR
from .report import ComparisonReport

if TYPE_CHECKING:
    from luxonis_ml.ldf import Detection
    from luxonis_ml.vizlab.render.canvas import Canvas
    from luxonis_ml.vizlab.scene.image import Image

    #: A matchable detection: a vizlab box or a full LDF detection tree.
    Detectionish = BBox | Detection

#: Verdict colors, tuned for the dark composite background. Themeable later.

_WHITE = Color(255, 255, 255)


_CM_TEXT = Color(202, 212, 226)


def _draw_text(
    canvas: "Canvas",
    text: str,
    x: float,
    y: float,
    size: float,
    color: Color,
    *,
    anchor: str = "center",
    weight: int = 600,
) -> None:
    """Draw ``text`` at ``(x, y)`` anchored center/left/right and vertically mid.

    ``text`` may carry inline markup; it is measured and drawn as such.
    """
    metrics = canvas.measure_markup(text, size, weight=weight)
    if anchor == "center":
        tx = x - metrics.width / 2
    elif anchor == "right":
        tx = x - metrics.width
    else:
        tx = x
    ty = y + (metrics.ascent - metrics.descent) / 2
    canvas.markup((tx, ty), text, size=size, color=color, weight=weight)


def _draw_confusion_cell(
    canvas: "Canvas",
    *,
    row: int,
    column: int,
    count: int,
    correct: bool,
    peak: int,
    left: int,
    top: int,
    cell: int,
    grid: Color,
) -> None:
    """Draw one shaded confusion-matrix cell and its count."""
    x0, y0 = float(left + column * cell), float(top + row * cell)
    x1, y1 = x0 + cell, y0 + cell
    corners: list[XY] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    fill = (
        (TP_COLOR if correct else FP_COLOR).with_alpha(
            0.18 + 0.72 * (count / peak)
        )
        if count
        else None
    )
    canvas.polygon(corners, fill=fill, stroke=grid, stroke_width=1.0)
    if not count:
        return
    _draw_text(
        canvas,
        str(count),
        x0 + cell / 2,
        y0 + cell / 2,
        13.0,
        _WHITE,
        weight=700,
    )


def confusion_matrix_figure(
    report: ComparisonReport,
    *,
    options: RenderOptions | None = None,
    cell: int = 60,
) -> "Image":
    """Render ``report``'s confusion matrix as a labeled heatmap `Image`.

    Rows are the ground-truth class, columns the predicted class, with a trailing
    ``∅`` row/column for false negatives (missed) and false positives (invented).
    A cell on the diagonal is green (correct), off-diagonal red (a confusion),
    each shaded by how many detections land in it, with the count printed inside.

    Args:
        report: The accumulated `ComparisonReport`.
        options: Render options, for the background/theme; a default is used when
            ``None``.
        cell: Cell size in pixels.

    Returns:
        A new `Image` of the matrix.

    """
    from luxonis_ml.vizlab.render.canvas import Canvas
    from luxonis_ml.vizlab.scene.image import Image

    options = options or RenderOptions()
    labels, matrix = report.confusion_matrix()
    n = len(labels)
    shown = [lbl if len(lbl) <= 9 else lbl[:8] + "…" for lbl in labels]
    left, top, pad, foot = 100, 70, 16, 28
    width = left + n * cell + pad
    height = top + n * cell + pad + foot

    bg = options.theme.background
    raster = np.empty((height, width, 4), np.uint8)
    raster[:] = (bg.r, bg.g, bg.b, 255)
    canvas = Canvas.from_rgba(raster)

    peak = max((c for row in matrix for c in row), default=0) or 1
    grid = Color(255, 255, 255, 26)
    for i in range(n):
        for j in range(n):
            count = matrix[i][j]
            _draw_confusion_cell(
                canvas,
                row=i,
                column=j,
                count=count,
                correct=i == j and labels[i] != NONE_LABEL,
                peak=peak,
                left=left,
                top=top,
                cell=cell,
                grid=grid,
            )

    for j, name in enumerate(shown):
        _draw_text(
            canvas, name, left + j * cell + cell / 2, top - 14, 12.0, _CM_TEXT
        )
    for i, name in enumerate(shown):
        _draw_text(
            canvas,
            name,
            left - 10,
            top + i * cell + cell / 2,
            12.0,
            _CM_TEXT,
            anchor="right",
        )
    _draw_text(
        canvas,
        "truth ↓  /  prediction →",
        8,
        22,
        13.0,
        _CM_TEXT,
        anchor="left",
        weight=700,
    )
    return Image(canvas.to_rgba())
