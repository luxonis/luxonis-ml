"""Vizlab charts for ``luxonis_ml data health``.

Renders, per task type, the class-distribution bar panel and the spatial
annotation-density heatmap that ``data health`` shows, composed into one titled
grid per task name. This replaces the previous matplotlib figures with vizlab's
`ClassDistribution` and `Heatmap` visualizers.

The look is configurable: a `Theme` (dark/light, optionally scaled), the heatmap
gradient, and the class-distribution mode all thread through from the CLI.

Importing this module pulls in vizlab, so it requires the ``viz`` extra
(``pip install luxonis-ml[viz]``); ``data health`` imports it lazily and reports
that hint if the extra is missing.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from luxonis_ml.data.utils.data_utils import HEATMAP_TASK_TYPES
from luxonis_ml.vizlab import (
    Caption,
    ClassDistribution,
    Color,
    Corner,
    Heatmap,
    Image,
    Renderable,
    RenderOptions,
    Theme,
    current_options,
    escape,
    grid,
)

DistributionMode = Literal["bars", "chips", "stacked", "pie", "donut"]
"""Class-distribution looks offered by ``data health``."""

#: Nominal font/mark scale for the plots. The base vizlab style (~16 px text) is
#: small on a large display, so ``data health`` renders larger by default; the
#: CLI ``--scale`` flag multiplies this.
_BASE_SCALE = 1.75
_MARGIN = 16.0
_MIN_SIDE = 300
_MAX_SIDE = 620

#: Human-readable descriptor for each panel, shown as the panel heading over its
#: task type. This lets the two columns be told apart at a glance without
#: repeating the task name already shown in the window title.
_CLASSES_DESC = "Class distribution"
_HEATMAP_DESC = "Spatial density"


def _panel_title(task_type: str, descriptor: str) -> str:
    """Build a styled two-line panel title: a heading over the task type.

    The human ``descriptor`` is drawn bold as the heading; the task type sits
    beneath it in monospace, giving the panels a proper title/subtitle look
    instead of two words joined by a dash. The task name belongs in the
    containing window title, so it is deliberately not repeated here.
    Uses vizlab title markup (see `luxonis_ml.vizlab.layout.compose.grid`).
    """
    return (
        f"<b>{escape(descriptor)}</b>\n<code>{escape(task_type)}</code>"
        if task_type
        else escape(descriptor)
    )


def _panel_bg(width: float, height: float, color: Color) -> np.ndarray:
    """Return a solid ``(H, W, 3)`` background of the given size and color."""
    return np.full(
        (int(height), int(width), 3),
        (color.r, color.g, color.b),
        dtype=np.uint8,
    )


def _placeholder(
    text: str, *, theme: Theme, width: int = 240, height: int = 160
) -> Image:
    """Build a small themed panel with a muted caption, for missing data."""
    return Image(
        _panel_bg(width, height, theme.background),
        options=RenderOptions(theme=theme),
    ).add(Caption(text=text, corner=Corner.TOP_LEFT))


def _distribution_panel(
    task_data: list[dict[str, Any]],
    *,
    theme: Theme,
    mode: DistributionMode,
) -> Image:
    """Render one task type's class counts as a `ClassDistribution` panel.

    Args:
        task_data: ``[{"class_name": str, "count": int}, ...]`` for one task type.
        theme: The theme supplying style, palette, and background.
        mode: How the distribution is drawn (``"bars"``/``"chips"``/``"stacked"``).

    Returns:
        A standalone `Image` sized to the chart, or a placeholder when empty.

    """
    if not task_data:
        return _placeholder("no class data", theme=theme)
    pairs = [
        (str(row["class_name"]), float(row["count"])) for row in task_data
    ]
    dist = ClassDistribution(
        probabilities=pairs,
        value_format="count+percent",
        top_k=None,
        mode=mode,
        corner=Corner.TOP_LEFT,
        margin=_MARGIN,
    )
    width, height = dist.content_size(theme=theme)
    return Image(
        _panel_bg(width + 2 * _MARGIN, height + 2 * _MARGIN, theme.background),
        options=RenderOptions(theme=theme),
    ).add(dist)


def _heatmap_panel(
    matrix: Sequence[Sequence[float]] | None,
    side: int,
    *,
    theme: Theme,
    gradient: str,
) -> Image:
    """Render one task type's density matrix as a solid `Heatmap` panel.

    Args:
        matrix: The square density grid (counts per cell), or ``None``.
        side: The panel's side length in pixels.
        theme: The theme supplying the panel background.
        gradient: Name of the heatmap colormap (e.g. ``"viridis"``).

    Returns:
        A square `Image` of the colormapped density, or a placeholder when
        ``matrix`` is ``None``.

    """
    if matrix is None:
        return _placeholder("no heatmap", theme=theme, width=side, height=side)
    values = np.asarray(matrix, dtype=float)
    return Image(
        _panel_bg(side, side, theme.background),
        options=RenderOptions(theme=theme),
    ).add(
        Heatmap(
            values=values,
            gradient=gradient,
            weight_by_value=False,
            vmin=0.0,
            alpha=1.0,
        )
    )


def build_health_grid(
    class_dist_by_type: Mapping[str, list[dict[str, Any]]],
    heatmaps_by_type: Mapping[str, Sequence[Sequence[float]] | None],
    *,
    theme: Theme | None = None,
    gradient: str = "viridis",
    mode: DistributionMode = "bars",
    scale: float = 1.0,
) -> Renderable:
    """Compose class-distribution and heatmap panels for one task.

    For each task type a distribution panel and a heatmap panel are placed side by
    side (two columns), titled with the task type and either ``"classes"`` or
    ``"heatmap"``. Non-spatial task types (such as metadata) are omitted; a
    spatial one is kept even when it has no heatmap, so its class distribution
    is not lost.

    Args:
        class_dist_by_type: Class counts per task type.
        heatmaps_by_type: Density matrices per task type (``None`` when absent).
        theme: Theme for the whole grid; ``None`` uses the process default.
        gradient: Name of the heatmap colormap.
        mode: How each class distribution is drawn.
        scale: User font/mark multiplier on top of the nominal plot scale.

    Returns:
        A single renderable grid.

    """
    theme = theme if theme is not None else current_options().theme
    theme = theme.with_style(theme.style.scaled(_BASE_SCALE * scale))
    # Only annotations with a spatial representation are plotted. Class
    # distributions may also include metadata, which must not create a
    # placeholder plot in the health view. Keying off the task type rather than
    # off the presence of a heatmap keeps the class distribution of a spatial
    # task whose annotations happened to yield no heatmap points (e.g. keypoints
    # that are all invisible).
    task_types = sorted(
        (set(class_dist_by_type) | set(heatmaps_by_type)) & HEATMAP_TASK_TYPES
    )
    images: list[Renderable] = []
    titles: list[str] = []
    for task_type in task_types:
        distribution = _distribution_panel(
            class_dist_by_type.get(task_type, []), theme=theme, mode=mode
        )
        side = int(min(_MAX_SIDE, max(_MIN_SIDE, distribution.height)))
        images.append(distribution)
        titles.append(_panel_title(task_type, _CLASSES_DESC))
        images.append(
            _heatmap_panel(
                heatmaps_by_type.get(task_type),
                side,
                theme=theme,
                gradient=gradient,
            )
        )
        titles.append(_panel_title(task_type, _HEATMAP_DESC))
    # Each task type contributes a distribution+heatmap pair (two cells). With
    # several task types a single pair-per-row column grows very tall and must be
    # shrunk to fit the screen — which shrinks the titles too. Pack two pairs per
    # row past a couple of task types so the grid stays wide and needs far less
    # shrinking, keeping titles readable.
    ncols = 4 if len(task_types) > 2 else 2
    return grid(
        images,
        ncols=ncols,
        titles=titles,
        bg=theme.background,
        style=theme.style,
    )
