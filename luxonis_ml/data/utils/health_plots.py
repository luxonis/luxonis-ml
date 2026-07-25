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

from luxonis_ml.vizlab import (
    Caption,
    ClassDistribution,
    Color,
    Corner,
    Heatmap,
    Image,
    Theme,
    get_default_theme,
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
    return Image(_panel_bg(width, height, theme.background), theme=theme).add(
        Caption(text=text, corner=Corner.TOP_LEFT)
    )


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
        theme=theme,
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
    return Image(_panel_bg(side, side, theme.background), theme=theme).add(
        Heatmap(
            values=values,
            gradient=gradient,
            weight_by_value=False,
            vmin=0.0,
            alpha=1.0,
        )
    )


def build_health_grid(
    task_name: str,
    class_dist_by_type: Mapping[str, list[dict[str, Any]]],
    heatmaps_by_type: Mapping[str, Sequence[Sequence[float]] | None],
    *,
    theme: Theme | None = None,
    gradient: str = "viridis",
    mode: DistributionMode = "bars",
    scale: float = 1.0,
) -> Image:
    """Compose the class-distribution and heatmap panels for one task name.

    For each task type a distribution panel and a heatmap panel are placed side by
    side (two columns), titled ``"<task>/<type> — classes"`` and ``"… — heatmap"``.

    Args:
        task_name: The task name (used as a title prefix; empty for the default).
        class_dist_by_type: Class counts per task type.
        heatmaps_by_type: Density matrices per task type (``None`` when absent).
        theme: Theme for the whole grid; ``None`` uses the process default.
        gradient: Name of the heatmap colormap.
        mode: How each class distribution is drawn.
        scale: User font/mark multiplier on top of the nominal plot scale.

    Returns:
        A single titled grid `Image` for the task name.

    """
    theme = theme if theme is not None else get_default_theme()
    theme = Theme(
        style=theme.style.scaled(_BASE_SCALE * scale),
        palette=theme.palette,
        background=theme.background,
    )
    task_types = sorted(set(class_dist_by_type) | set(heatmaps_by_type))
    prefix = f"{task_name}/" if task_name else ""
    images: list[Image] = []
    titles: list[str] = []
    for task_type in task_types:
        distribution = _distribution_panel(
            class_dist_by_type.get(task_type, []), theme=theme, mode=mode
        )
        side = int(min(_MAX_SIDE, max(_MIN_SIDE, distribution.height)))
        images.append(distribution)
        titles.append(f"{prefix}{task_type} — classes")
        images.append(
            _heatmap_panel(
                heatmaps_by_type.get(task_type),
                side,
                theme=theme,
                gradient=gradient,
            )
        )
        titles.append(f"{prefix}{task_type} — heatmap")
    return grid(
        images,
        ncols=2,
        titles=titles,
        bg=theme.background,
        style=theme.style,
    )
