"""A `Frame`: a displayable `Image` paired with its hover `HitMap`.

The composition helpers that thread hover regions (`grid_hits`, `combine_hits`,
`fit_grid`) and `Image.frame` all return a `Frame`, so an image and its hit map
travel together as one typed value instead of a loose ``(image, hitmap)`` tuple
that is easy to mismatch. A `Viewer` shows a `Frame` directly.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .hitmap import ClickMap, HitMap
from .image import Image

if TYPE_CHECKING:
    from .color import ColorLike
    from .panel import PanelData
    from .style import Style


@dataclass(frozen=True)
class Frame:
    """An `Image` together with the `HitMap` for its hover regions.

    The hit map is expressed in the image's native `Image.render` pixels; a
    `Viewer` scales it to match when it screen-fits the frame.

    Attributes:
        image: The image to display.
        hitmap: Hover regions in the image's native render pixels; defaults to an
            empty map (no hover).
        clickmap: Clickable regions (panel controls / legend swatches) in native
            render pixels, each with an action string a `Viewer` dispatches;
            defaults to an empty map (nothing clickable).

    """

    image: Image
    hitmap: HitMap = field(default_factory=HitMap.empty)
    clickmap: ClickMap = field(default_factory=ClickMap.empty)

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Render the underlying image (see `Image.render`)."""
        return self.image.render(size)

    def with_image(self, image: Image) -> "Frame":
        """Return a copy carrying ``image`` but this frame's hit map.

        For coordinate-preserving changes only: an overlay (e.g. a `Legend`) drawn
        on top leaves the existing hit rectangles valid, so the map is reused
        as-is. To attach a side panel (which reframes the image at an offset) use
        `with_panel`, which shifts the map to match.
        """
        return Frame(image, self.hitmap)

    def with_panel(
        self,
        data: "PanelData",
        *,
        side: str = "right",
        width: float | None = None,
        title: str | None = None,
        style: "Style | None" = None,
        bg: "ColorLike | None" = None,
    ) -> "Frame":
        """Attach a metadata panel and keep the hover map aligned.

        Frames the image and the panel as separate rounded surfaces (see
        `Image.with_panel`). Because that reframes the image at a margin offset,
        the hit map is translated by the same offset so hover stays correct.

        Args:
            data: JSON-like metadata to render in the panel.
            side: Which edge to attach the panel to (``"right"`` default).
            width: Panel width in pixels, or ``None`` to auto-size.
            title: Optional bold heading above the panel content.
            style: Style whose font is used (defaults to the library default).
            bg: Panel/composite background; defaults to the image's theme.

        Returns:
            A new `Frame` of the framed image plus panel, with the hit map shifted
            to the image's new position and a click map for the panel's controls
            and legend swatches.

        """
        from .panel import _compose_panel

        image, (dx, dy), clicks = _compose_panel(
            self.image,
            data,
            side=side,
            width=width,
            title=title,
            style=style,
            bg=bg,
        )
        return Frame(image, self.hitmap.offset(dx, dy), ClickMap(clicks))
