"""A `Frame`: a displayable `Image` paired with its hover `HitMap`.

The composition helpers that thread hover regions (`grid_hits`, `combine_hits`,
`fit_grid`) and `Image.frame` all return a `Frame`, so an image and its hit map
travel together as one typed value instead of a loose ``(image, hitmap)`` tuple
that is easy to mismatch. A `Viewer` shows a `Frame` directly.
"""

from dataclasses import dataclass, field

import numpy as np

from .hitmap import HitMap
from .image import Image


@dataclass(frozen=True)
class Frame:
    """An `Image` together with the `HitMap` for its hover regions.

    The hit map is expressed in the image's native `Image.render` pixels; a
    `Viewer` scales it to match when it screen-fits the frame.

    Attributes:
        image: The image to display.
        hitmap: Hover regions in the image's native render pixels; defaults to an
            empty map (no hover).

    """

    image: Image
    hitmap: HitMap = field(default_factory=HitMap.empty)

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Render the underlying image (see `Image.render`)."""
        return self.image.render(size)

    def with_image(self, image: Image) -> "Frame":
        """Return a copy carrying ``image`` but this frame's hit map.

        For coordinate-preserving changes only: overlays (e.g. a `Legend`) and a
        right/bottom side panel (see `Image.with_panel`) leave the existing hit
        rectangles valid, so the map can be reused as-is.
        """
        return Frame(image, self.hitmap)
