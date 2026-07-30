"""A `Frame`: a displayable scene paired with its interaction maps.

Every `Renderable` captures hover and click regions through the same scene graph
that places its pixels. `Renderable.frame` snapshots those regions alongside the
scene, so a `Viewer` receives one typed value instead of loose maps that are easy
to mismatch. Composition preserves the regions, so a grid or panel yields the
same value as a single image does.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from luxonis_ml.vizlab.render import RenderEnvironment
from luxonis_ml.vizlab.render.capture import ClickMap, HitMap
from luxonis_ml.vizlab.scene.image import Renderable

if TYPE_CHECKING:
    from luxonis_ml.vizlab.color import ColorLike
    from luxonis_ml.vizlab.layout.panel import PanelData
    from luxonis_ml.vizlab.style import Style


@dataclass(frozen=True)
class Frame:
    """An `Image` together with the `HitMap` for its hover regions.

    The hit map is expressed in the image's native `Image.render` pixels; a
    `Viewer` scales it to match when it screen-fits the frame.

    Attributes:
        image: The scene to display — an `Image` or a composed `Renderable`.
        hitmap: Hover regions in the image's native render pixels; defaults to an
            empty map (no hover).
        clickmap: Clickable regions (panel controls / legend swatches) in native
            render pixels, each with an action string a `Viewer` dispatches;
            defaults to an empty map (nothing clickable).
        environment: Ambient style snapshot used when the frame captured its
            interaction maps. Manually constructed frames leave it unset.

    """

    image: Renderable
    hitmap: HitMap = field(default_factory=HitMap.empty)
    clickmap: ClickMap = field(default_factory=ClickMap.empty)
    environment: RenderEnvironment | None = None

    @classmethod
    def capture(
        cls, scene: Renderable, size: tuple[int, int] | None = None
    ) -> "Frame":
        """Draw ``scene`` for its interaction regions and pair them with it.

        Args:
            scene: The scene to capture.
            size: ``(width, height)`` to capture at; ``None`` uses the scene's
                `Renderable.render_at` size.

        Returns:
            A `Frame` a `Viewer` can display.

        """
        interactions, environment = scene.capture(size)
        return cls(
            scene,
            HitMap(interactions.hover),
            ClickMap(interactions.clicks),
            environment,
        )

    def render(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Render with the same style snapshot as the interaction maps."""
        if self.environment is not None:
            return self.image._render_in(self.environment, size)
        return self.image.render(size)

    def with_image(self, image: Renderable) -> "Frame":
        """Return a copy carrying ``image`` but this frame's interaction maps.

        For coordinate-preserving changes only: an overlay (e.g. a `Legend`) drawn
        on top leaves the existing hover and click rectangles valid, so both maps
        are reused as-is. To attach a side panel (which reframes the image at an
        offset) use `with_panel`, which shifts the map to match.
        """
        return Frame(image, self.hitmap, self.clickmap, self.environment)

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
        from luxonis_ml.vizlab.layout.panel import compose_panel

        image, (dx, dy), clicks = compose_panel(
            self.image,
            data,
            side=side,
            width=width,
            title=title,
            style=style,
            bg=bg,
        )
        return Frame(
            image,
            self.hitmap.offset(dx, dy),
            self.clickmap.offset(dx, dy).merge(ClickMap(clicks)),
            self.environment,
        )
