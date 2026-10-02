"""Blocking presentation of static, scale-aware renders.

The window half of chart-like commands (``data health``): content with no
hover map or layer state, but drawn from vectors — so it is re-rendered at
the size it will actually occupy on screen instead of downscaling a finished
raster, which would soften every edge it drew.
"""

from collections.abc import Callable

from luxonis_ml.vizlab.scene.image import Renderable

from .backend import WindowBackend
from .cv2_backend import Cv2Backend


def show_fitted(
    window: str,
    render: "Callable[[float], Renderable]",
    *,
    scale: float = 1.0,
    screen: "tuple[int, int] | None" = None,
    backend: "WindowBackend | None" = None,
) -> int:
    """Show ``render(scale)`` fitted to ``screen`` and hold it for a key press.

    Args:
        window: The window identifier (left open on return, so a caller
            presenting a series closes its backend once at the end).
        render: Rebuilds the scene at a given scale. Called again at a
            smaller scale when the first render overflows the screen, so
            charts and text are drawn at the size they are actually shown.
        scale: The nominal scale of the content.
        screen: The ``(width, height)`` to fit into, or ``None`` to present
            at the render's own size.
        backend: The window backend to present through; a fresh `Cv2Backend`
            when not given.

    Returns:
        The pressed key's code, as the backend reports it.

    """
    backend = backend or Cv2Backend()
    image = render(scale)
    if screen is not None:
        # Draw the charts/text at the size they will actually be shown: if the
        # render overflows the screen, re-render it smaller rather than
        # downscaling the finished raster (resampling drawn vector content
        # always softens/aliases it).
        fit = min(
            0.9 * screen[0] / image.width,
            0.9 * screen[1] / image.height,
            1.0,
        )
        if fit < 0.98:
            image = render(scale * fit)
    out = image.to_numpy("bgr")
    if screen is not None:
        # If the render is still larger than the screen, shrink it ourselves
        # with a high-quality area filter and show 1:1. Letting OpenCV's
        # WINDOW_NORMAL scale the full-size raster instead uses a crude filter
        # that re-aliases the smooth chart edges.
        import cv2

        out_h, out_w = out.shape[:2]
        fit = min(0.9 * screen[0] / out_w, 0.9 * screen[1] / out_h, 1.0)
        if fit < 1.0:
            out = cv2.resize(
                out,
                (max(1, round(out_w * fit)), max(1, round(out_h * fit))),
                interpolation=cv2.INTER_AREA,
            )
    out_h, out_w = out.shape[:2]
    backend.create_window(window)
    backend.resize(window, out_w, out_h)
    if screen is not None:
        backend.center(window, out_w, out_h, screen)
    backend.show(window, out)
    return backend.poll_key(0)
