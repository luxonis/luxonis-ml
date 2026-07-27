"""The `Viewer`: an interactive, backend-agnostic image window with hover.

`Viewer` presents finished `Image` frames, fits them to the screen, routes mouse
moves through a `HitMap` to show per-annotation tooltips, and runs the keyboard
loop — all over a pluggable `WindowBackend` (OpenCV by default). Callers hand it
frames and hit maps (see `Image.render_hits`, `luxonis_ml.vizlab.grid_hits`) and
never touch windowing themselves.
"""

from dataclasses import dataclass, field

import numpy as np

from luxonis_ml.vizlab import io
from luxonis_ml.vizlab.hitmap import HitMap
from luxonis_ml.vizlab.image import Image
from luxonis_ml.vizlab.tooltip import Tooltip

from .backend import MouseHandler, WindowBackend
from .cv2_backend import Cv2Backend
from .tooltip_render import draw_tooltip


@dataclass
class _HoverState:
    """Per-window state used to redraw the hover tooltip on mouse-move."""

    base: np.ndarray
    hitmap: HitMap
    hover: Tooltip | None = None
    mouse: tuple[int, int] = field(default=(0, 0))
    dirty: bool = False


def _key_char(key: int) -> str:
    """Map a backend key code to a single character (``""`` when none)."""
    return chr(key & 0xFF) if key != -1 else ""


class Viewer:
    """Show `Image` frames interactively, with screen-fit and hover tooltips.

    Typical use is one frame per named window per step: call `show` for each
    window, then `wait` to block for a keypress while hover tooltips redraw. Use
    `show_blocking` for a single non-interactive frame (no hover), `destroy_stale`
    to close windows no longer in use, and `close` to tear everything down.

    Args:
        backend: The window backend to drive; defaults to `Cv2Backend`.

    """

    def __init__(self, backend: WindowBackend | None = None) -> None:
        self._backend: WindowBackend = (
            backend if backend is not None else Cv2Backend()
        )
        self._screen = self._backend.screen_size()
        self._windows: dict[str, _HoverState] = {}
        self._live: set[str] = set()

    @property
    def screen(self) -> tuple[int, int] | None:
        """The backend's screen size in pixels, or ``None`` if unavailable."""
        return self._screen

    def _frame(
        self, display: Image, hitmap: HitMap
    ) -> tuple[np.ndarray, HitMap]:
        """Render ``display`` to a screen-fitted BGR frame and scale the map.

        The image is rendered once at its natural size to learn its dimensions
        (which may already reflect a `Image.render_at` size); if that overflows
        the screen it is re-rendered smaller — so labels stay crisp rather than
        being resampled afterwards — and the hit map is scaled to match.
        """
        rgba = display.render()
        out_h, out_w = rgba.shape[:2]
        fit = 1.0
        if self._screen is not None:
            fit = min(
                0.9 * self._screen[0] / out_w,
                0.9 * self._screen[1] / out_h,
                1.0,
            )
        if fit < 1.0:
            size = (max(1, round(out_w * fit)), max(1, round(out_h * fit)))
            rgba = display.render(size)
            hitmap = hitmap.scaled(fit)
        return io.export(rgba, "bgr"), hitmap

    def _open(self, name: str, frame: np.ndarray) -> None:
        """Create (if needed), size, and center the window for ``frame``."""
        height, width = frame.shape[:2]
        if name not in self._live:
            self._backend.create_window(name)
            self._live.add(name)
        self._backend.resize(name, width, height)
        if self._screen is not None:
            self._backend.center(name, width, height, self._screen)

    def show(self, name: str, display: Image, hitmap: HitMap) -> None:
        """Present ``display`` in window ``name`` and arm its hover tooltips.

        Args:
            name: The window identifier (created on first use).
            display: The finished frame to show.
            hitmap: Hover regions for ``display`` in its native pixels (from
                `Image.render_hits` / the ``*_hits`` compose helpers); it is
                scaled internally to match the shown frame.

        """
        frame, hitmap = self._frame(display, hitmap)
        self._open(name, frame)
        state = _HoverState(base=frame, hitmap=hitmap)
        self._windows[name] = state
        self._backend.set_mouse_handler(name, self._handler(state))
        self._backend.show(name, frame)

    def show_blocking(self, name: str, display: Image) -> str:
        """Show one frame (no hover) and block until a key; return its char."""
        frame, _ = self._frame(display, HitMap.empty())
        self._open(name, frame)
        self._windows.pop(name, None)
        self._backend.show(name, frame)
        return _key_char(self._backend.poll_key(0))

    def wait(self) -> str:
        """Block for a keypress, redrawing hover tooltips; return its char."""
        while True:
            key = self._backend.poll_key(20)
            if key != -1:
                return _key_char(key)
            for name, state in self._windows.items():
                if not state.dirty:
                    continue
                state.dirty = False
                if state.hover is None:
                    self._backend.show(name, state.base)
                else:
                    frame = state.base.copy()
                    draw_tooltip(frame, state.hover, state.mouse)
                    self._backend.show(name, frame)

    def destroy_stale(self, current: set[str]) -> None:
        """Close every open window whose name is not in ``current``."""
        for name in self._live - current:
            self._backend.destroy_window(name)
            self._windows.pop(name, None)
        self._live &= current

    def close(self) -> None:
        """Destroy all windows and reset the viewer."""
        self._backend.close()
        self._live.clear()
        self._windows.clear()

    def _handler(self, state: _HoverState) -> MouseHandler:
        """Build the mouse-move handler that tracks the hovered tooltip."""

        def handler(x: int, y: int) -> None:
            tooltip = state.hitmap.hit(x, y)
            pos = (int(x), int(y))
            if tooltip is not state.hover or (
                tooltip is not None and pos != state.mouse
            ):
                state.hover = tooltip
                state.mouse = pos
                state.dirty = True

        return handler
