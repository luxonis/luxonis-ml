"""The `Viewer`: an interactive, backend-agnostic image window with hover.

`Viewer` presents `Frame` objects (an `Image` plus its hover `HitMap`), fits them
to the screen, routes mouse moves through the map to show per-annotation
tooltips, and runs the keyboard loop — all over a pluggable `WindowBackend`
(OpenCV by default). Callers build frames with `Image.frame` or the ``*_hits``
compose helpers (`luxonis_ml.vizlab.grid_hits`) and never touch windowing
themselves.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from luxonis_ml.vizlab import io
from luxonis_ml.vizlab.frame import Frame
from luxonis_ml.vizlab.hitmap import HitMap
from luxonis_ml.vizlab.image import Image
from luxonis_ml.vizlab.tooltip import Tooltip

from .backend import MouseHandler, WindowBackend
from .cv2_backend import Cv2Backend
from .layers import LayerState
from .tooltip_render import (
    blit_rgba_on_bgr,
    draw_tooltip,
    render_tooltip_card,
)

#: A per-window callback that re-renders the window's `Frame` for a `LayerState`.
RenderFn = Callable[[LayerState], Frame]

#: Type size and inset (px) of the controls HUD drawn on interactive windows.
_HUD_SIZE = 13
_HUD_INSET = 10


@dataclass
class _HoverState:
    """Per-window state used to redraw the hover tooltip on mouse-move.

    ``render``, when set, makes the window *interactive*: layer-control keys
    re-render it through this callback with the viewer's current `LayerState`.
    """

    base: np.ndarray
    hitmap: HitMap
    hover: Tooltip | None = None
    mouse: tuple[int, int] = field(default=(0, 0))
    dirty: bool = False
    render: RenderFn | None = None


def _key_char(key: int) -> str:
    """Map a backend key code to a single character (``""`` when none)."""
    return chr(key & 0xFF) if key != -1 else ""


class Viewer:
    """Show `Image` frames interactively, with screen-fit and hover tooltips.

    Two interaction models, matching the backend flavor (see `WindowBackend`):

    - **pull** (the default `Cv2Backend`): call `show` for each window, then
      `wait` to block for a keypress while hover tooltips redraw; `show_blocking`
      shows a single non-interactive frame.
    - **push** (event-driven backends): call `show`, then `run` once with an
      ``on_key`` callback and return to the backend's own loop; hover tooltips
      then redraw inline on each mouse move.

    Use `destroy_stale` to close windows no longer in use and `close` to tear
    everything down.

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
        # Push (event-driven) backends redraw hover inline instead of in a loop.
        self._driven = False
        # Shared layer toggles for every interactive window this viewer shows.
        self._layers = LayerState()

    @property
    def screen(self) -> tuple[int, int] | None:
        """The backend's screen size in pixels, or ``None`` if unavailable."""
        return self._screen

    @property
    def layers(self) -> LayerState:
        """The shared `LayerState` driving interactive windows.

        Configure it before showing frames — e.g. set ``layers.classes`` so the
        ``c`` key can cycle a class focus — and read it back to see the current
        toggles.
        """
        return self._layers

    def _prepare(
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

    def show(
        self, name: str, frame: Frame, *, render: RenderFn | None = None
    ) -> None:
        """Present ``frame`` in window ``name`` and arm its hover tooltips.

        Args:
            name: The window identifier (created on first use).
            frame: The `Frame` to show — an image plus its hover `HitMap` (from
                `Image.frame` or the ``*_hits`` compose helpers). The map is in
                the image's native pixels and is scaled internally to match the
                shown frame.
            render: Optional callback that rebuilds this window's `Frame` for a
                `LayerState`. When given, the window becomes interactive: the
                layer-control keys (see `layers`) re-render it through this
                callback and a small controls HUD is drawn on it.

        """
        bgr, hitmap = self._prepare(frame.image, frame.hitmap)
        self._open(name, bgr)
        if render is not None:
            self._draw_hud(bgr)
        state = _HoverState(base=bgr, hitmap=hitmap, render=render)
        self._windows[name] = state
        self._backend.set_mouse_handler(name, self._handler(name, state))
        self._backend.show(name, bgr)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Draw the controls HUD (current `LayerState`) at the frame's lower-left."""
        card = render_tooltip_card(self._layers.hud(), _HUD_SIZE)
        y = frame.shape[0] - card.shape[0] - _HUD_INSET
        blit_rgba_on_bgr(frame, card, _HUD_INSET, y)

    def _controllable(self) -> bool:
        """Whether any open window has a re-render callback (is interactive)."""
        return any(s.render is not None for s in self._windows.values())

    def _rerender(self, name: str, state: _HoverState) -> None:
        """Rebuild ``state``'s frame for the current layers and repaint it."""
        if state.render is None:
            return
        frame = state.render(self._layers)
        bgr, hitmap = self._prepare(frame.image, frame.hitmap)
        self._draw_hud(bgr)
        state.base = bgr
        state.hitmap = hitmap
        state.hover = None
        state.dirty = False
        self._backend.show(name, bgr)

    def show_blocking(self, name: str, display: Image) -> str:
        """Show one frame (no hover) and block until a key; return its char."""
        bgr, _ = self._prepare(display, HitMap.empty())
        self._open(name, bgr)
        self._windows.pop(name, None)
        self._backend.show(name, bgr)
        return _key_char(self._backend.poll_key(0))

    def wait(self) -> str:
        """Block for a keypress, redrawing hover tooltips; return its char.

        For pull backends (`Cv2Backend`); push backends use `run` instead.
        """
        while True:
            key = self._backend.poll_key(20)
            if key != -1:
                char = _key_char(key)
                if self._controllable() and self._layers.handle(char):
                    for name, state in self._windows.items():
                        self._rerender(name, state)
                    continue
                return char
            for name, state in self._windows.items():
                if state.dirty:
                    state.dirty = False
                    self._render_hover(name, state)

    def run(self, on_key: Callable[[str], None]) -> None:
        """Enter driven mode: the backend's own event loop delivers events.

        Present frames with `show` as usual; hover tooltips then redraw inline on
        each mouse move (there is no `wait` loop), and every keypress is delivered
        to ``on_key`` as a one-character string. Returns immediately — the
        backend's native loop (Qt, a notebook kernel, a web socket) keeps calling
        back until `close`; the caller keeps that loop alive.

        Requires a push-capable backend; pull backends like `Cv2Backend` raise
        ``NotImplementedError`` (use `wait` instead).

        Args:
            on_key: Called with the pressed key's character on each keypress.

        """
        self._driven = True

        def dispatch(key: int) -> None:
            char = _key_char(key)
            if self._controllable() and self._layers.handle(char):
                for name, state in self._windows.items():
                    self._rerender(name, state)
                return
            on_key(char)

        self._backend.set_key_handler(dispatch)

    def _render_hover(self, name: str, state: _HoverState) -> None:
        """Present ``state``'s base frame, with its current tooltip if any."""
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

    def _handler(self, name: str, state: _HoverState) -> MouseHandler:
        """Build the mouse-move handler that tracks the hovered tooltip.

        Pull mode marks the window dirty for `wait` to repaint; driven mode
        repaints inline (there is no loop to defer to).
        """

        def handler(x: int, y: int) -> None:
            tooltip = state.hitmap.hit(x, y)
            pos = (int(x), int(y))
            changed = tooltip is not state.hover or (
                tooltip is not None and pos != state.mouse
            )
            if not changed:
                return
            state.hover = tooltip
            state.mouse = pos
            if self._driven:
                self._render_hover(name, state)
            else:
                state.dirty = True

        return handler
