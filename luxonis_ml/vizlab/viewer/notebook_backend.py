"""A push (event-driven) `WindowBackend` for Jupyter / IPython notebooks.

Each window is an ``ipywidgets.Image``; navigation is a row of control buttons
that synthesize key events, so it pairs with `Viewer.run`. Hover works when
``ipyevents`` is installed (mouse-move over the image is mapped to frame pixels);
without it the buttons still drive navigation. Every notebook dependency
(``ipywidgets``, ``ipyevents``, Pillow, IPython) is imported lazily, so importing
this module — and ``luxonis_ml.vizlab.viewer`` — never requires them.
"""

import importlib
import io as _io
from typing import Any

import numpy as np

from .backend import KeyHandler, MouseHandler

#: Where the notebook viewer's optional dependencies live.
_EXTRA = "the 'notebook' extra (pip install 'luxonis-ml[notebook]')"


def _require(module: str) -> Any:
    """Import ``module`` lazily, or raise a friendly install hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module!r} is required for the notebook viewer; install {_EXTRA}."
        ) from exc


def _encode_png(frame_bgr: np.ndarray) -> bytes:
    """Encode a BGR frame as PNG bytes (for an ``ipywidgets.Image``)."""
    pil = _require("PIL.Image")
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    buf = _io.BytesIO()
    pil.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _relative_to_frame(
    rel_x: float,
    rel_y: float,
    rect_w: float,
    rect_h: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int]:
    """Map an element-relative cursor position to frame pixels.

    ``ipyevents`` reports the cursor relative to the image element and the
    element's displayed size, which may differ from the frame's pixel size.
    """
    if not rect_w or not rect_h:
        return int(rel_x), int(rel_y)
    return int(rel_x * frame_w / rect_w), int(rel_y * frame_h / rect_h)


class NotebookBackend:
    """Show vizlab frames in a notebook via ipywidgets — a push backend.

    Use it with `Viewer.run`: present frames with `Viewer.show`, then hand off to
    the notebook's own event loop. The control buttons deliver keys to the
    viewer's ``on_key`` (their labels map to the characters given in
    ``controls``), and mouse-move over a frame drives hover tooltips.

    Args:
        screen: Optional ``(width, height)`` budget so the viewer shrinks large
            frames to fit; ``None`` renders at native size.
        controls: ``{label: key_char}`` for the navigation buttons; each click
            delivers ``key_char`` as if it were pressed.

    """

    def __init__(
        self,
        *,
        screen: tuple[int, int] | None = None,
        controls: dict[str, str] | None = None,
    ) -> None:
        self._screen = screen
        self._controls = controls or {
            "◀ Prev": "p",
            "Next ▶": "n",
            "Quit": "q",
        }
        self._images: dict[str, Any] = {}
        self._frame_size: dict[str, tuple[int, int]] = {}
        self._mouse: dict[str, MouseHandler] = {}
        self._key_handler: KeyHandler | None = None
        self._root: Any = None
        self._stack: Any = None

    def screen_size(self) -> tuple[int, int] | None:
        return self._screen

    def create_window(self, name: str) -> None:
        self._ensure_root()
        if name in self._images:
            return
        widgets = _require("ipywidgets")
        image = widgets.Image(format="png")
        self._images[name] = image
        self._stack.children = (*self._stack.children, image)
        self._wire_mouse(name, image)

    def destroy_window(self, name: str) -> None:
        image = self._images.pop(name, None)
        self._mouse.pop(name, None)
        self._frame_size.pop(name, None)
        if image is not None and self._stack is not None:
            self._stack.children = tuple(
                child for child in self._stack.children if child is not image
            )

    def show(self, name: str, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        image = self._images[name]
        image.value = _encode_png(frame)
        image.layout.width = f"{width}px"
        image.layout.height = f"{height}px"
        self._frame_size[name] = (width, height)

    def resize(self, name: str, width: int, height: int) -> None:
        image = self._images.get(name)
        if image is not None:
            image.layout.width = f"{width}px"
            image.layout.height = f"{height}px"

    def center(
        self, name: str, width: int, height: int, screen: tuple[int, int]
    ) -> None:
        pass  # a notebook has no free-floating window to place

    def set_mouse_handler(self, name: str, handler: MouseHandler) -> None:
        self._mouse[name] = handler

    def poll_key(self, timeout_ms: int) -> int:
        return -1  # push backend: keys arrive via buttons / set_key_handler

    def set_key_handler(self, handler: KeyHandler) -> None:
        self._key_handler = handler

    def close(self) -> None:
        for name in list(self._images):
            self.destroy_window(name)
        self._key_handler = None

    # -- internals ----------------------------------------------------------

    def _ensure_root(self) -> None:
        """Build and display the buttons + image stack once."""
        if self._root is not None:
            return
        widgets = _require("ipywidgets")
        display = _require("IPython.display").display
        buttons = []
        for label, char in self._controls.items():
            button = widgets.Button(description=label)
            button.on_click(lambda _b, char=char: self._emit_key(char))
            buttons.append(button)
        self._stack = widgets.VBox([])
        self._root = widgets.VBox([widgets.HBox(buttons), self._stack])
        display(self._root)

    def _emit_key(self, char: str) -> None:
        """Deliver ``char`` to the registered key handler as a key code."""
        if self._key_handler is not None and char:
            self._key_handler(ord(char[0]))

    def _wire_mouse(self, name: str, image: Any) -> None:
        """Route mouse move/click over ``image`` to the window's handler.

        Moves drive hover tooltips; clicks (left button) drive the panel's
        interactive controls and class legend. Both are optional — without
        ``ipyevents`` the navigation buttons still work.
        """
        try:
            ipyevents = importlib.import_module("ipyevents")
        except ImportError:
            return  # hover/clicks are optional; buttons still drive navigation
        event = ipyevents.Event(
            source=image, watched_events=["mousemove", "click"]
        )

        def on_event(payload: dict) -> None:
            handler = self._mouse.get(name)
            if handler is None:
                return
            frame_w, frame_h = self._frame_size.get(name, (0, 0))
            x, y = _relative_to_frame(
                payload.get("relativeX", 0),
                payload.get("relativeY", 0),
                payload.get("boundingRectWidth", frame_w),
                payload.get("boundingRectHeight", frame_h),
                frame_w,
                frame_h,
            )
            handler(x, y, payload.get("type") == "click")

        event.on_dom_event(on_event)
