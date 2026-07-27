"""The `WindowBackend` protocol: the surface a `Viewer` draws through.

A backend owns the platform specifics — creating windows, presenting BGR frames,
reporting the screen size, routing mouse-move events, and delivering keypresses.
The `Viewer` stays backend-agnostic and drives one of these. `Cv2Backend` is the
default (OpenCV highgui); other surfaces (a notebook, an HTML canvas) can conform
to the same protocol without touching the viewer.

Backends come in two flavors, matching the `Viewer`'s two entry points:

- **pull** (e.g. `Cv2Backend`): the `Viewer` owns the loop and asks for keys via
  `poll_key`; used with `Viewer.wait`. Such backends need not implement
  `set_key_handler`.
- **push** (event-driven: Qt, a notebook kernel, a web socket): the backend owns
  the loop and calls a handler registered via `set_key_handler`; used with
  `Viewer.run`. Such backends' `poll_key` may simply return ``-1``.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np

#: Called with the cursor's ``(x, y)`` in frame pixels on every mouse move.
MouseHandler = Callable[[int, int], None]

#: Called with a backend key code on each keypress (push backends only).
KeyHandler = Callable[[int], None]


class WindowBackend(Protocol):
    """Platform window operations a `Viewer` needs."""

    def screen_size(self) -> tuple[int, int] | None:
        """Return the screen ``(width, height)`` in pixels, or ``None``."""
        ...

    def create_window(self, name: str) -> None:
        """Create (or re-show) a window identified by ``name``."""
        ...

    def destroy_window(self, name: str) -> None:
        """Destroy the window ``name`` if it exists."""
        ...

    def show(self, name: str, frame: np.ndarray) -> None:
        """Present a BGR ``(H, W, 3)`` uint8 frame in window ``name``."""
        ...

    def resize(self, name: str, width: int, height: int) -> None:
        """Resize window ``name`` to ``width`` x ``height`` pixels."""
        ...

    def center(
        self, name: str, width: int, height: int, screen: tuple[int, int]
    ) -> None:
        """Center a ``width`` x ``height`` window on a ``screen``-sized display."""
        ...

    def set_mouse_handler(self, name: str, handler: MouseHandler) -> None:
        """Route ``name``'s mouse-move events to ``handler`` (frame pixels)."""
        ...

    def poll_key(self, timeout_ms: int) -> int:
        """Wait up to ``timeout_ms`` for a key; return its code, or ``-1``.

        A ``timeout_ms`` of ``0`` blocks until a key is pressed. Pull backends
        implement this; push backends may return ``-1``.
        """
        ...

    def set_key_handler(self, handler: KeyHandler) -> None:
        """Route keypresses to ``handler`` — push backends only.

        Called once by `Viewer.run`. Pull backends that deliver keys through
        `poll_key` may leave this unimplemented (raise ``NotImplementedError``).
        """
        ...

    def close(self) -> None:
        """Destroy every window this backend created."""
        ...
