"""The `WindowBackend` protocol: the surface a `Viewer` draws through.

A backend owns the platform specifics — creating windows, presenting BGR frames,
reporting the screen size, routing mouse-move events, and polling the keyboard.
The `Viewer` stays backend-agnostic and drives one of these. `Cv2Backend` is the
default (OpenCV highgui); other surfaces (a notebook, an HTML canvas) can conform
to the same protocol without touching the viewer.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np

#: Called with the cursor's ``(x, y)`` in frame pixels on every mouse move.
MouseHandler = Callable[[int, int], None]


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

        A ``timeout_ms`` of ``0`` blocks until a key is pressed.
        """
        ...

    def close(self) -> None:
        """Destroy every window this backend created."""
        ...
