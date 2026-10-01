"""The `WindowBackend` protocol: the surface a `Viewer` draws through.

A backend owns the platform specifics — creating windows, presenting BGR frames,
reporting the screen size, routing mouse-move events, and delivering keypresses.
The `Viewer` stays backend-agnostic and drives one of these. `Cv2Backend` is the
default (OpenCV highgui); other surfaces (a notebook, an HTML canvas) can conform
to the same protocol without touching the viewer.

Backends come in two flavors, matching the `Viewer`'s two entry points:

- **pull** (e.g. `Cv2Backend`): the `Viewer` owns the loop and asks for keys via
  `WindowBackend.poll_key`; used with `Viewer.wait`. Such backends need not
  implement `WindowBackend.set_key_handler`.
- **push** (event-driven: Qt, a notebook kernel, a web socket): the backend owns
  the loop and calls a handler registered via
  `WindowBackend.set_key_handler`; used with `Viewer.run`. Such backends'
  `WindowBackend.poll_key` may simply return ``-1``.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np

#: Called with the cursor's ``(x, y)`` in frame pixels and whether this was a
#: click (left-button press) rather than a move, on every mouse move or click.
MouseHandler = Callable[[int, int, bool], None]

#: Called with a backend key code on each keypress (push backends only).
KeyHandler = Callable[[int], None]


class WindowBackend(Protocol):
    """Platform window operations a `Viewer` needs."""

    def screen_size(self) -> tuple[int, int] | None:
        """Return the size of the screen.

        Returns:
            The screen ``(width, height)`` in pixels, or ``None`` when the
            backend cannot tell.

        """
        ...

    def create_window(self, name: str) -> None:
        """Create a window, or show it again if it exists.

        Args:
            name: The name that identifies the window.

        """
        ...

    def destroy_window(self, name: str) -> None:
        """Destroy a window, if it exists.

        Args:
            name: The name of the window.

        """
        ...

    def show(self, name: str, frame: np.ndarray) -> None:
        """Present a frame in a window.

        Args:
            name: The name of the window.
            frame: A BGR ``(H, W, 3)`` uint8 frame.

        """
        ...

    def resize(self, name: str, width: int, height: int) -> None:
        """Resize a window.

        Args:
            name: The name of the window.
            width: The new width in pixels.
            height: The new height in pixels.

        """
        ...

    def center(
        self, name: str, width: int, height: int, screen: tuple[int, int]
    ) -> None:
        """Center a window on the screen.

        Args:
            name: The name of the window.
            width: The width of the window in pixels.
            height: The height of the window in pixels.
            screen: The ``(width, height)`` of the screen in pixels.

        """
        ...

    def set_mouse_handler(self, name: str, handler: MouseHandler) -> None:
        """Route the mouse moves and clicks over a window to a handler.

        Args:
            name: The name of the window.
            handler: Called with the cursor position in frame pixels and
                whether the event is a click.

        """
        ...

    def poll_key(self, timeout_ms: int) -> int:
        """Wait for a keypress.

        Pull backends implement this. Push backends may return ``-1``.

        Args:
            timeout_ms: How long to wait, in milliseconds. ``0`` waits until a
                key is pressed.

        Returns:
            The key code, or ``-1`` when no key was pressed in time.

        """
        ...

    def set_key_handler(self, handler: KeyHandler) -> None:
        """Route keypresses to a handler. Push backends only.

        `Viewer.run` calls this once.

        Args:
            handler: Called with the key code of each keypress.

        Raises:
            NotImplementedError: In a pull backend, which delivers keys through
                `poll_key` instead.

        """
        ...

    def close(self) -> None:
        """Destroy every window this backend created."""
        ...
