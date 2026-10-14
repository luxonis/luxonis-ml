"""The OpenCV (``cv2``) window backend.

``cv2`` is imported lazily inside each method — never at module import — so that
``import luxonis_ml.vizlab.viewer`` stays free of the heavy OpenCV import until an
actual window is opened (mirroring `luxonis_ml.vizlab.annotations.mask`).
"""

import contextlib
from typing import Protocol, cast

import numpy as np

from .backend import KeyHandler, MouseHandler, WindowBackend

#: How often a wait without a timeout checks that a window is still open.
_CLOSE_CHECK_MS = 100

#: What `Cv2Backend.poll_key` reports once every window is closed: the quit
#: key of the vizlab presentations.
_CLOSED_KEY = ord("q")


class _TkRoot(Protocol):
    """Subset of a Tk root used for screen-size discovery."""

    def withdraw(self) -> None: ...

    def winfo_screenwidth(self) -> int: ...

    def winfo_screenheight(self) -> int: ...

    def destroy(self) -> None: ...


class Cv2Backend(WindowBackend):
    """A `WindowBackend` backed by OpenCV's highgui windows."""

    def __init__(self) -> None:
        """Create a backend with no open windows."""
        self._live: set[str] = set()
        # The windows seen on screen: only these can read as closed, so a build
        # that cannot report visibility never ends a wait.
        self._seen: set[str] = set()
        # cv2 may drop a callback that is not referenced from Python, so keep one.
        self._callbacks: dict[str, object] = {}

    def screen_size(self) -> tuple[int, int] | None:
        """Return the screen resolution, read through Tk.

        Uses Tk (standard library), so a viewer can skip scaling and centering
        when the size is unknown.

        Returns:
            The screen ``(width, height)`` in pixels, or ``None`` on any
            failure (a headless session, or Tk missing).

        """
        root: _TkRoot | None = None
        try:
            import tkinter as tk

            root = cast(_TkRoot, tk.Tk())
            root.withdraw()
            size = (root.winfo_screenwidth(), root.winfo_screenheight())
        except Exception:
            return None
        finally:
            if root is not None:
                with contextlib.suppress(Exception):
                    root.destroy()
        return size

    def create_window(self, name: str) -> None:
        import cv2

        # ``WINDOW_GUI_NORMAL`` opts out of the Qt build's *expanded* chrome —
        # the status bar reporting the pixel under the cursor, and the
        # right-click zoom/pan toolbar. It is the default otherwise, and it
        # doubles the cost of presenting a frame (~9.6ms to ~4.9ms at 1080p),
        # because the status bar repaints on every mouse-move — exactly when a
        # hover tooltip is redrawing. The viewer draws its own controls, and
        # the window stays freely resizable either way.
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        self._live.add(name)

    def destroy_window(self, name: str) -> None:
        import cv2

        cv2.destroyWindow(name)
        self._live.discard(name)
        self._seen.discard(name)
        self._callbacks.pop(name, None)

    def show(self, name: str, frame: np.ndarray) -> None:
        import cv2

        cv2.imshow(name, frame)

    def resize(self, name: str, width: int, height: int) -> None:
        import cv2

        cv2.resizeWindow(name, width, height)

    def center(
        self, name: str, width: int, height: int, screen: tuple[int, int]
    ) -> None:
        import cv2

        cv2.moveWindow(
            name,
            max(0, (screen[0] - width) // 2),
            max(0, (screen[1] - height) // 2),
        )

    def set_mouse_handler(self, name: str, handler: MouseHandler) -> None:
        import cv2

        def callback(
            event: int, x: int, y: int, flags: int, param: object
        ) -> None:
            if event == cv2.EVENT_MOUSEMOVE:
                handler(x, y, False)
            elif event == cv2.EVENT_LBUTTONDOWN:
                handler(x, y, True)

        self._callbacks[name] = callback
        cv2.setMouseCallback(name, callback)

    def poll_key(self, timeout_ms: int) -> int:
        """Wait for a keypress, the way `WindowBackend.poll_key` does.

        The full key code is kept, so an arrow stays apart from the letter
        that shares its low byte. When the user closes every window from its
        title bar, no more keys arrive, so that reads as ``q``.
        """
        import cv2

        while True:
            key = cv2.waitKeyEx(timeout_ms or _CLOSE_CHECK_MS)
            if key != -1:
                return key
            if self._all_closed():
                return _CLOSED_KEY
            if timeout_ms:
                return -1

    def _all_closed(self) -> bool:
        """Tell whether the user closed every window that was on screen."""
        import cv2

        visible = set()
        for name in self._live:
            try:
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
                    visible.add(name)
            except cv2.error:  # some builds raise for a closed window
                pass
        self._seen |= visible
        return bool(self._live) and not visible and self._live <= self._seen

    def set_key_handler(self, handler: KeyHandler) -> None:
        raise NotImplementedError(
            "Cv2Backend is pull-based (keys come from poll_key); use "
            "Viewer.wait(), not Viewer.run()."
        )

    def close(self) -> None:
        import cv2

        for name in list(self._live):
            cv2.destroyWindow(name)
        self._live.clear()
        self._seen.clear()
        self._callbacks.clear()
