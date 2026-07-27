"""The OpenCV (`cv2`) window backend.

``cv2`` is imported lazily inside each method — never at module import — so that
``import luxonis_ml.vizlab.viewer`` stays free of the heavy OpenCV import until an
actual window is opened (mirroring `luxonis_ml.vizlab.annotations.mask`).
"""

import numpy as np

from .backend import KeyHandler, MouseHandler


class Cv2Backend:
    """A `WindowBackend` backed by OpenCV's highgui windows."""

    def __init__(self) -> None:
        self._live: set[str] = set()
        # cv2 may drop a callback that is not referenced from Python, so keep one.
        self._callbacks: dict[str, object] = {}

    def screen_size(self) -> tuple[int, int] | None:
        """Best-effort screen resolution via Tk, or ``None`` if unavailable.

        Uses Tk (standard library); returns ``None`` on any failure (headless
        session, Tk missing), letting the viewer skip scaling and centering.
        """
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            size = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()
        except Exception:
            return None
        else:
            return size

    def create_window(self, name: str) -> None:
        import cv2

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        self._live.add(name)

    def destroy_window(self, name: str) -> None:
        import cv2

        cv2.destroyWindow(name)
        self._live.discard(name)
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
                handler(x, y)

        self._callbacks[name] = callback
        cv2.setMouseCallback(name, callback)

    def poll_key(self, timeout_ms: int) -> int:
        import cv2

        return cv2.waitKey(timeout_ms)

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
        self._callbacks.clear()
