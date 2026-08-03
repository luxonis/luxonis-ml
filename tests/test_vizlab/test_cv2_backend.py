"""Headless behavioral coverage for the lazy OpenCV window backend."""

import sys
from collections.abc import Callable
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from luxonis_ml.vizlab.viewer import Cv2Backend

_CvCallback = Callable[[int, int, int, int, None], None]


class _FakeCv2(ModuleType):
    WINDOW_NORMAL = 17
    WINDOW_GUI_NORMAL = 32
    EVENT_MOUSEMOVE = 1
    EVENT_LBUTTONDOWN = 2

    def __init__(self) -> None:
        super().__init__("cv2")
        self.calls: list[tuple[str, tuple[int | str, ...]]] = []
        self.callbacks: dict[str, _CvCallback] = {}

    def namedWindow(self, name: str, mode: int) -> None:
        self.calls.append(("named", (name, mode)))

    def destroyWindow(self, name: str) -> None:
        self.calls.append(("destroy", (name,)))

    def imshow(self, name: str, frame: np.ndarray) -> None:
        self.calls.append(("show", (name, frame.shape[1], frame.shape[0])))

    def resizeWindow(self, name: str, width: int, height: int) -> None:
        self.calls.append(("resize", (name, width, height)))

    def moveWindow(self, name: str, x: int, y: int) -> None:
        self.calls.append(("move", (name, x, y)))

    def setMouseCallback(self, name: str, callback: _CvCallback) -> None:
        self.callbacks[name] = callback

    def waitKey(self, timeout_ms: int) -> int:
        self.calls.append(("wait", (timeout_ms,)))
        return 113


class _TkRoot:
    def __init__(
        self,
        *,
        fail_height: bool = False,
        fail_destroy: bool = False,
    ) -> None:
        self.fail_height = fail_height
        self.fail_destroy = fail_destroy
        self.withdrawn = False
        self.destroyed = False

    def withdraw(self) -> None:
        self.withdrawn = True

    def winfo_screenwidth(self) -> int:
        return 1920

    def winfo_screenheight(self) -> int:
        if self.fail_height:
            raise RuntimeError("headless")
        return 1080

    def destroy(self) -> None:
        self.destroyed = True
        if self.fail_destroy:
            raise RuntimeError("cleanup failed")


class _FakeTk(ModuleType):
    def __init__(self, root: _TkRoot) -> None:
        super().__init__("tkinter")
        self.Tk: Callable[[], _TkRoot] = lambda: root


def _install_cv2(monkeypatch: pytest.MonkeyPatch) -> _FakeCv2:
    cv2 = _FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


def _install_tk(monkeypatch: pytest.MonkeyPatch, root: _TkRoot) -> ModuleType:
    tkinter = _FakeTk(root)
    monkeypatch.setitem(sys.modules, "tkinter", tkinter)
    return tkinter


def test_screen_size_cleans_up_tk_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _TkRoot()
    _install_tk(monkeypatch, root)

    assert Cv2Backend().screen_size() == (1920, 1080)
    assert root.withdrawn
    assert root.destroyed


def test_screen_size_cleans_up_after_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _TkRoot(fail_height=True)
    _install_tk(monkeypatch, root)

    assert Cv2Backend().screen_size() is None
    assert root.destroyed


def test_screen_size_keeps_valid_result_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _TkRoot(fail_destroy=True)
    _install_tk(monkeypatch, root)

    assert Cv2Backend().screen_size() == (1920, 1080)


def test_window_lifecycle_and_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = _install_cv2(monkeypatch)
    backend = Cv2Backend()
    frame = np.zeros((20, 30, 3), np.uint8)

    backend.create_window("main")
    backend.show("main", frame)
    backend.resize("main", 300, 200)
    backend.center("main", 300, 200, (200, 100))
    backend.destroy_window("main")

    assert cv2.calls == [
        # GUI_NORMAL opts out of the Qt status bar, which repaints on every
        # mouse-move and doubled the cost of presenting a frame.
        ("named", ("main", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)),
        ("show", ("main", 30, 20)),
        ("resize", ("main", 300, 200)),
        ("move", ("main", 0, 0)),
        ("destroy", ("main",)),
    ]
    assert backend._live == set()


def test_mouse_handler_routes_move_and_left_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = _install_cv2(monkeypatch)
    backend = Cv2Backend()
    calls: list[tuple[int, int, bool]] = []
    backend.set_mouse_handler(
        "main", lambda x, y, clicked: calls.append((x, y, clicked))
    )
    callback = cast(_CvCallback, backend._callbacks["main"])

    callback(cv2.EVENT_MOUSEMOVE, 7, 8, 0, None)
    callback(cv2.EVENT_LBUTTONDOWN, 9, 10, 0, None)
    callback(999, 11, 12, 0, None)

    assert calls == [(7, 8, False), (9, 10, True)]
    assert cv2.callbacks["main"] is callback


def test_poll_key_and_close_all_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = _install_cv2(monkeypatch)
    backend = Cv2Backend()
    backend.create_window("a")
    backend.create_window("b")
    backend.set_mouse_handler("a", lambda _x, _y, _clicked: None)

    assert backend.poll_key(20) == 113
    backend.close()

    destroyed = {args[0] for name, args in cv2.calls if name == "destroy"}
    assert destroyed == {"a", "b"}
    assert backend._live == set()
    assert backend._callbacks == {}
