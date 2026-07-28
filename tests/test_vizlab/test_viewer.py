"""Tests for the interactive `Viewer`, driven by a fake window backend."""

import numpy as np
import pytest

from luxonis_ml.vizlab import BBox, Image, Tooltip
from luxonis_ml.vizlab.frame import Frame
from luxonis_ml.vizlab.viewer import (
    Cv2Backend,
    LayerState,
    Viewer,
    draw_tooltip,
    render_tooltip_card,
)
from luxonis_ml.vizlab.viewer.backend import KeyHandler, MouseHandler


class FakeBackend:
    """A `WindowBackend` that records calls and replays scripted keys."""

    def __init__(
        self,
        screen: tuple[int, int] | None = None,
        keys: list[int] | None = None,
    ) -> None:
        self._screen = screen
        self._keys = list(keys or [])
        self.shown: list[tuple[str, np.ndarray]] = []
        self.created: list[str] = []
        self.destroyed: list[str] = []
        self.handlers: dict[str, MouseHandler] = {}
        self.key_handler: KeyHandler | None = None

    def screen_size(self) -> tuple[int, int] | None:
        return self._screen

    def create_window(self, name: str) -> None:
        self.created.append(name)

    def destroy_window(self, name: str) -> None:
        self.destroyed.append(name)

    def show(self, name: str, frame: np.ndarray) -> None:
        self.shown.append((name, frame.copy()))

    def resize(self, name: str, width: int, height: int) -> None:
        pass

    def center(
        self, name: str, width: int, height: int, screen: tuple[int, int]
    ) -> None:
        pass

    def set_mouse_handler(self, name: str, handler: MouseHandler) -> None:
        self.handlers[name] = handler

    def poll_key(self, timeout_ms: int) -> int:
        return self._keys.pop(0) if self._keys else -1

    def set_key_handler(self, handler: KeyHandler) -> None:
        self.key_handler = handler

    def close(self) -> None:
        pass


def _tooltip_image(h: int = 120, w: int = 200) -> tuple[Image, Tooltip]:
    tip = Tooltip(title="car", rows=(("id", "7"),))
    image = Image(np.zeros((h, w, 3), np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.8, h=0.8, tooltip=tip)
    )
    return image, tip


def test_show_arms_hover_and_wait_draws_tooltip_then_quits() -> None:
    backend = FakeBackend(keys=[-1, ord("q")])
    viewer = Viewer(backend)
    image, _ = _tooltip_image()
    viewer.show("w", image.frame())
    assert backend.created == ["w"]
    base = backend.shown[-1][1]

    backend.handlers["w"](100, 60)  # over the box center
    key = viewer.wait()  # -1 -> redraw dirty, then "q"
    assert key == "q"
    assert not np.array_equal(backend.shown[-1][1], base)  # tooltip drawn


def test_hover_outside_boxes_keeps_base_frame() -> None:
    backend = FakeBackend(keys=[-1, ord("q")])
    viewer = Viewer(backend)
    image, _ = _tooltip_image()
    viewer.show("w", image.frame())
    base = backend.shown[-1][1]

    backend.handlers["w"](3, 3)  # corner, outside the box -> no tooltip
    viewer.wait()
    assert np.array_equal(backend.shown[-1][1], base)


def test_show_fits_frame_to_screen() -> None:
    backend = FakeBackend(screen=(100, 100))
    viewer = Viewer(backend)
    image = Image(np.zeros((400, 600, 3), np.uint8))
    viewer.show("w", image.frame())
    _, frame = backend.shown[-1]
    height, width = frame.shape[:2]
    assert width <= 90  # within 0.9 * screen
    assert height <= 90


def test_show_blocking_returns_key() -> None:
    backend = FakeBackend(keys=[ord("n")])
    viewer = Viewer(backend)
    image = Image(np.zeros((40, 60, 3), np.uint8))
    assert viewer.show_blocking("w", image) == "n"


def test_destroy_stale_closes_absent_windows() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    image = Image(np.zeros((40, 60, 3), np.uint8))
    frame = image.frame()
    viewer.show("a", frame)
    viewer.show("b", frame)
    viewer.destroy_stale({"a"})
    assert backend.destroyed == ["b"]


def test_draw_tooltip_modifies_frame_and_clamps() -> None:
    frame = np.full((200, 300, 3), 40, np.uint8)
    before = frame.copy()
    tip = Tooltip(title="car #3", rows=(("track_id", "42"),))
    draw_tooltip(frame, tip, (295, 195))  # bottom-right corner -> clamps
    assert not np.array_equal(frame, before)


def test_draw_tooltip_empty_is_noop() -> None:
    frame = np.full((100, 100, 3), 40, np.uint8)
    before = frame.copy()
    draw_tooltip(frame, Tooltip(), (10, 10))
    assert np.array_equal(frame, before)


def test_render_tooltip_card_is_rgba() -> None:
    card = render_tooltip_card(Tooltip(title="car", rows=(("id", "7"),)), 14)
    assert card.ndim == 3
    assert card.shape[2] == 4


def test_tint_swatch_shows_class_color_without_a_title() -> None:
    from luxonis_ml.vizlab import Color

    # A rows-only tooltip has no title to tint; the swatch surfaces the color.
    plain = render_tooltip_card(Tooltip(rows=(("id", "7"),)), 14)
    tinted = render_tooltip_card(
        Tooltip(rows=(("id", "7"),), tint=Color(255, 0, 0)), 14
    )
    # The swatch adds a header band, so the tinted card is taller.
    assert tinted.shape[0] > plain.shape[0]
    # And the swatch actually paints red pixels the plain card lacks.
    red = (
        (tinted[..., 0] > 180) & (tinted[..., 1] < 80) & (tinted[..., 2] < 80)
    )
    assert red.any()


def test_run_delivers_keys_and_hovers_inline() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    image, _ = _tooltip_image()
    viewer.show("w", image.frame())
    got: list[str] = []
    viewer.run(got.append)

    # A key event from the backend's own loop reaches on_key as a character.
    assert backend.key_handler is not None
    backend.key_handler(ord("n"))
    assert got == ["n"]

    # Hover now repaints inline (no wait loop): the shown frame changes.
    base = backend.shown[-1][1]
    backend.handlers["w"](100, 60)  # over the box center
    assert not np.array_equal(backend.shown[-1][1], base)


def test_cv2_backend_is_pull_only() -> None:
    # Constructing it touches neither cv2 nor Tk; only the raise is exercised.
    with pytest.raises(NotImplementedError):
        Cv2Backend().set_key_handler(lambda _key: None)


def _layer_image() -> Image:
    return Image(np.zeros((300, 400, 3), np.uint8)).add(
        BBox(
            x=0.1, y=0.1, w=0.6, h=0.6, label="car", tooltip=Tooltip(title="c")
        )
    )


def test_control_key_rerenders_and_is_not_returned_to_caller() -> None:
    backend = FakeBackend(keys=[ord("m"), ord("q")])
    viewer = Viewer(backend)
    seen: list[bool] = []

    def render(state: LayerState) -> Frame:
        seen.append(state.masks)
        return _layer_image().frame()

    viewer.show("w", _layer_image().frame(), render=render)
    shown_before = len(backend.shown)
    key = viewer.wait()  # 'm' toggles + re-renders, then 'q' returns
    assert key == "q"
    assert seen == [False]  # render ran once, with masks toggled off
    assert len(backend.shown) > shown_before  # the window was repainted


def test_control_keys_pass_through_without_a_render_callback() -> None:
    backend = FakeBackend(keys=[ord("m")])
    viewer = Viewer(backend)
    viewer.show("w", _layer_image().frame())  # not interactive
    assert viewer.wait() == "m"  # forwarded to the caller unchanged


def test_hud_is_drawn_only_on_interactive_windows() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    frame = _layer_image().frame()
    viewer.show("plain", frame)
    plain = backend.shown[-1][1]
    viewer.show("live", frame, render=lambda _state: frame)
    live = backend.shown[-1][1]
    assert not np.array_equal(plain, live)  # the HUD card was composited on


def test_run_swallows_control_keys_and_forwards_the_rest() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    viewer.show(
        "w",
        _layer_image().frame(),
        render=lambda _state: _layer_image().frame(),
    )
    got: list[str] = []
    viewer.run(got.append)
    assert backend.key_handler is not None

    backend.key_handler(ord("k"))  # control key: consumed, re-renders
    backend.key_handler(ord("n"))  # non-control: forwarded
    assert got == ["n"]
    assert viewer.layers.keypoints is False
