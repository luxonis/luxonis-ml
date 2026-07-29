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
        self.closed = False

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
        self.closed = True


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

    backend.handlers["w"](100, 60, False)  # over the box center
    key = viewer.wait()  # -1 -> redraw dirty, then "q"
    assert key == "q"
    assert not np.array_equal(backend.shown[-1][1], base)  # tooltip drawn


def test_hover_outside_boxes_keeps_base_frame() -> None:
    backend = FakeBackend(keys=[-1, ord("q")])
    viewer = Viewer(backend)
    image, _ = _tooltip_image()
    viewer.show("w", image.frame())
    base = backend.shown[-1][1]

    backend.handlers["w"](3, 3, False)  # corner, outside the box -> no tooltip
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


def test_screen_property_exposes_backend_size() -> None:
    assert Viewer(FakeBackend(screen=(640, 480))).screen == (640, 480)


def test_show_prepared_does_not_render_the_frame_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    frame = Image(np.zeros((40, 60, 3), np.uint8)).frame()
    original_render = Frame.render
    render_sizes: list[tuple[int, int] | None] = []

    def tracked_render(
        self: Frame,
        size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        render_sizes.append(size)
        return original_render(self, size)

    monkeypatch.setattr(Frame, "render", tracked_render)
    prepared = viewer.prepare(frame)
    assert render_sizes == [None]

    viewer.show_prepared("w", prepared)

    assert render_sizes == [None]
    assert backend.shown[-1][0] == "w"


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


def test_close_resets_windows_and_closes_backend() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    viewer.show("w", _tooltip_image()[0].frame())

    viewer.close()

    assert backend.closed
    assert viewer._live == set()
    assert viewer._windows == {}


def test_draw_tooltip_modifies_frame_and_clamps() -> None:
    frame = np.full((200, 300, 3), 40, np.uint8)
    before = frame.copy()
    tip = Tooltip(title="car #3", rows=(("track_id", "42"),))
    draw_tooltip(frame, tip, (295, 195))  # bottom-right corner -> clamps
    assert not np.array_equal(frame, before)


def test_blit_frost_blurs_the_backdrop_only_under_the_card() -> None:
    from luxonis_ml.vizlab.viewer.tooltip_render import blit_rgba_on_bgr

    # A high-frequency background so a blur is measurable.
    rng = np.random.default_rng(0)
    frame = (rng.integers(0, 2, (200, 300, 1)) * 255).astype(np.uint8)
    frame = np.repeat(frame, 3, axis=2)
    card = render_tooltip_card(
        Tooltip(title="car #1", rows=(("speed", "48"),)), 16
    )
    ch, cw = card.shape[:2]
    x, y = 40, 40

    sharp = frame.copy()
    blit_rgba_on_bgr(sharp, card, x, y, blur=0.0)
    frost = frame.copy()
    blit_rgba_on_bgr(frost, card, x, y, blur=11.0)

    # The frost softens the backdrop under the card body...
    def local_std(img: np.ndarray) -> float:
        cy, cx = y + ch // 2, x + cw // 2
        return float(img[cy - 8 : cy + 8, cx - 8 : cx + 8].std())

    assert local_std(frost) < local_std(sharp)
    # ...and touches nothing outside the card (a far region is untouched).
    far = (slice(150, 190), slice(250, 290))
    assert np.array_equal(frost[far], frame[far])
    # blur=0 must match the original alpha composite exactly (back-compat).
    plain = frame.copy()
    blit_rgba_on_bgr(plain, card, x, y)
    assert np.array_equal(plain, sharp)


def test_frost_backdrop_fills_the_body_but_not_the_shadow_or_margin() -> None:
    # The frost must fill the colored body at *any* panel opacity, yet leave the
    # drop shadow sharp (the shadow is pure black, so keying on color not alpha
    # avoids a blurred halo around the panel) and the transparent margin sharp.
    from luxonis_ml.vizlab.viewer.tooltip_render import _frost_backdrop

    rng = np.random.default_rng(0)
    roi = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    roi_f = roi.astype(np.float32)
    roi_std = roi_f.std()

    for body_alpha in (230, 150, 90):  # opaque -> quite translucent panel
        card = np.zeros((80, 120, 4), np.uint8)
        card[5:75, 5:115, 3] = 90  # soft black drop shadow (rgb stays 0)
        card[15:65, 20:100, :3] = (29, 41, 57)  # the colored body fill...
        card[15:65, 20:100, 3] = body_alpha  # ...at the panel's opacity
        card[30:34, 30:80, :3] = (200, 210, 250)  # opaque "text" on the body
        card[30:34, 30:80, 3] = 255
        out = _frost_backdrop(roi, card, 12.0)

        assert out[45:63, 25:95].std() < roi_std * 0.5  # body clearly blurred
        # The shadow ring (black, translucent) is left sharp -> no blurred halo.
        assert np.allclose(out[8:12, 40:80], roi_f[8:12, 40:80])
        # The fully-transparent margin is left sharp too.
        assert np.allclose(out[:4, :4], roi_f[:4, :4])


def test_draw_tooltip_empty_is_noop() -> None:
    frame = np.full((100, 100, 3), 40, np.uint8)
    before = frame.copy()
    draw_tooltip(frame, Tooltip(), (10, 10))
    assert np.array_equal(frame, before)


def test_draw_tooltip_too_large_for_frame_is_noop() -> None:
    frame = np.full((8, 8, 3), 40, np.uint8)
    before = frame.copy()
    draw_tooltip(
        frame,
        Tooltip(title="large", rows=(("description", "too large"),)),
        (1, 1),
    )
    assert np.array_equal(frame, before)


def test_blit_outside_frame_is_noop() -> None:
    from luxonis_ml.vizlab.viewer.tooltip_render import blit_rgba_on_bgr

    frame = np.full((20, 20, 3), 40, np.uint8)
    before = frame.copy()
    card = np.full((5, 5, 4), 255, np.uint8)
    blit_rgba_on_bgr(frame, card, 30, 30)
    assert np.array_equal(frame, before)


def test_small_frost_blur_uses_full_resolution_path() -> None:
    from luxonis_ml.vizlab.viewer.tooltip_render import _blur_bgr

    roi = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    blurred = _blur_bgr(roi, 1.0)
    assert blurred.shape == roi.shape
    assert blurred.dtype == np.float32


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
    backend.handlers["w"](100, 60, False)  # over the box center
    assert not np.array_equal(backend.shown[-1][1], base)


def test_hud_flag_gates_the_floating_controls() -> None:
    # With hud=False (controls shown in a panel instead) the floating HUD is not
    # drawn; with hud=True it composites the controls card onto the frame.
    frame = np.zeros((200, 300, 3), np.uint8)
    off = frame.copy()
    Viewer(FakeBackend(), hud=False)._draw_hud(off)
    assert np.array_equal(off, frame)  # untouched

    on = frame.copy()
    Viewer(FakeBackend(), hud=True)._draw_hud(on)
    assert not np.array_equal(on, frame)  # controls drawn


def test_apply_action_dispatches_control_and_class_clicks() -> None:
    viewer = Viewer(FakeBackend(), hud=False)
    viewer.layers.classes = ("car", "person")
    viewer._apply_action("class:car")
    assert viewer.layers.hidden == {"car"}  # a legend click hides a class
    viewer._apply_action("key:m")
    assert viewer.layers.masks is False  # a control click triggers its key
    # The master switch: with "car" already hidden, one click shows all...
    viewer._apply_action("classes:toggle")
    assert viewer.layers.hidden == set()
    viewer._apply_action("classes:toggle")  # ...and the next hides every class
    assert viewer.layers.hidden == {"car", "person"}


def test_layer_state_copy_is_independent() -> None:
    state = LayerState(
        masks=False,
        hidden={"car"},
        classes=("car", "person"),
        _focus=1,
    )

    copied = state.copy()
    assert copied == state

    copied.hidden.add("person")
    copied.labels = False

    assert state.hidden == {"car"}
    assert state.labels is True


def test_panel_click_toggles_a_class_through_wait() -> None:
    from luxonis_ml.vizlab.geometry import Rect
    from luxonis_ml.vizlab.hitmap import ClickMap

    backend = FakeBackend(keys=[-1, ord("q")])
    viewer = Viewer(backend, hud=False)
    viewer.layers.classes = ("car",)
    image, _ = _tooltip_image()
    frame = Frame(
        image, clickmap=ClickMap([(Rect(0, 0, 50, 50), "class:car")])
    )
    viewer.show("w", frame, render=lambda _state: frame)

    backend.handlers["w"](10, 10, True)  # click inside the swatch region
    viewer.wait()  # -1 -> apply the pending click action, then "q"
    assert viewer.layers.hidden == {"car"}


def test_click_outside_clickmap_is_ignored() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    image, _ = _tooltip_image()
    viewer.show("w", image.frame())
    backend.handlers["w"](1, 1, True)
    assert viewer.layers.hidden == set()


def test_driven_click_applies_action_immediately() -> None:
    from luxonis_ml.vizlab.geometry import Rect
    from luxonis_ml.vizlab.hitmap import ClickMap

    backend = FakeBackend()
    viewer = Viewer(backend, hud=False)
    viewer.layers.classes = ("car",)
    image, _ = _tooltip_image()
    frame = Frame(
        image, clickmap=ClickMap([(Rect(0, 0, 50, 50), "class:car")])
    )
    viewer.show("w", frame, render=lambda _state: frame)
    viewer.run(lambda _key: None)

    backend.handlers["w"](10, 10, True)

    assert viewer.layers.hidden == {"car"}


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


def test_rerender_is_noop_without_a_render_callback() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    viewer.show("w", _layer_image().frame())
    shown = len(backend.shown)
    viewer._rerender("w", viewer._windows["w"])
    assert len(backend.shown) == shown
    viewer._render_hover("w", viewer._windows["w"])
    assert len(backend.shown) == shown + 1


def test_hud_is_drawn_only_on_interactive_windows() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    frame = _layer_image().frame()
    viewer.show("plain", frame)
    plain = backend.shown[-1][1]
    viewer.show("live", frame, render=lambda _state: frame)
    live = backend.shown[-1][1]
    assert not np.array_equal(plain, live)  # the HUD card was composited on


def test_controls_card_is_rgba_and_scales_with_size() -> None:
    from luxonis_ml.vizlab.viewer.hud import render_controls_card

    controls = LayerState(
        masks=False, hidden={"person"}, classes=("car", "person")
    ).controls()
    small = render_controls_card(controls, 12)
    large = render_controls_card(controls, 22)
    assert small.ndim == 3
    assert small.shape[2] == 4  # RGBA
    assert large.shape[0] > small.shape[0]  # bigger type -> bigger card
    assert large.shape[1] > small.shape[1]


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


def test_unknown_panel_action_is_ignored() -> None:
    viewer = Viewer(FakeBackend())
    viewer._apply_action("unknown:value")
    assert viewer.layers.is_default()
