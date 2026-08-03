"""Tests for the interactive `Viewer`, driven by a fake window backend."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.vizlab import BBox, Image, Tooltip, io
from luxonis_ml.vizlab.interaction.frame import Frame
from luxonis_ml.vizlab.viewer import (
    Cv2Backend,
    LayerState,
    Viewer,
    draw_tooltip,
    render_tooltip_card,
)
from luxonis_ml.vizlab.viewer.backend import KeyHandler, MouseHandler
from luxonis_ml.vizlab.viewer.viewer import _key_char


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


def test_moving_hover_repaints_only_the_last_card_and_matches_a_full_redraw() -> (
    None
):
    # The tooltip is composited into a scratch buffer and only the region the
    # previous card covered is restored, rather than re-copying the whole frame.
    # Restore too little and the old card smears across the window, so every
    # frame is checked against the same thing drawn from scratch.
    backend = FakeBackend(keys=[-1] * 6 + [ord("q")])
    viewer = Viewer(backend, hud=False)
    image, tip = _tooltip_image(h=300, w=400)
    viewer.show("w", image.frame())
    base = backend.shown[-1][1]

    for x, y in ((90, 70), (120, 90), (100, 75), (200, 150), (60, 60)):
        backend.handlers["w"](x, y, False)
        viewer._flush_hover()
        expected = base.copy()
        draw_tooltip(expected, tip, (x, y))
        assert np.array_equal(backend.shown[-1][1], expected), (x, y)


def test_hover_card_is_rendered_once_per_region_not_once_per_move() -> None:
    # Re-rendering the card on every mouse-move dominated the redraw; only its
    # position changes while the pointer stays inside one hover region.
    backend = FakeBackend()
    viewer = Viewer(backend, hud=False)
    image, tip = _tooltip_image(h=300, w=400)
    viewer.show("w", image.frame())
    state = viewer._windows["w"]

    backend.handlers["w"](90, 70, False)
    viewer._flush_hover()
    card = state.card
    assert card is not None
    assert card.tooltip is tip

    backend.handlers["w"](120, 95, False)
    viewer._flush_hover()
    assert state.card is card  # same region -> the very same card object

    other = Tooltip(title="bike", rows=(("id", "9"),))
    state.hover = other
    viewer._render_hover("w", state)
    assert state.card is not None
    assert state.card is not card
    assert state.card.tooltip is other


def test_rerender_drops_the_hover_scratch_so_no_tooltip_survives_it() -> None:
    # The scratch buffer is a copy of the *old* frame; keeping it across a
    # re-render would show a stale tooltip over stale pixels.
    backend = FakeBackend()
    viewer = Viewer(backend, hud=False)
    image, _ = _tooltip_image(h=300, w=400)
    frame = image.frame()
    viewer.show("w", frame, render=lambda _layers: frame)
    state = viewer._windows["w"]

    backend.handlers["w"](90, 70, False)
    viewer._flush_hover()
    assert state.scratch is not None
    assert state.painted is not None

    viewer._rerender("w", state)
    assert state.scratch is None
    assert state.card is None
    assert state.painted is None
    assert np.array_equal(backend.shown[-1][1], state.base)


def test_wait_polls_tightly_while_a_tooltip_is_moving() -> None:
    # A hover redraw costs far less than the idle poll it used to wait out, so
    # the loop polls tightly right after one and relaxes once nothing moves.
    timeouts: list[int] = []

    class PollRecordingBackend(FakeBackend):
        def poll_key(self, timeout_ms: int) -> int:
            timeouts.append(timeout_ms)
            if len(timeouts) == 1:  # first poll: deliver a move to hover over
                self.handlers["w"](100, 60, False)
            return -1 if len(timeouts) < 4 else ord("q")

    backend = PollRecordingBackend()
    viewer = Viewer(backend, hud=False)
    image, _ = _tooltip_image()
    viewer.show("w", image.frame())

    assert viewer.wait() == "q"
    # idle -> the move is flushed -> tight -> nothing left to draw -> idle again
    assert timeouts == [20, 1, 20, 20]


def _saved_bgr(path: Path) -> np.ndarray:
    """Read a saved PNG back as the BGR frame it was written from."""
    return io.export(io.load_rgba(path), "bgr")


def test_save_key_writes_the_shown_frame_and_is_not_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replaces the save action the OpenCV Qt window used to offer, which went
    # away with its (expensive) expanded chrome.
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend(keys=[ord("s"), ord("q")])
    viewer = Viewer(backend, hud=False)
    image, _ = _tooltip_image()
    viewer.show("shots", image.frame())
    shown = backend.shown[-1][1]

    assert viewer.wait() == "q"  # "s" was consumed, so it does not advance
    assert np.array_equal(_saved_bgr(tmp_path / "shots.png"), shown)


def test_save_captures_the_hover_tooltip_exactly_as_displayed(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    viewer = Viewer(backend, hud=False)
    image, _ = _tooltip_image()
    viewer.show("w", image.frame())
    base = backend.shown[-1][1]

    backend.handlers["w"](100, 60, False)  # over the box -> tooltip up
    viewer._flush_hover()
    with_tooltip = backend.shown[-1][1]
    assert not np.array_equal(with_tooltip, base)

    # What lands on disk is the frame on screen, tooltip included -- not the
    # clean base the tooltip is composited over.
    (path,) = viewer.save(tmp_path)
    assert np.array_equal(_saved_bgr(path), with_tooltip)


def test_repeated_saves_are_numbered_rather_than_overwriting(
    tmp_path: Path,
) -> None:
    viewer = Viewer(FakeBackend(), hud=False)
    viewer.show("w", _tooltip_image()[0].frame())

    names = [viewer.save(tmp_path)[0].name for _ in range(3)]
    assert names == ["w.png", "w-1.png", "w-2.png"]


def test_a_frame_is_saved_under_its_own_name_not_the_window_title(
    tmp_path: Path,
) -> None:
    # One window shows every sample in turn, so its title says nothing about
    # which sample is on screen; the caller passes the source image's name.
    viewer = Viewer(FakeBackend(), hud=False, save_dir=tmp_path)
    image, _ = _tooltip_image()

    viewer.show("coco", image.frame(), save_as="000000012345")
    assert [p.name for p in viewer.save()] == ["000000012345.png"]

    viewer.show("coco", image.frame(), save_as="000000067890")
    assert [p.name for p in viewer.save()] == ["000000067890.png"]

    # Without one, the window name still stands in.
    viewer.show("coco", image.frame())
    assert [p.name for p in viewer.save()] == ["coco.png"]


def test_save_dir_is_used_by_the_key_and_created_only_on_the_first_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shots = tmp_path / "luxonis-inspect" / "coco"
    backend = FakeBackend(keys=[ord("s"), ord("q")])
    viewer = Viewer(backend, hud=False, save_dir=shots)
    viewer.show("coco", _tooltip_image()[0].frame(), save_as="street")

    assert viewer.save_dir == shots
    assert not shots.exists()  # a session that never saves leaves nothing

    assert viewer.wait() == "q"
    assert [p.name for p in shots.iterdir()] == ["street.png"]


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ("COCO train", "COCO-train.png"),
        ("shapes/v2: split", "shapes-v2-split.png"),
        ("...", "frame.png"),  # nothing usable survives -> a fallback stem
    ],
)
def test_window_names_become_safe_filenames(
    tmp_path: Path, window: str, expected: str
) -> None:
    viewer = Viewer(FakeBackend(), hud=False)
    viewer.show(window, _tooltip_image()[0].frame())

    assert viewer.save(tmp_path)[0].name == expected


def test_clicking_the_panel_save_row_writes_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    viewer = Viewer(FakeBackend(), hud=False)
    viewer.show("w", _tooltip_image()[0].frame())

    viewer._apply_action("key:s")
    assert (tmp_path / "w.png").exists()


def test_save_key_falls_through_when_there_is_nothing_to_save() -> None:
    # No window open -> the key belongs to the caller rather than being eaten.
    assert Viewer(FakeBackend(), hud=False)._handle_control_key("s") is False


def test_a_failed_save_is_reported_without_killing_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = Viewer(FakeBackend(), hud=False)
    viewer.show("w", _tooltip_image()[0].frame())

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(io, "save", refuse)
    viewer._save_shown()  # reported, not raised


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


def test_frost_weight_covers_the_body_but_not_the_shadow_or_margin() -> None:
    # The frost must fill the colored body at *any* panel opacity, yet leave the
    # drop shadow sharp (the shadow is pure black, so keying on color not alpha
    # avoids a blurred halo around the panel) and the transparent margin sharp.
    # `_CardArrays` folds that decision into the blurred backdrop's weight.
    from luxonis_ml.vizlab.viewer.tooltip_render import _CardArrays

    for body_alpha in (230, 150, 90):  # opaque -> quite translucent panel
        card = np.zeros((80, 120, 4), np.uint8)
        card[5:75, 5:115, 3] = 90  # soft black drop shadow (rgb stays 0)
        card[15:65, 20:100, :3] = (29, 41, 57)  # the colored body fill...
        card[15:65, 20:100, 3] = body_alpha  # ...at the panel's opacity
        card[30:34, 30:80, :3] = (200, 210, 250)  # opaque "text" on the body
        card[30:34, 30:80, 3] = 255
        arrays = _CardArrays(card, blur=12.0)
        assert arrays.frost_weight is not None
        frost = arrays.frost_weight[..., 0]

        # Under the body, the blurred backdrop gets exactly what the panel lets
        # through -- the frost is full-strength wherever the fill is colored.
        assert np.allclose(frost[35:60, 25:95], 1.0 - body_alpha / 255.0)
        # The shadow ring (black, translucent) gets none -> no blurred halo.
        assert not frost[8:12, 40:80].any()
        # Nor does the fully-transparent margin.
        assert not frost[:4, :4].any()
        # And every pixel stays fully accounted for: card + frost + frame == 1.
        alpha = card[..., 3].astype(np.float32) / 255.0
        assert np.allclose(alpha + frost + arrays.frame_weight[..., 0], 1.0)
        # Both weights carry all three channels rather than broadcasting from a
        # length-1 trailing axis. NumPy multiplies by such an axis an order of
        # magnitude slower, and these are multiplied over the card per redraw.
        assert arrays.frost_weight.shape == (*card.shape[:2], 3)
        assert arrays.frame_weight.shape == (*card.shape[:2], 3)


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
    from luxonis_ml.vizlab.render.capture import ClickMap

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
    from luxonis_ml.vizlab.render.capture import ClickMap

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


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ord("c"), "c"),  # plain ASCII
        (0x100000 | ord("c"), "c"),  # GTK/Qt: modifier bits above the low byte
        (-1, ""),  # no key
        (65379, ""),  # Insert  (X11 keysym 0xFF63 -> would alias to 'c')
        (65361, ""),  # Left    (0xFF51 -> 'Q')
        (65362, ""),  # Up      (0xFF52 -> 'R')
        (65363, ""),  # Right   (0xFF53 -> 'S')
        (65364, ""),  # Down    (0xFF54 -> 'T')
        (65360, ""),  # Home    (0xFF50 -> 'P')
    ],
)
def test_key_char_ignores_special_keysyms(code: int, expected: str) -> None:
    """Special keys must not alias onto a letter that drives a layer control.

    OpenCV's GTK/Qt builds return characters with state bits set above the low
    byte, so the low byte identifies the key — but special keys arrive as X11
    keysyms in 0xFF00-0xFFFF, whose low byte is an unrelated letter.
    """
    assert _key_char(code) == expected


def test_special_keys_do_not_toggle_layers() -> None:
    backend = FakeBackend()
    viewer = Viewer(backend)
    viewer.show(
        "w",
        _layer_image().frame(),
        render=lambda _state: _layer_image().frame(),
    )
    viewer.run(lambda _key: None)
    assert backend.key_handler is not None

    backend.key_handler(65379)  # Insert: used to read as 'c' and isolate
    backend.key_handler(65362)  # Up: used to read as 'R'
    assert viewer.layers.is_default()
