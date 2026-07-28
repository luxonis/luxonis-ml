"""Tests for the notebook `WindowBackend` (the ipywidgets-free logic)."""

import importlib.util
import io

import numpy as np
import pytest

from luxonis_ml.vizlab.viewer import NotebookBackend
from luxonis_ml.vizlab.viewer.notebook_backend import (
    _encode_png,
    _relative_to_frame,
)

_HAS_IPYWIDGETS = importlib.util.find_spec("ipywidgets") is not None
_HAS_IPYEVENTS = importlib.util.find_spec("ipyevents") is not None


def test_encode_png_swaps_bgr_to_rgb() -> None:
    from PIL import Image as PILImage

    frame = np.zeros((10, 20, 3), np.uint8)
    frame[..., 0] = 255  # blue channel in BGR
    data = _encode_png(frame)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature
    decoded = np.asarray(PILImage.open(io.BytesIO(data)).convert("RGB"))
    # BGR blue -> RGB (0, 0, 255): stays blue on screen.
    assert tuple(int(c) for c in decoded[0, 0]) == (0, 0, 255)


def test_relative_to_frame_scales_by_displayed_size() -> None:
    # Element shown at 200x120 but the frame is 100x60 -> halve the coords.
    assert _relative_to_frame(100, 60, 200, 120, 100, 60) == (50, 30)


def test_relative_to_frame_passthrough_without_rect() -> None:
    assert _relative_to_frame(7, 8, 0, 0, 100, 60) == (7, 8)


def test_emit_key_delivers_code_to_handler() -> None:
    backend = NotebookBackend()
    got: list[int] = []
    backend.set_key_handler(got.append)
    backend._emit_key("n")
    assert got == [ord("n")]


def test_control_labels_map_to_keys() -> None:
    backend = NotebookBackend(controls={"Go": "g"})
    got: list[int] = []
    backend.set_key_handler(got.append)
    backend._emit_key(backend._controls["Go"])
    assert got == [ord("g")]


def test_poll_key_is_push_only() -> None:
    assert NotebookBackend().poll_key(0) == -1


def test_screen_size_is_configurable() -> None:
    assert NotebookBackend(screen=(800, 600)).screen_size() == (800, 600)
    assert NotebookBackend().screen_size() is None


@pytest.mark.skipif(
    _HAS_IPYWIDGETS, reason="ipywidgets is installed in this environment"
)
def test_create_window_requires_ipywidgets() -> None:
    with pytest.raises(ImportError, match="ipywidgets"):
        NotebookBackend().create_window("w")


@pytest.mark.skipif(
    not _HAS_IPYWIDGETS, reason="needs the 'notebook' extra (ipywidgets)"
)
def test_show_sets_png_on_real_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise the real ipywidgets path (create + encode + assign) with the
    # frontend `display` stubbed out, so it runs headless in CI.
    import IPython.display

    from luxonis_ml.vizlab import Image
    from luxonis_ml.vizlab.viewer import Viewer

    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)
    backend = NotebookBackend()
    viewer = Viewer(backend)
    image = Image(np.zeros((40, 60, 3), np.uint8))
    viewer.show("w", image.frame())

    widget = backend._images["w"]
    assert widget.value[:8] == b"\x89PNG\r\n\x1a\n"
    assert widget.layout.width == "60px"


@pytest.mark.skipif(
    not (_HAS_IPYWIDGETS and _HAS_IPYEVENTS),
    reason="needs the 'notebook' extra (ipywidgets + ipyevents)",
)
def test_mouse_events_route_move_and_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A DOM move dispatches with clicked=False (hover), a click with True (panel
    # interaction); both are scaled from the displayed element size.
    import ipyevents
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: None)

    captured: dict = {}

    class FakeEvent:
        def __init__(self, *, source: object, watched_events: list) -> None:
            captured["watched"] = watched_events

        def on_dom_event(self, callback: object) -> None:
            captured["callback"] = callback

    monkeypatch.setattr(ipyevents, "Event", FakeEvent)

    backend = NotebookBackend()
    backend.create_window("w")
    backend._frame_size["w"] = (100, 60)
    calls: list[tuple[int, int, bool]] = []
    backend.set_mouse_handler(
        "w", lambda x, y, clicked: calls.append((x, y, clicked))
    )

    dispatch = captured["callback"]
    rect = {"boundingRectWidth": 200, "boundingRectHeight": 120}
    dispatch({"type": "mousemove", "relativeX": 100, "relativeY": 60, **rect})
    dispatch({"type": "click", "relativeX": 20, "relativeY": 40, **rect})

    assert {"mousemove", "click"} <= set(captured["watched"])
    assert calls == [(50, 30, False), (10, 20, True)]  # scaled, click flagged
