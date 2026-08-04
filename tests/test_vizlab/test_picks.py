"""Clicking an annotation: source capture, viewer dispatch, and the clipboard."""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from luxonis_ml.ldf import BBoxAnnotation, Detection, KeypointAnnotation
from luxonis_ml.vizlab import BBox, Image, PickMap
from luxonis_ml.vizlab.adapters.ldf import detection_to_annotations
from luxonis_ml.vizlab.geometry import Rect
from luxonis_ml.vizlab.interaction.frame import Frame
from luxonis_ml.vizlab.render.capture import ClickMap
from luxonis_ml.vizlab.viewer import Viewer, clipboard, report_pick

from .test_viewer import FakeBackend

#: A car with a nested plate, so a click can land on either.
CAR = Detection(
    class_name="car",
    instance_id=4,
    metadata={"color": "red"},
    boundingbox=BBoxAnnotation(x=0.1, y=0.2, w=0.4, h=0.4),
    sub_detections={
        "plate": Detection(
            class_name="plate",
            boundingbox=BBoxAnnotation(x=0.15, y=0.45, w=0.1, h=0.1),
        )
    },
)


#: Rich colorizes its JSON when the session looks like a terminal, which pytest
#: capture does not reliably suppress (``FORCE_COLOR`` alone flips it).
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _printed_json(captured: str) -> Any:
    """Parse what `report_pick` wrote, colored or not."""
    return json.loads(_ANSI.sub("", captured))


def _writer(target: Path) -> list[str]:
    """Build a writer command that saves stdin to ``target`` instead."""
    return [
        sys.executable,
        "-c",
        "import pathlib, sys;"
        " pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
        str(target),
    ]


def _scene(detection: Detection, size: int = 200) -> Image:
    """Build a render-ready image carrying one detection's annotations."""
    image = Image(np.zeros((size, size, 3), np.uint8))
    for annotation in detection_to_annotations(detection):
        image.add(annotation)
    return image


def test_a_boxed_detection_carries_the_annotation_it_was_drawn_from() -> None:
    box = detection_to_annotations(CAR)[0]

    assert box.source == CAR.model_dump(
        mode="json", exclude_none=True, exclude_defaults=True
    )
    # The dump is LDF's own shape, so it can be pasted back into a generator.
    assert Detection.model_validate(box.source) == CAR


def test_a_boxless_detection_puts_its_source_on_its_shapes() -> None:
    detection = Detection(
        class_name="face",
        keypoints=KeypointAnnotation(keypoints=[(0.2, 0.3, 2)]),
    )

    annotations = detection_to_annotations(detection)

    assert annotations
    assert all(a.source is not None for a in annotations)


def test_parts_of_a_boxed_detection_defer_to_the_box() -> None:
    # The box already answers a click anywhere inside it, so its keypoints and
    # mask would only repeat the same payload through a smaller rectangle.
    detection = Detection(
        class_name="face",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.5, h=0.5),
        keypoints=KeypointAnnotation(keypoints=[(0.2, 0.3, 2)]),
    )

    box = detection_to_annotations(detection)[0]

    assert box.source is not None
    assert [child.source for child in box.children] == [None]


def test_a_click_resolves_to_the_smallest_annotation_under_it() -> None:
    frame = _scene(CAR).frame()

    car = frame.pickmap.hit(0.4 * 200, 0.3 * 200)
    plate = frame.pickmap.hit(0.2 * 200, 0.5 * 200)

    assert isinstance(car, dict)
    assert isinstance(plate, dict)
    # The parent's JSON is the whole subtree; the nested box answers with its own.
    assert car["class_name"] == "car"
    assert "sub_detections" in car
    assert plate["class_name"] == "plate"
    assert "sub_detections" not in plate


def test_clicking_the_background_picks_nothing() -> None:
    frame = _scene(CAR).frame()

    assert frame.pickmap.hit(199, 199) is None


def test_attaching_a_panel_shifts_the_pick_regions_with_the_image() -> None:
    frame = _scene(CAR).frame()
    before = frame.pickmap.items[0][0]

    shifted = frame.with_panel({"frame": 3}).pickmap.items[0][0]

    assert shifted.left > before.left
    assert shifted.right - shifted.left == pytest.approx(
        before.right - before.left
    )


def test_a_baked_scene_keeps_answering_clicks() -> None:
    # What ``data compare`` does when it flattens a composite to add its legend.
    frame = _scene(CAR).frame()

    baked = Image(frame.render()).with_pickmap(frame.pickmap)

    assert baked.frame().pickmap.items == frame.pickmap.items


def test_screen_fitting_scales_the_pick_regions() -> None:
    frame = _scene(CAR, size=1000).frame()
    # A screen this small forces the viewer to render the frame at half size.
    viewer = Viewer(FakeBackend(screen=(556, 556)), hud=False)

    prepared = viewer.prepare(frame)

    native = frame.pickmap.items[0][0]
    fitted = prepared.pickmap.items[0][0]
    assert fitted.left == pytest.approx(native.left / 2, abs=1.0)
    assert len(prepared.pickmap.items) == len(frame.pickmap.items)


def test_clicking_an_annotation_reports_its_source() -> None:
    picked: list[Any] = []
    backend = FakeBackend(keys=[-1, ord("q")])
    viewer = Viewer(backend, hud=False, on_pick=picked.append)
    viewer.show("w", _scene(CAR).frame())

    backend.handlers["w"](int(0.4 * 200), int(0.3 * 200), True)
    viewer.wait()  # -1 -> report the queued pick, then "q"

    assert len(picked) == 1
    assert picked[0]["class_name"] == "car"


def test_a_driven_backend_reports_the_click_immediately() -> None:
    picked: list[Any] = []
    backend = FakeBackend()
    viewer = Viewer(backend, hud=False, on_pick=picked.append)
    frame = _scene(CAR).frame()
    viewer.show("w", frame, render=lambda _layers: frame)
    viewer.run(lambda _key: None)

    backend.handlers["w"](int(0.4 * 200), int(0.3 * 200), True)

    assert len(picked) == 1


def test_a_control_wins_over_the_annotation_beneath_it() -> None:
    picked: list[Any] = []
    backend = FakeBackend(keys=[-1, ord("q")])
    viewer = Viewer(backend, hud=False, on_pick=picked.append)
    viewer.layers.classes = ("car",)
    frame = Frame(
        _scene(CAR),
        clickmap=ClickMap([(Rect(0, 0, 200, 200), "class:car")]),
        pickmap=_scene(CAR).frame().pickmap,
    )
    viewer.show("w", frame, render=lambda _layers: frame)

    backend.handlers["w"](int(0.4 * 200), int(0.3 * 200), True)
    viewer.wait()

    assert viewer.layers.hidden == {"car"}
    assert picked == []


def test_a_failing_pick_handler_does_not_end_the_session() -> None:
    def refuse(_source: object) -> None:
        raise RuntimeError("no clipboard for you")

    backend = FakeBackend(keys=[-1, ord("q")])
    viewer = Viewer(backend, hud=False, on_pick=refuse)
    viewer.show("w", _scene(CAR).frame())

    backend.handlers["w"](int(0.4 * 200), int(0.3 * 200), True)

    assert viewer.wait() == "q"


def test_report_pick_prints_the_source_and_queues_the_copy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    queued: list[str] = []
    monkeypatch.setattr(clipboard, "copy_later", queued.append)

    report_pick({"class_name": "car", "score": 0.5})

    expected = {"class_name": "car", "score": 0.5}
    assert _printed_json(capsys.readouterr().out) == expected
    assert json.loads(queued[0]) == expected


def test_a_mask_is_summarized_in_print_but_copied_whole(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Run-length mask data is ~99% of a segmented detection's JSON and hundreds
    # of wrapped terminal lines, yet it is what makes the copy reproduce the
    # mask: unreadable, so elided in print; essential, so copied in full.
    mask = np.zeros((240, 320), dtype=bool)
    mask[40:200, 60:260] = True
    mask[::5, ::3] = True
    detection = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.1, y=0.2, w=0.4, h=0.4),
        instance_segmentation={"mask": mask},  # type: ignore[arg-type]
    )
    queued: list[str] = []
    monkeypatch.setattr(clipboard, "copy_later", queued.append)

    report_pick(detection_to_annotations(detection)[0].source)

    printed = _printed_json(capsys.readouterr().out)
    counts = printed["instance_segmentation"]["counts"]
    assert "copied in full" in counts
    assert len(counts) < 200
    # The class and geometry — what the click was about — survive intact.
    assert printed["class_name"] == "car"
    assert printed["boundingbox"]["w"] == 0.4
    # And the clipboard still holds an annotation, not a description of one.
    assert Detection.model_validate(json.loads(queued[0])) == detection


def test_ordinary_metadata_is_never_elided(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    note = "reviewed by hand; " * 5  # long-ish, but still something to read
    monkeypatch.setattr(clipboard, "copy_later", lambda _text: None)

    report_pick({"class_name": "car", "metadata": {"note": note}})

    assert _printed_json(capsys.readouterr().out)["metadata"]["note"] == note


def test_report_pick_stringifies_what_json_cannot_represent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(clipboard, "copy_later", lambda _text: None)
    # An array is not JSON, and a click must not raise over it.
    source: Any = {"mask": np.zeros((2, 2))}

    report_pick(source)

    assert "mask" in capsys.readouterr().out


def test_report_pick_returns_before_the_clipboard_does(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Regression: the copy ran inline, and a clipboard helper that forks to own
    # the selection held the viewer's loop for seconds on every click.
    started = threading.Event()

    def slow(_text: str) -> str:
        started.set()
        time.sleep(1)
        return "slow"

    monkeypatch.setattr(clipboard, "copy", slow)

    elapsed = time.perf_counter()
    report_pick({"class_name": "car"})
    elapsed = time.perf_counter() - elapsed

    capsys.readouterr()
    assert started.wait(5), "the queued clipboard write never ran"
    assert elapsed < 0.5


def test_only_the_newest_of_several_queued_copies_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Clicking twice in quick succession must leave the *second* annotation on
    # the clipboard, not whichever helper won the race.
    release = threading.Event()
    written: list[str] = []
    done = threading.Event()

    def take(text: str) -> str:
        # Hold the writer until every text is queued, so they pile up behind it.
        release.wait(5)
        written.append(text)
        if text == "third":
            done.set()
        return "fake"

    monkeypatch.setattr(clipboard, "copy", take)

    clipboard.copy_later("first")
    clipboard.copy_later("second")
    clipboard.copy_later("third")
    release.set()

    assert done.wait(5), "the queued clipboard writes never ran"
    assert "second" not in written


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="needs fork to imitate a clipboard helper"
)
def test_copy_does_not_wait_on_a_helper_that_holds_its_output(
    tmp_path: Path,
) -> None:
    # What ``wl-copy`` does: fork a child that owns the selection and inherits
    # the parent's stdout. Reading that to EOF waits for the child, not the
    # copy — which is how a click came to block for the whole 5s timeout.
    target = tmp_path / "held.txt"
    forking = [
        sys.executable,
        "-c",
        "import os, pathlib, sys, time;"
        " data = sys.stdin.buffer.read();"
        " sys.exit(0) if os.fork() else None;"
        " time.sleep(30);"
        " pathlib.Path(sys.argv[1]).write_bytes(data)",
        str(target),
    ]

    elapsed = time.perf_counter()
    tool = clipboard.copy("x", writers=[forking])
    elapsed = time.perf_counter() - elapsed

    assert tool == sys.executable
    assert elapsed < 3


def test_copy_hands_the_text_to_the_first_working_tool(
    tmp_path: Path,
) -> None:
    target = tmp_path / "clipboard.txt"

    tool = clipboard.copy("π = 3.14", writers=[_writer(target)])

    assert tool == sys.executable
    assert target.read_text(encoding="utf-8") == "π = 3.14"


def test_copy_falls_through_a_tool_that_fails(tmp_path: Path) -> None:
    target = tmp_path / "clipboard.txt"
    failing = [sys.executable, "-c", "raise SystemExit(3)"]

    assert clipboard.copy("x", writers=[failing, _writer(target)])
    assert target.read_text(encoding="utf-8") == "x"


def test_copy_reports_that_no_tool_was_available() -> None:
    assert (
        clipboard.copy("x", writers=[["definitely-not-a-real-tool"]]) is None
    )
    assert clipboard.copy("x", writers=[]) is None


def test_an_annotation_without_a_source_is_not_pickable() -> None:
    image = Image(np.zeros((50, 50, 3), np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.5, label="car")
    )

    assert image.frame().pickmap == PickMap.empty()
