"""Coverage for the clip writer: every format, and the ways frames disagree."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageSequence

from luxonis_ml.vizlab import (
    VIDEO_FORMATS,
    BBox,
    Image,
    VideoWriter,
    is_video_path,
    save_video,
)
from luxonis_ml.vizlab.video import _ANIMATION_FORMATS

_VIDEO = tuple(
    suffix for suffix in VIDEO_FORMATS if suffix not in _ANIMATION_FORMATS
)


def _ramp(count: int, width: int = 96, height: int = 64) -> list[np.ndarray]:
    """Build frames that differ from each other, so encoders do real work."""
    frames = []
    for index in range(count):
        frame = np.full((height, width, 3), 20, dtype=np.uint8)
        frame[:, : int(width * (index + 1) / count)] = 200
        frames.append(frame)
    return frames


def _count_video_frames(path: Path) -> int:
    """Decode ``path`` and count the frames that actually come back."""
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def _read_frames(path: Path) -> int:
    """Frame count of a clip, whichever family it belongs to."""
    if path.suffix in _ANIMATION_FORMATS:
        with PILImage.open(path) as animation:
            return sum(1 for _ in ImageSequence.Iterator(animation))
    return _count_video_frames(path)


@pytest.mark.parametrize("suffix", VIDEO_FORMATS)
def test_every_format_round_trips_its_frames(
    suffix: str, tmp_path: Path
) -> None:
    path = save_video(_ramp(5), tmp_path / f"clip{suffix}", fps=8)
    assert path.stat().st_size > 0
    assert _read_frames(path) == 5


@pytest.mark.parametrize("suffix", VIDEO_FORMATS)
def test_writer_reports_what_it_negotiated(
    suffix: str, tmp_path: Path
) -> None:
    with VideoWriter(tmp_path / f"clip{suffix}", fps=8) as clip:
        clip.extend(_ramp(3))
    assert len(clip) == 3
    assert clip.codec  # a FourCC, or the Pillow format name
    assert clip.size == (96, 64)
    assert repr(clip).startswith("VideoWriter(")


@pytest.mark.parametrize("suffix", _VIDEO)
def test_odd_sizes_are_padded_rather_than_silently_cropped(
    suffix: str, tmp_path: Path
) -> None:
    # Video codecs work on 2x2 chroma blocks and crop an odd dimension *down*
    # without saying so, losing the render's last row and column. The writer
    # rounds up instead, so nothing drawn at the edge disappears.
    path = tmp_path / f"odd{suffix}"
    with VideoWriter(path, fps=5) as clip:
        clip.add(np.full((65, 101, 3), 90, dtype=np.uint8))
    assert clip.size == (102, 66)
    capture = cv2.VideoCapture(str(path))
    decoded = capture.read()[1]
    capture.release()
    assert decoded.shape[:2] == (66, 102)


def test_animated_formats_keep_odd_sizes_exactly(tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "odd.webp", fps=5) as clip:
        clip.add(np.full((65, 101, 3), 90, dtype=np.uint8))
    assert clip.size == (101, 65)


def test_smaller_frames_are_letterboxed_not_stretched(tmp_path: Path) -> None:
    path = tmp_path / "mixed.mp4"
    with VideoWriter(path, fps=5, background="#000000") as clip:
        clip.add(np.full((100, 200, 3), 90, dtype=np.uint8))  # 2:1 sets it
        clip.add(np.full((200, 200, 3), 220, dtype=np.uint8))  # 1:1 must fit
    assert clip.size == (200, 100)

    capture = cv2.VideoCapture(str(path))
    capture.read()
    second = capture.read()[1]
    capture.release()
    # A square scaled into a 2:1 canvas fills the height and is pillarboxed,
    # which is only true if the aspect ratio survived.
    assert second[:, 2].max() < 60  # background at the left edge
    lit = np.where(second[:, 100, 0] > 150)[0]
    assert (lit.min(), lit.max()) == (0, 99)


def test_an_explicit_size_wins_over_the_first_frame(tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "fixed.mp4", size=(64, 48)) as clip:
        clip.add(np.full((200, 200, 3), 90, dtype=np.uint8))
    assert clip.size == (64, 48)


def test_renderables_and_frames_are_both_accepted(tmp_path: Path) -> None:
    scene = Image(np.zeros((60, 90, 3), np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.5, label="car")
    )
    with VideoWriter(tmp_path / "scenes.webp", fps=5) as clip:
        clip.add(scene)  # a Renderable
        clip.add(scene.frame())  # a Frame, whose hit maps a file cannot carry
    assert len(clip) == 2
    assert clip.size == (90, 60)


def test_video_flattens_transparency_onto_the_background(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clear.mp4"
    transparent = np.zeros((20, 20, 4), dtype=np.uint8)
    with VideoWriter(path, background="#ff0000") as clip:
        clip.add(transparent)
    capture = cv2.VideoCapture(str(path))
    decoded = capture.read()[1]
    capture.release()
    blue, green, red = (int(value) for value in decoded[10, 10])
    assert red > 200
    assert blue < 40
    assert green < 40


def test_animations_keep_transparency(tmp_path: Path) -> None:
    path = tmp_path / "clear.webp"
    with VideoWriter(path, background="#ff0000") as clip:
        clip.add(np.zeros((20, 20, 4), dtype=np.uint8))
    decoded = np.array(PILImage.open(path).convert("RGBA"))
    assert decoded[10, 10, 3] == 0


def test_an_empty_clip_says_so_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "empty.mp4"
    with pytest.raises(ValueError, match="no frames were added"):  # noqa: SIM117
        with VideoWriter(path):
            pass
    assert not path.exists()


def test_a_failure_in_the_body_is_not_masked_by_the_empty_clip(
    tmp_path: Path,
) -> None:
    # __exit__ closes quietly while an exception is propagating, so the caller
    # sees what actually went wrong instead of a complaint about zero frames.
    with pytest.raises(KeyError, match="the real failure"):  # noqa: SIM117
        with VideoWriter(tmp_path / "masked.mp4"):
            raise KeyError("the real failure")


def test_close_is_idempotent(tmp_path: Path) -> None:
    clip = VideoWriter(tmp_path / "twice.mp4")
    clip.add(np.zeros((8, 8, 3), np.uint8))
    clip.close()
    clip.close()  # no second release, no error
    assert len(clip) == 1


def test_adding_after_close_is_refused(tmp_path: Path) -> None:
    with VideoWriter(tmp_path / "shut.mp4") as clip:
        clip.add(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(RuntimeError, match="already closed"):
        clip.add(np.zeros((8, 8, 3), np.uint8))


def test_unsupported_extensions_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported video format"):
        VideoWriter(tmp_path / "clip.txt")


def test_a_still_png_destination_points_at_apng(tmp_path: Path) -> None:
    # '.png' is a still elsewhere in vizlab, so the error names the animated
    # spelling rather than just listing every format.
    with pytest.raises(ValueError, match=r"use '\.apng'"):
        VideoWriter(tmp_path / "clip.png")


def test_a_nonpositive_frame_rate_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        VideoWriter(tmp_path / "clip.mp4", fps=0)


def test_an_unusable_codec_reports_what_ffmpeg_said(tmp_path: Path) -> None:
    # 'xyzw' is not a codec anywhere, so this exercises the failure path
    # regardless of which encoders the local OpenCV build happens to ship.
    with pytest.raises(RuntimeError, match="no usable encoder") as failure:  # noqa: SIM117
        with VideoWriter(tmp_path / "bad.mp4", codec="xyzw") as clip:
            clip.add(np.zeros((16, 16, 3), np.uint8))
    assert "xyzw" in str(failure.value)


def test_a_working_codec_leaves_no_chatter_on_stderr(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    # OpenCV's FFmpeg backend warns that the VP8 tag "is not supported" and
    # then encodes VP8 correctly. That goes to the C-level stderr, so only a
    # file-descriptor capture sees it.
    with VideoWriter(tmp_path / "quiet.webm", fps=5) as clip:
        clip.extend(_ramp(2))
    assert "not supported" not in capfd.readouterr().err


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("clip.mp4", True),
        ("clip.AVIF", True),
        ("renders", False),
        ("render.png", False),
        ("render.svg", False),
    ],
)
def test_is_video_path_classifies_destinations(
    path: str, expected: bool
) -> None:
    assert is_video_path(path) is expected


def test_save_video_rejects_an_empty_sequence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames were added"):
        save_video([], tmp_path / "nothing.gif")
