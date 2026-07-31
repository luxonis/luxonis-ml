"""Write a sequence of rendered scenes to a video or an animated image.

`Renderable.save` writes one scene to one file. `VideoWriter` is the same idea
over a sequence: feed it frames — `Renderable`, `Frame`, or anything
`luxonis_ml.vizlab.io.load_rgba` accepts — and it encodes them into a single
playable file. The destination's extension picks the format, exactly as it does
for stills.

Two families are supported, and which one you want depends on where the result
is going to be watched:

==========  ===========================================================
``.mp4``    H.264 when the OpenCV build has an encoder, MPEG-4 Part 2
            otherwise. The fastest option and the one every desktop
            player opens. ``.mov`` is the same encoders in a QuickTime
            container.
``.webm``   VP8. The only *video* here that plays inline in a browser
            (and in a GitHub comment), at roughly a twentieth of the
            encoding speed of ``.mp4``.
``.avi``    Motion JPEG — every frame encoded independently, so it
            scrubs and seeks frame-exactly at the cost of size.
``.mkv``    FFV1, mathematically lossless. For archival or for diffing
            renders frame by frame; expect very large files.
``.gif``    256 colors, no real alpha, understood by everything. Big
            and banded, but it plays where nothing else does.
``.webp``   Animated WebP: full RGBA, roughly half the size of the
            equivalent GIF, and rendered inline by browsers.
``.apng``   Animated PNG: lossless RGBA. The one to reach for when the
            frames must survive byte-exact.
``.avif``   Animated AVIF: by far the smallest — an order of magnitude
            under GIF at the same resolution — with full RGBA. Newer,
            so check that your viewer supports it.
==========  ===========================================================

The last four are *animated images* rather than videos. They keep the alpha
channel (`Renderable.render` produces RGBA), they loop forever by default, and
Pillow's encoders need every frame at once — so the whole sequence is held in
memory until `VideoWriter.close`. That is fine for the short clips those formats
are good at and a poor fit for a few thousand frames; use ``.mp4`` or ``.webm``
for a whole dataset, which stream frame by frame and stay flat in memory.

Frames do not have to agree on size. The first frame (or an explicit ``size``)
fixes the canvas, and later frames are scaled to fit and centered on the theme
background rather than stretched, so nothing is ever distorted. Feeding frames
that already share a size skips the resampling entirely.

.. image:: TODO-HOST/motion.webp
   :alt: A tracked detection moving across a street frame, as an animation.

Examples:
    >>> import numpy as np
    >>> from luxonis_ml.vizlab import BBox, Image, save_video
    >>> frames = [
    ...     Image(np.zeros((64, 96, 3), np.uint8)).add(
    ...         BBox(x=0.1 * i, y=0.2, w=0.3, h=0.4, label="car")
    ...     )
    ...     for i in range(4)
    ... ]
    >>> save_video(frames, "detections.mp4", fps=12).name
    'detections.mp4'

    A `Frame` works the same way, so a hoverable scene and its metadata panel
    can be recorded straight from the loop that would have displayed it.

"""

import os
import sys
import tempfile
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from typing_extensions import Self

from luxonis_ml.vizlab import io
from luxonis_ml.vizlab.color import Color
from luxonis_ml.vizlab.options import current_options

if TYPE_CHECKING:
    import cv2
    from PIL import Image as PILImage

    from luxonis_ml.vizlab.color import ColorLike
    from luxonis_ml.vizlab.io import ImageSource

#: Video containers, mapped to the FourCC codecs tried in preference order.
#: ``avc1`` is first for MP4 because H.264 is the universally playable choice,
#: but many OpenCV wheels ship an FFmpeg without an H.264 *encoder* (decoding is
#: unaffected), so ``mp4v`` is kept as the fallback that always works.
_VIDEO_CODECS: "dict[str, tuple[str, ...]]" = {
    ".mp4": ("avc1", "mp4v"),
    ".mov": ("avc1", "mp4v"),
    ".webm": ("VP80",),
    ".avi": ("MJPG",),
    ".mkv": ("FFV1",),
}

#: Animated image formats, mapped to the Pillow format they are saved as.
_ANIMATION_FORMATS: "dict[str, str]" = {
    ".gif": "GIF",
    ".webp": "WEBP",
    ".apng": "PNG",
    ".avif": "AVIF",
}

VIDEO_FORMATS: "tuple[str, ...]" = (
    *_VIDEO_CODECS,
    *_ANIMATION_FORMATS,
)
"""Every file extension `VideoWriter` can write, video and animation alike."""


@runtime_checkable
class _Renderish(Protocol):
    """Anything that renders itself to an array — `Renderable` or `Frame`."""

    def render(self, size: "tuple[int, int] | None" = None) -> np.ndarray: ...


def is_video_path(path: "str | Path") -> bool:
    """Whether ``path`` names a format `VideoWriter` can write.

    Lets a caller that accepts either an output *directory* or an output *clip*
    tell the two apart by extension.

    Args:
        path: The destination path to classify.

    Returns:
        ``True`` if the extension is one of `VIDEO_FORMATS`.

    Examples:
        >>> is_video_path("clip.mp4"), is_video_path("renders/")
        (True, False)

    """
    return Path(path).suffix.lower() in VIDEO_FORMATS


@contextmanager
def _muted_stderr(captured: "list[str]") -> "Generator[None]":
    """Divert file descriptor 2, appending what was written to ``captured``.

    OpenCV's FFmpeg backend writes straight to the C-level ``stderr`` rather
    than through Python, and it is chatty even when it succeeds — opening a
    WebM prints ``tag 'VP80' is not supported`` and then encodes VP8 perfectly
    well. Swallowing that keeps a working command quiet; the text is handed
    back so a *failed* open can quote it in the exception instead of losing it.
    """
    try:
        saved = os.dup(2)
    except (AttributeError, OSError):  # pragma: no cover - platform guard
        yield
        return
    try:
        with tempfile.TemporaryFile(mode="w+") as sink:
            try:
                sys.stderr.flush()
                os.dup2(sink.fileno(), 2)
                yield
            finally:
                sys.stderr.flush()
                os.dup2(saved, 2)
                sink.seek(0)
                captured.append(sink.read())
    finally:
        os.close(saved)


def _flatten(rgba: np.ndarray, background: Color) -> np.ndarray:
    """Composite ``rgba`` over ``background``, or pass it through if opaque."""
    alpha = rgba[:, :, 3]
    if bool((alpha == 255).all()):
        return rgba
    weight = (alpha / 255.0)[:, :, None]
    base = np.asarray(background.rgb, dtype=np.float32)
    blended = rgba[:, :, :3] * weight + base * (1.0 - weight)
    return np.dstack(
        [blended.astype(np.uint8), np.full(alpha.shape, 255, np.uint8)]
    )


def _resize(rgba: np.ndarray, size: "tuple[int, int]") -> np.ndarray:
    """Resample an RGBA array to ``(width, height)``."""
    from PIL import Image as PILImage

    resized = PILImage.fromarray(rgba, mode="RGBA").resize(
        size, PILImage.Resampling.LANCZOS
    )
    return np.asarray(resized)


def _fit(
    rgba: np.ndarray, size: "tuple[int, int]", background: Color
) -> np.ndarray:
    """Scale ``rgba`` to fit ``size`` and center it on ``background``.

    Aspect ratio is preserved, so a frame whose shape disagrees with the clip
    gets bars rather than a stretch.
    """
    width, height = size
    source_height, source_width = rgba.shape[:2]
    if (source_width, source_height) == size:
        return rgba
    scale = min(width / source_width, height / source_height)
    scaled = (
        max(1, min(width, round(source_width * scale))),
        max(1, min(height, round(source_height * scale))),
    )
    canvas = np.empty((height, width, 4), dtype=np.uint8)
    canvas[:, :] = background.rgba
    left = (width - scaled[0]) // 2
    top = (height - scaled[1]) // 2
    canvas[top : top + scaled[1], left : left + scaled[0]] = _resize(
        rgba, scaled
    )
    return canvas


class VideoWriter:
    """An open clip that frames are appended to, one render at a time.

    The extension of ``path`` selects the encoder (see the module docstring).
    Use it as a context manager so the file is finalized even if the loop
    raises; a clip that never receives a frame writes no file at all.

    Attributes:
        path: The destination the clip is written to.
        codec: The FourCC actually negotiated for a video container, or the
            Pillow format name for an animated image.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import VideoWriter
        >>> with VideoWriter("clip.webp", fps=8) as clip:
        ...     for shade in (40, 120, 200):
        ...         _ = clip.add(np.full((32, 32, 3), shade, np.uint8))
        >>> len(clip)
        3

    """

    def __init__(
        self,
        path: "str | Path",
        *,
        fps: float = 10.0,
        size: "tuple[int, int] | None" = None,
        codec: "str | None" = None,
        quality: int = 90,
        loop: int = 0,
        background: "ColorLike | None" = None,
    ) -> None:
        """Open a clip for writing.

        Args:
            path: Destination file; its extension picks the format and must be
                one of `VIDEO_FORMATS`.
            fps: Frames per second. Animated images store a per-frame duration
                instead, so a fractional rate survives there but GIF rounds it
                to the nearest hundredth of a second.
            size: ``(width, height)`` every frame is fitted to. Defaults to the
                first frame's own size.
            codec: FourCC override for video containers, such as ``"VP90"`` for
                a smaller (much slower) WebM. Ignored by animated images.
            quality: Encoder quality from 0 to 100, used by the lossy animated
                formats (WebP and AVIF).
            loop: How many times an animated image repeats; ``0`` means
                forever. Ignored by video containers.
            background: Color shown behind frames that do not fill the clip,
                and behind transparency in the video formats, which have no
                alpha channel. Defaults to the active theme's background.

        Raises:
            ValueError: If the extension is not a supported format, or if
                ``fps`` is not positive.

        """
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix not in VIDEO_FORMATS:
            expected = ", ".join(VIDEO_FORMATS)
            extra = (
                " (use '.apng' to write an animated PNG)"
                if suffix == ".png"
                else ""
            )
            raise ValueError(
                f"unsupported video format {suffix!r}{extra}: "
                f"expected one of {expected}"
            )
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self.codec: str | None = None
        self._suffix = suffix
        self._animated = suffix in _ANIMATION_FORMATS
        self._fps = fps
        self._size = size if size is None else self._canvas(size)
        self._requested_codec = codec
        self._quality = quality
        self._loop = loop
        self._background = (
            Color.parse(background)
            if background is not None
            else current_options().theme.background
        )
        self._writer: cv2.VideoWriter | None = None
        self._buffer: list[PILImage.Image] = []
        self._count = 0
        self._closed = False

    def _canvas(self, size: "tuple[int, int]") -> "tuple[int, int]":
        """Round a frame size up to what this format can actually store.

        Video codecs work on 2x2 chroma blocks and quietly *crop* an odd width
        or height down to the next even number, silently losing the last row or
        column of the render. Rounding up instead pads it with background. The
        animated image formats have no such constraint.
        """
        width, height = size
        if self._animated:
            return size
        return (width + width % 2, height + height % 2)

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            This writer.

        """
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        """Finalize the clip, quietly if the body is already raising."""
        self.close(quiet=exc_type is not None)

    def __len__(self) -> int:
        """Count the frames appended so far."""
        return self._count

    def __repr__(self) -> str:
        """Show the destination, frame count, and negotiated codec."""
        codec = f", codec={self.codec!r}" if self.codec else ""
        return f"VideoWriter({self.path.name!r}, frames={self._count}{codec})"

    @property
    def size(self) -> "tuple[int, int] | None":
        """The clip's ``(width, height)``, once the first frame has fixed it."""
        return self._size

    def add(self, frame: "_Renderish | ImageSource") -> Self:
        """Append one frame to the clip.

        Args:
            frame: A `Renderable` or `Frame` to render, or any image source
                `luxonis_ml.vizlab.io.load_rgba` understands.

        Returns:
            This writer, so appends can be chained.

        Raises:
            RuntimeError: If the clip has already been closed.

        """
        if self._closed:
            raise RuntimeError(f"clip {self.path.name!r} is already closed")
        rgba = (
            frame.render()
            if isinstance(frame, _Renderish)
            else io.load_rgba(frame)
        )
        if self._size is None:
            self._size = self._canvas((rgba.shape[1], rgba.shape[0]))
        rgba = _fit(rgba, self._size, self._background)
        if self._animated:
            self._buffer.append(io.to_pil(rgba))
        else:
            self._write_video_frame(rgba)
        self._count += 1
        return self

    def extend(self, frames: "Iterable[_Renderish | ImageSource]") -> Self:
        """Append every frame in ``frames``.

        Args:
            frames: The frames to append, in order.

        Returns:
            This writer.

        """
        for frame in frames:
            self.add(frame)
        return self

    def close(self, *, quiet: bool = False) -> None:
        """Finalize and close the clip.

        Args:
            quiet: Suppress the "no frames" error. Set when an exception is
                already propagating, so the real failure is not masked by a
                complaint about the empty clip it left behind.

        Raises:
            ValueError: If no frames were ever added (and ``quiet`` is unset).

        """
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        elif self._buffer:
            self._write_animation()
        elif not quiet:
            raise ValueError(
                f"no frames were added to {self.path.name!r}; "
                f"nothing was written"
            )

    def _write_video_frame(self, rgba: np.ndarray) -> None:
        """Hand one frame to the OpenCV writer, opening it on the first call."""
        if self._writer is None:
            self._writer = self._open_video()
        self._writer.write(io.export(_flatten(rgba, self._background), "bgr"))

    def _open_video(self) -> "cv2.VideoWriter":
        """Open the OpenCV writer, trying each candidate codec in turn.

        Returns:
            The opened ``cv2.VideoWriter``.

        Raises:
            RuntimeError: If no candidate codec could be opened, quoting
                whatever FFmpeg said about it.

        """
        import cv2

        assert self._size is not None
        candidates = (
            (self._requested_codec,)
            if self._requested_codec
            else _VIDEO_CODECS[self._suffix]
        )
        chatter: list[str] = []
        for fourcc in candidates:
            with _muted_stderr(chatter):
                writer = cv2.VideoWriter(
                    str(self.path),
                    cv2.VideoWriter.fourcc(*fourcc),
                    self._fps,
                    self._size,
                )
                opened = writer.isOpened()
            if opened:
                self.codec = fourcc
                return writer
            writer.release()
        detail = "".join(chatter).strip()
        raise RuntimeError(
            f"no usable encoder for {self._suffix!r}: tried "
            f"{', '.join(candidates)}. This OpenCV build may lack the codec — "
            f"'.mp4' with the 'mp4v' codec and '.avi' with 'MJPG' are the most "
            f"widely available fallbacks."
            + (f"\nFFmpeg reported:\n{detail}" if detail else "")
        )

    def _write_animation(self) -> None:
        """Encode the buffered frames as a single animated image."""
        first, *rest = self._buffer
        options: dict[str, object] = {
            "save_all": True,
            "append_images": rest,
            "duration": max(1, round(1000.0 / self._fps)),
            "loop": self._loop,
        }
        if self._suffix in (".webp", ".avif"):
            options["quality"] = self._quality
        self.codec = _ANIMATION_FORMATS[self._suffix]
        first.save(self.path, format=self.codec, **options)
        self._buffer.clear()


def save_video(
    frames: "Iterable[_Renderish | ImageSource]",
    path: "str | Path",
    **kwargs: object,
) -> Path:
    """Write a whole sequence of frames to one clip.

    The one-shot form of `VideoWriter` for when the frames are already in hand.

    Args:
        frames: The frames to encode, in order.
        path: Destination file; its extension picks the format.
        **kwargs: Passed through to `VideoWriter`.

    Returns:
        The path written to.

    Raises:
        ValueError: If ``frames`` is empty.

    Examples:
        >>> import numpy as np
        >>> from luxonis_ml.vizlab import save_video
        >>> save_video(
        ...     [np.full((16, 16, 3), i * 40, np.uint8) for i in range(4)],
        ...     "ramp.gif",
        ...     fps=4,
        ... ).name
        'ramp.gif'

    """
    with VideoWriter(path, **kwargs) as clip:  # pyright: ignore[reportArgumentType]
        clip.extend(frames)
    return Path(path)
