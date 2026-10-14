"""Coverage for the IO layer: loading, exporting, and saving."""

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from luxonis_ml.vizlab import io


class _FakeTensor:
    """A duck-typed stand-in for a torch tensor (no torch dependency)."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


# Make load_rgba's module-name dispatch treat instances as torch tensors.
_FakeTensor.__module__ = "torch"


def test_as_uint8_scales_floats_and_clips_ints() -> None:
    unit = io.load_rgba(
        np.ones((2, 2, 3), dtype=np.float32)
    )  # max <= 1 -> *255
    assert unit[0, 0, 0] == 255
    big = io.load_rgba(
        np.full((2, 2, 3), 300.0, dtype=np.float32)
    )  # >1 -> clip
    assert big[0, 0, 0] == 255
    ints = io.load_rgba(np.full((2, 2, 3), 500, dtype=np.int16))  # clip to 255
    assert ints[0, 0, 0] == 255


def test_grayscale_is_broadcast_to_rgb() -> None:
    gray = io.load_rgba(np.full((3, 4), 128, dtype=np.uint8))
    assert gray.shape == (3, 4, 4)
    assert gray[0, 0, 0] == gray[0, 0, 1] == gray[0, 0, 2] == 128


def test_bad_shape_and_mode_raise() -> None:
    with pytest.raises(ValueError, match="unsupported image shape"):
        io.load_rgba(np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be"):
        io.load_rgba(np.zeros((2, 2, 3), dtype=np.uint8), "xyz")


def test_bgr_swaps_channels_for_3_and_4() -> None:
    rgb3 = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb3[0, 0] = (10, 20, 30)
    out3 = io.load_rgba(rgb3, "bgr")
    assert tuple(out3[0, 0, :3]) == (30, 20, 10)

    rgba4 = np.zeros((1, 1, 4), dtype=np.uint8)
    rgba4[0, 0] = (10, 20, 30, 40)
    out4 = io.load_rgba(rgba4, "bgr")
    assert tuple(out4[0, 0]) == (30, 20, 10, 40)


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 1), dtype=np.uint8),
        np.zeros((3, 4, 3), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
    ],
    ids=["gray", "gray-1ch", "rgb", "rgba"],
)
def test_load_rgba_never_aliases_the_source(source: np.ndarray) -> None:
    # An already-RGBA uint8 array used to be handed straight through, so an
    # `Image` shared the caller's buffer. A render caches without looking at the
    # raster, so a later in-place edit would silently show the old pixels.
    rgba = io.load_rgba(source)
    assert not np.shares_memory(rgba, source)


def test_image_owns_a_snapshot_of_its_source() -> None:
    # `Image.render` caches on annotations and size alone, so a raster that can
    # change underneath it would render stale. The image takes a snapshot
    # instead, and editing the caller's array afterwards is a no-op.
    from luxonis_ml.vizlab import Image

    source = np.zeros((4, 4, 4), dtype=np.uint8)
    source[..., 3] = 255
    image = Image(source)
    source[..., :3] = 200
    assert image.base_rgba()[..., :3].max() == 0


def test_tensor_chw_and_hwc() -> None:
    chw = _FakeTensor(
        np.zeros((3, 4, 5), dtype=np.uint8)
    )  # C,H,W -> transposed
    assert io.load_rgba(chw).shape == (4, 5, 4)
    hwc = _FakeTensor(np.zeros((4, 5, 3), dtype=np.uint8))
    assert io.load_rgba(hwc).shape == (4, 5, 4)


def test_single_channel_tensor_is_broadcast_to_rgb() -> None:
    tensor = _FakeTensor(np.full((1, 4, 5), 128, dtype=np.uint8))
    rgba = io.load_rgba(tensor)

    assert rgba.shape == (4, 5, 4)
    assert np.all(rgba[..., :3] == 128)
    assert np.all(rgba[..., 3] == 255)


def test_pil_source() -> None:
    pil = PILImage.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), "RGB")
    assert io.load_rgba(pil).shape == (3, 4, 4)


def test_unsupported_source_raises() -> None:
    with pytest.raises(TypeError, match="unsupported image source"):
        io.load_rgba(12345)  # type: ignore[arg-type]


def test_load_path_roundtrip_and_errors(tmp_path: Path) -> None:
    rgba = np.zeros((5, 6, 4), dtype=np.uint8)
    rgba[..., :3] = 100
    rgba[..., 3] = 255
    png = tmp_path / "img.png"
    io.save(rgba, png)
    loaded = io.load_rgba(str(png))
    assert loaded.shape == (5, 6, 4)

    with pytest.raises(FileNotFoundError, match="could not read"):
        io.load_rgba(str(tmp_path / "missing.png"))

    bogus = tmp_path / "bogus.png"
    bogus.write_text("not an image")
    with pytest.raises(FileNotFoundError, match="could not decode"):
        io.load_rgba(str(bogus))


def test_export_all_modes_and_error() -> None:
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[0, 0] = (1, 2, 3, 4)
    assert io.export(rgba, "rgba").shape == (2, 3, 4)
    assert io.export(rgba, "rgb").shape == (2, 3, 3)
    assert tuple(io.export(rgba, "bgr")[0, 0]) == (3, 2, 1)
    assert tuple(io.export(rgba, "bgra")[0, 0]) == (3, 2, 1, 4)
    with pytest.raises(ValueError, match="unknown export mode"):
        io.export(rgba, "nope")


def test_to_pil_and_save_formats(tmp_path: Path) -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    assert io.to_pil(rgba).mode == "RGBA"
    for name in ("a.png", "b.jpg", "c.jpeg", "d.webp"):
        io.save(rgba, tmp_path / name)
        assert (tmp_path / name).exists()
    with pytest.raises(ValueError, match="unsupported output format"):
        io.save(rgba, tmp_path / "e.gif")


def test_repr_png_renders_scenes_and_composites() -> None:
    """A scene reprs as a PNG of itself, so notebooks display it inline."""
    from luxonis_ml.vizlab import BBox, Image, grid

    scene = Image(np.zeros((16, 24, 3), dtype=np.uint8)).add(
        BBox(x=0.1, y=0.1, w=0.5, h=0.5, label="car")
    )
    for renderable in (scene, grid([scene, scene], ncols=2)):
        png = renderable._repr_png_()
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        with PILImage.open(BytesIO(png)) as decoded:
            assert decoded.size == (renderable.width, renderable.height)


def test_to_pil_without_pillow_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """to_pil raises a helpful error when Pillow is absent."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(ImportError, match="Pillow is required"):
        io.to_pil(np.zeros((2, 2, 4), dtype=np.uint8))
