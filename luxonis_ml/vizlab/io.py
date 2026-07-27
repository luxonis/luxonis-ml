"""Image loading and saving.

The goal is that users pass whatever they already have — an OpenCV ``BGR`` array,
a float tensor, a PIL image, or a path — and the library figures it out. Loading
normalizes everything to a contiguous ``(H, W, 4)`` ``uint8`` RGBA array; saving
and export go back out in whatever layout the caller asks for.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import numpy as np
import skia

if TYPE_CHECKING:
    from PIL import Image as PILImage


class _TensorLike(Protocol):
    """Duck-typed torch tensor: enough to pull it into a NumPy array.

    ``torch`` is not a dependency of this package, so tensors are accepted
    structurally rather than by importing ``torch.Tensor``.
    """

    def detach(self) -> "_TensorLike": ...
    def cpu(self) -> "_TensorLike": ...
    def numpy(self) -> np.ndarray: ...


ImageSource: TypeAlias = (
    "np.ndarray | PILImage.Image | _TensorLike | str | Path"
)
"""Any image the library can load: a NumPy array, a Pillow image, a (duck-typed)
torch tensor, or a filesystem path. See `load_rgba`."""


def _as_uint8(array: np.ndarray) -> np.ndarray:
    """Convert an image array to ``uint8``, scaling floats in ``[0, 1]``.

    Args:
        array: Image array of any numeric dtype.

    Returns:
        A ``uint8`` copy. Float arrays whose max value is ``<= 1`` are scaled by 255.

    """
    if array.dtype == np.uint8:
        return array
    if array.dtype.kind == "f":
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        return np.clip(array * scale, 0, 255).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def _ndarray_to_rgba(array: np.ndarray, mode: str) -> np.ndarray:
    """Normalize a numpy image to RGBA.

    Args:
        array: An ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)``, or
            ``(H, W, 4)`` array.
        mode: Channel order of 3/4-channel input, ``"rgb"`` or ``"bgr"``.

    Returns:
        A contiguous ``(H, W, 4)`` ``uint8`` RGBA array.

    Raises:
        ValueError: If the array shape or mode is unsupported.

    """
    arr = _as_uint8(array)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape {array.shape!r}")
    if mode not in ("rgb", "bgr"):
        raise ValueError(f"mode must be 'rgb' or 'bgr', got {mode!r}")
    if mode == "bgr":
        arr = arr[:, :, ::-1] if arr.shape[2] == 3 else arr[:, :, [2, 1, 0, 3]]
    if arr.shape[2] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    return np.ascontiguousarray(arr)


def _tensor_to_rgba(tensor: "_TensorLike", mode: str) -> np.ndarray:
    """Normalize a torch tensor to RGBA.

    Handles both ``(C, H, W)`` and ``(H, W, C)`` layouts.

    Args:
        tensor: A torch tensor (duck-typed; ``torch`` is not a hard dependency).
        mode: Channel order, ``"rgb"`` or ``"bgr"``.

    Returns:
        A contiguous ``(H, W, 4)`` ``uint8`` RGBA array.

    """
    numpy_array = (
        tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
    )
    array = np.asarray(numpy_array)  # type: ignore[arg-type]
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[2] not in (1, 3, 4)
    ):
        array = np.transpose(array, (1, 2, 0))
    return _ndarray_to_rgba(array, mode)


def load_rgba(source: "ImageSource", mode: str = "rgb") -> np.ndarray:
    """Load any supported image source into an RGBA array.

    Integer inputs are clipped to ``[0, 255]``. Floating-point inputs whose
    maximum is at most ``1`` are interpreted as normalized and scaled by 255;
    other floating-point inputs are treated as byte-range values.

    Args:
        source: A numpy array, a PIL image, a torch tensor, or a filesystem path
            (``str`` or ``pathlib.Path``).
        mode: Channel order for array/tensor sources, ``"rgb"`` (default) or
            ``"bgr"`` (e.g. ``cv2.imread`` output). Ignored for PIL images and
            paths, which carry their own color information.

    Returns:
        A contiguous ``(H, W, 4)`` ``uint8`` RGBA array.

    Raises:
        TypeError: If the source type is not supported.
        ValueError: If an array/tensor shape or channel mode is unsupported.
        FileNotFoundError: If a path cannot be read or decoded.

    """
    if isinstance(source, np.ndarray):
        return _ndarray_to_rgba(source, mode)
    if isinstance(source, (str, Path)):
        return _load_path(source)
    module = type(source).__module__
    if module.startswith("PIL."):
        pil = cast("PILImage.Image", source)
        return _ndarray_to_rgba(np.asarray(pil.convert("RGBA")), "rgb")
    if module.startswith("torch"):
        return _tensor_to_rgba(cast("_TensorLike", source), mode)
    raise TypeError(f"unsupported image source type: {type(source)!r}")


def _load_path(path: str | Path) -> np.ndarray:
    """Decode an image file into RGBA using Skia's codecs.

    Args:
        path: Path to an image file.

    Returns:
        A contiguous ``(H, W, 4)`` ``uint8`` RGBA array.

    Raises:
        FileNotFoundError: If the file cannot be read or decoded.

    """
    data = skia.Data.MakeFromFileName(str(path))
    if data is None:
        raise FileNotFoundError(f"could not read image file: {path}")
    image = skia.Image.MakeFromEncoded(data)
    if image is None:
        raise FileNotFoundError(f"could not decode image file: {path}")
    return np.ascontiguousarray(
        image.toarray(colorType=skia.ColorType.kRGBA_8888_ColorType)
    )


def export(rgba: np.ndarray, mode: str = "rgb") -> np.ndarray:
    """Convert an internal RGBA array to a caller-requested layout.

    Args:
        rgba: An ``(H, W, 4)`` RGBA array.
        mode: Output layout: ``"rgb"``, ``"bgr"``, ``"rgba"``, or ``"bgra"``.

    Returns:
        The requested channel layout: three channels for ``rgb``/``bgr`` and
        four for ``rgba``/``bgra``. ``"rgba"`` returns the input array itself;
        the other modes return contiguous arrays.

    Raises:
        ValueError: If ``mode`` is not recognized.

    """
    if mode == "rgba":
        return rgba
    if mode == "bgra":
        return np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])
    if mode == "rgb":
        return np.ascontiguousarray(rgba[:, :, :3])
    if mode == "bgr":
        return np.ascontiguousarray(rgba[:, :, [2, 1, 0]])
    raise ValueError(f"unknown export mode {mode!r}")


def to_pil(rgba: np.ndarray) -> "PILImage.Image":
    """Convert an internal RGBA array to a PIL image.

    Args:
        rgba: An ``(H, W, 4)`` RGBA array.

    Returns:
        A PIL ``Image`` in RGBA mode.

    Raises:
        ImportError: If Pillow is not installed.

    """
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for to_pil(); install the 'pil' extra"
        ) from exc
    return PILImage.fromarray(rgba, mode="RGBA")


def save(rgba: np.ndarray, path: str | Path, *, quality: int = 95) -> None:
    """Encode and write an RGBA array to a file.

    The format is inferred from the file extension (``.png``, ``.jpg``/``.jpeg``,
    ``.webp``).

    Args:
        rgba: An ``(H, W, 4)`` RGBA array.
        path: Destination path. Its suffix selects PNG, JPEG, or WebP.
        quality: Encoder quality for JPEG and WebP, from 0 to 100. PNG ignores
            this setting.

    Raises:
        ValueError: If the file extension is unsupported.

    """
    suffix = Path(path).suffix.lower()
    formats = {
        ".png": skia.EncodedImageFormat.kPNG,
        ".jpg": skia.EncodedImageFormat.kJPEG,
        ".jpeg": skia.EncodedImageFormat.kJPEG,
        ".webp": skia.EncodedImageFormat.kWEBP,
    }
    if suffix not in formats:
        raise ValueError(f"unsupported output format {suffix!r}")
    image = skia.Image.fromarray(
        np.ascontiguousarray(rgba),
        colorType=skia.ColorType.kRGBA_8888_ColorType,
    )
    image.save(str(path), formats[suffix], quality)
