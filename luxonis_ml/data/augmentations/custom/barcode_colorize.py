from typing import Any

import albumentations as A
import cv2
import numpy as np

from luxonis_ml.data.utils.visualizations import resolve_color
from luxonis_ml.typing import Color


class BarcodeColorize(A.ImageOnlyTransform):
    def __init__(
        self,
        target_color: Color = (128, 0, 255),
        darkness_threshold: float = 0.75,
        softness: float = 0.15,
        chroma_threshold: float = 0.08,
        local_window_size: int = 31,
        local_contrast_threshold: float = 0.08,
        p: float = 0.5,
    ):
        """Colorizes grayscale-ish dark pixels while preserving light
        background pixels.

        This is intended for barcodes, text, and similar black-on-white
        graphics where `HueSaturationValue` is ineffective because the
        foreground pixels have near-zero saturation.

        @param target_color: RGB color to map dark barcode strokes to.
        @param darkness_threshold: Maximum normalized grayscale value
            still considered part of the dark foreground.
        @param softness: Width of the transition band around
            `darkness_threshold`.
        @param chroma_threshold: Maximum per-pixel chroma (max channel -
            min channel) for a pixel to still be treated as grayscale.
        @param local_window_size: Window size used to estimate the local
            background intensity around a candidate pixel.
        @param local_contrast_threshold: Minimum normalized amount by
            which a matched pixel must be darker than its local
            neighborhood.
        @param p: Probability of applying the augmentation.
        """
        super().__init__(p=p)

        if not 0.0 <= darkness_threshold <= 1.0:
            raise ValueError(
                "darkness_threshold must be in the range [0, 1]."
            )
        if softness < 0.0:
            raise ValueError("softness must be non-negative.")
        if not 0.0 <= chroma_threshold <= 1.0:
            raise ValueError("chroma_threshold must be in the range [0, 1].")
        if local_window_size < 1 or local_window_size % 2 == 0:
            raise ValueError(
                "local_window_size must be a positive odd integer."
            )
        if not 0.0 <= local_contrast_threshold <= 1.0:
            raise ValueError(
                "local_contrast_threshold must be in the range [0, 1]."
            )

        self.target_color = self._normalize_color(target_color)
        self.darkness_threshold = darkness_threshold
        self.softness = softness
        self.chroma_threshold = chroma_threshold
        self.local_window_size = local_window_size
        self.local_contrast_threshold = local_contrast_threshold

    @staticmethod
    def _normalize_color(color: Color) -> np.ndarray:
        resolved = np.asarray(resolve_color(color), dtype=np.float32)
        if resolved.shape != (3,):
            raise ValueError(
                "target_color must resolve to an RGB triplet."
            )
        if resolved.max() > 1.0:
            resolved /= 255.0
        return np.clip(resolved, 0.0, 1.0)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] != 3:
            raise TypeError(
                "BarcodeColorize expects an RGB image with shape (H, W, 3)."
            )

        if np.issubdtype(img.dtype, np.integer):
            work = img.astype(np.float32) / 255.0
            restore = lambda arr: np.clip(np.round(arr * 255.0), 0, 255).astype(  # noqa: E731
                img.dtype
            )
        elif np.issubdtype(img.dtype, np.floating):
            work = np.clip(img.astype(np.float32), 0.0, 1.0)
            restore = lambda arr: np.clip(arr, 0.0, 1.0).astype(img.dtype)  # noqa: E731
        else:
            raise TypeError(
                "BarcodeColorize supports integer and float image dtypes."
            )

        gray = work.mean(axis=2)
        chroma = work.max(axis=2) - work.min(axis=2)
        local_mean = cv2.blur(
            gray,
            (self.local_window_size, self.local_window_size),
            borderType=cv2.BORDER_REPLICATE,
        )

        if self.softness == 0.0:
            mask = (gray <= self.darkness_threshold).astype(np.float32)
        else:
            lower = max(0.0, self.darkness_threshold - self.softness)
            upper = min(1.0, self.darkness_threshold + self.softness)
            if upper == lower:
                mask = (gray <= self.darkness_threshold).astype(np.float32)
            else:
                mask = np.clip((upper - gray) / (upper - lower), 0.0, 1.0)

        grayscale_mask = (chroma <= self.chroma_threshold).astype(np.float32)
        contrast_mask = (
            (local_mean - gray) >= self.local_contrast_threshold
        ).astype(np.float32)
        mask *= grayscale_mask
        mask *= contrast_mask

        darkness = 1.0 - gray
        colorized = 1.0 - darkness[..., None] * (1.0 - self.target_color)
        out = work * (1.0 - mask[..., None]) + colorized * mask[..., None]
        return restore(out)
