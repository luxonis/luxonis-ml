import math
from typing import Any, Literal

import albumentations as A
import numpy as np
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from typing_extensions import override

from luxonis_ml.data.augmentations.batch_transform import BatchTransform
from luxonis_ml.data.augmentations.custom import LetterboxResize


class CutMix(BatchTransform):
    r"""Batch-based augmentation that patches one image into another.

    CutMix samples a rectangular region from the second image and pastes it
    into the first image. The rectangle area is derived from a mixing
    coefficient sampled from :math:`Beta(\alpha, \alpha)`.

    If the images have different sizes, the second image and its labels are
    resized to match the first image before the patch is copied.

    Bounding boxes from the first image are kept if their visible area
    (original area minus the intersection with the patch) is at least
    ``bbox_min_visibility`` times their original area. Bounding boxes from
    the second image are clipped to the patch region. Keypoints from the
    first image that fall inside the patch are marked invisible, while
    keypoints from the second image outside the patch are marked invisible.

    See:
        `CutMix: Regularization Strategy to Train Strong Classifiers with
        Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        keep_aspect_ratio: bool = True,
        bbox_min_visibility: float = 0.5,
        p: float = 0.5,
    ):
        r"""Create a CutMix augmentation.

        Args:
            alpha: Positive parameter for the
                :math:`Beta(\alpha, \alpha)` distribution.
            keep_aspect_ratio: Whether to preserve the second image's
                aspect ratio when resizing.
            bbox_min_visibility: Minimum fraction of a first-image
                bounding box that must remain visible (outside the patch)
                for the box to be kept. Must lie in :math:`[0, 1]`.
            p: Probability of applying the transform.

        Raises:
            ValueError: If ``alpha`` is not positive or
                ``bbox_min_visibility`` is outside :math:`[0, 1]`.

        """
        super().__init__(batch_size=2, p=p)

        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0.")

        if not 0.0 <= bbox_min_visibility <= 1.0:
            raise ValueError(
                "bbox_min_visibility must be in range [0, 1]."
            )

        self._alpha = alpha
        self._bbox_min_visibility = bbox_min_visibility
        if keep_aspect_ratio:
            self._resize_transform = LetterboxResize(1, 1)
        else:
            self._resize_transform = A.Resize(1, 1)

    @override
    def get_params(self) -> dict[str, Any]:
        """Sample the CutMix mixing coefficient."""
        lambda_value = float(np.random.beta(self._alpha, self._alpha))
        return {"lambda_value": lambda_value}

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Sample a CutMix rectangle from the first image shape."""
        image = data["image"][0]
        height, width = image.shape[:2]
        center_x = int(np.random.randint(0, width))
        center_y = int(np.random.randint(0, height))
        x1, y1, x2, y2 = self._compute_patch(
            params["lambda_value"], height, width, center_x, center_y
        )
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @override
    def apply(
        self,
        image_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        **_,
    ) -> np.ndarray:
        r"""Apply CutMix to a batch of images.

        Args:
            image_batch: Images to transform. Each image should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            image_shapes: Shapes of the original images.
            x1: Left edge of the patch in the first image.
            y1: Top edge of the patch in the first image.
            x2: Right edge of the patch in the first image.
            y2: Bottom edge of the patch in the first image.

        Returns:
            A single image of shape :math:`\left(H_{out}, W_{out}, C\right)`
            or :math:`\left(H_{out}, W_{out}\right)` where the rectangle
            :math:`\left[y_1:y_2, x_1:x_2\right]` is replaced with the
            corresponding region of the (resized) second image.

        """
        image = image_batch[0].copy()
        if x1 == x2 or y1 == y2:
            return image

        patch_source = self._resize(image_batch[1], image_shapes, "image")
        patch_source = self._match_dimensions(patch_source, image)
        image[y1:y2, x1:x2] = patch_source[y1:y2, x1:x2]
        return image

    @override
    def apply_to_mask(
        self,
        masks_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        **_,
    ) -> np.ndarray:
        r"""Apply CutMix to semantic segmentation masks.

        Args:
            masks_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            image_shapes: Shapes of the original images.
            x1: Left edge of the patch in the first image.
            y1: Top edge of the patch in the first image.
            x2: Right edge of the patch in the first image.
            y2: Bottom edge of the patch in the first image.

        Returns:
            A single semantic segmentation mask of shape
            :math:`\left(H_{out}, W_{out}, C\right)` where the patch region
            comes from the second (resized) mask and the rest comes from
            the first mask.

        """
        mask1, mask2 = masks_batch

        if mask2.size > 0:
            mask2 = self._resize(mask2, image_shapes, "mask")
            if mask2.ndim == 2:
                mask2 = mask2[..., None]

        if mask1.size == 0 and mask2.size == 0:
            return mask1
        if x1 == x2 or y1 == y2:
            return mask1

        if mask1.size == 0:
            mask = np.zeros_like(mask2)
        else:
            mask = mask1.copy()
            if mask2.size == 0:
                mask2 = np.zeros_like(mask)
            else:
                mask2 = self._match_dimensions(mask2, mask)

        mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        return mask

    @override
    def apply_to_instance_mask(
        self,
        masks_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        **_,
    ) -> np.ndarray:
        r"""Apply CutMix to instance segmentation masks.

        Args:
            masks_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, N\right)`, where :math:`N` is the number
                of instances.
            image_shapes: Shapes of the original images.
            x1: Left edge of the patch in the first image.
            y1: Top edge of the patch in the first image.
            x2: Right edge of the patch in the first image.
            y2: Bottom edge of the patch in the first image.

        Returns:
            Concatenated instance masks of shape
            :math:`\left(H_{out}, W_{out}, N_1 + N_2\right)`. First-image
            instances have their patch region cleared; second-image
            instances have everything outside the patch cleared.

        """
        out_height, out_width = image_shapes[0]
        mask1, mask2 = masks_batch
        dtype = mask1.dtype if mask1.size > 0 else mask2.dtype

        if mask1.size == 0:
            mask1_out = np.zeros((out_height, out_width, 0), dtype=dtype)
        else:
            mask1_out = mask1.copy()
            if mask1_out.ndim == 2:
                mask1_out = mask1_out[..., None]
            mask1_out[y1:y2, x1:x2] = 0

        if mask2.size == 0 or x1 == x2 or y1 == y2:
            return mask1_out

        mask2 = self._resize(mask2, image_shapes, "mask")
        if mask2.ndim == 2:
            mask2 = mask2[..., None]

        mask2_out = np.zeros_like(mask2)
        mask2_out[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]
        return np.concatenate([mask1_out, mask2_out], axis=-1)

    @override
    def apply_to_bboxes(
        self,
        bboxes_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        **_,
    ) -> np.ndarray:
        """Apply CutMix to bounding boxes.

        Bounding boxes from the first image are kept when their visible
        area (original area minus intersection with the patch) is at least
        ``bbox_min_visibility`` of the original area. Bounding boxes from
        the second image are clipped to the patch region.

        Args:
            bboxes_batch: Bounding boxes to transform, in normalized
                Albumentations format.
            image_shapes: Original image shapes.
            x1: Left edge of the patch in the first image.
            y1: Top edge of the patch in the first image.
            x2: Right edge of the patch in the first image.
            y2: Bottom edge of the patch in the first image.

        Returns:
            Concatenated bounding boxes from both images.

        """
        bbox1, bbox2 = bboxes_batch
        if bbox1.size == 0:
            bbox1 = self._empty_rows(bbox1, 6)
        else:
            bbox1 = self._filter_bboxes_by_visibility(
                bbox1,
                image_shapes[0][0],
                image_shapes[0][1],
                x1,
                y1,
                x2,
                y2,
                self._bbox_min_visibility,
            )

        if bbox2.size == 0:
            bbox2 = self._empty_rows(bbox2, bbox1.shape[1])
        else:
            bbox2 = self._resize(
                bbox2.copy(),
                image_shapes,
                "bboxes",
                orig_height=image_shapes[1][0],
                orig_width=image_shapes[1][1],
            )
            bbox2 = self._clip_bboxes_to_patch(
                bbox2, image_shapes[0][0], image_shapes[0][1], x1, y1, x2, y2
            )

        return np.concatenate([bbox1, bbox2], axis=0)

    @override
    def apply_to_keypoints(
        self,
        keypoints_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        **_,
    ) -> np.ndarray:
        """Apply CutMix to keypoints.

        Keypoints from the first image that fall inside the patch are
        marked invisible; keypoints from the second image outside the
        patch are marked invisible.

        Args:
            keypoints_batch: Keypoints to transform.
            image_shapes: Original image shapes.
            x1: Left edge of the patch in the first image.
            y1: Top edge of the patch in the first image.
            x2: Right edge of the patch in the first image.
            y2: Bottom edge of the patch in the first image.

        Returns:
            Concatenated keypoints from both images with visibility flags
            updated according to their position relative to the patch.

        """
        keypoints1, keypoints2 = keypoints_batch

        if keypoints1.size == 0:
            keypoints1 = self._empty_rows(keypoints1, 5)
        else:
            keypoints1 = self._mark_keypoints_in_patch(
                keypoints1.copy(), x1, y1, x2, y2, visible_inside=False
            )

        if keypoints2.size == 0:
            keypoints2 = self._empty_rows(keypoints2, keypoints1.shape[1])
        else:
            keypoints2 = self._resize(
                keypoints2.copy(),
                image_shapes,
                "keypoints",
                shape=image_shapes[1],
                orig_height=image_shapes[1][0],
                orig_width=image_shapes[1][1],
            )
            keypoints2 = self._mark_keypoints_in_patch(
                keypoints2.copy(), x1, y1, x2, y2, visible_inside=True
            )

        return np.concatenate([keypoints1, keypoints2], axis=0)

    def _resize(
        self,
        data: np.ndarray,
        shapes: list[tuple[int, int]],
        target_type: Literal["image", "mask", "bboxes", "keypoints"],
        **kwargs,
    ) -> np.ndarray:
        out_height, out_width = shapes[0]
        orig_height, orig_width = shapes[1]
        self._resize_transform.height = out_height
        self._resize_transform.width = out_width
        padding = []
        if isinstance(self._resize_transform, LetterboxResize):
            padding = LetterboxResize.compute_padding(
                orig_height, orig_width, out_height, out_width
            )

        if target_type == "image":
            return self._resize_transform.apply(data, *padding, **kwargs)
        if target_type == "mask":
            return self._resize_transform.apply_to_mask(
                data, *padding, **kwargs
            )
        if target_type == "bboxes":
            return self._resize_transform.apply_to_bboxes(
                data, *padding, **kwargs
            )
        if target_type == "keypoints":
            return self._resize_transform.apply_to_keypoints(
                data, *padding, **kwargs
            )
        raise ValueError(f"Unsupported target type: {target_type}")

    @staticmethod
    def _compute_patch(
        lambda_value: float,
        height: int,
        width: int,
        center_x: int,
        center_y: int,
    ) -> tuple[int, int, int, int]:
        lambda_value = float(np.clip(lambda_value, 0.0, 1.0))
        cut_ratio = math.sqrt(1.0 - lambda_value)
        cut_width = int(width * cut_ratio)
        cut_height = int(height * cut_ratio)

        x1 = int(np.clip(center_x - cut_width // 2, 0, width))
        y1 = int(np.clip(center_y - cut_height // 2, 0, height))
        x2 = int(np.clip(center_x + cut_width // 2, 0, width))
        y2 = int(np.clip(center_y + cut_height // 2, 0, height))
        return x1, y1, x2, y2

    @staticmethod
    def _clip_bboxes_to_patch(
        bboxes: np.ndarray,
        height: int,
        width: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        if bboxes.size == 0 or x1 == x2 or y1 == y2:
            return CutMix._empty_rows(bboxes, 6)

        clipped = denormalize_bboxes(bboxes.copy(), (height, width))
        clipped[:, 0] = np.clip(clipped[:, 0], x1, x2)
        clipped[:, 2] = np.clip(clipped[:, 2], x1, x2)
        clipped[:, 1] = np.clip(clipped[:, 1], y1, y2)
        clipped[:, 3] = np.clip(clipped[:, 3], y1, y2)

        valid = (clipped[:, 2] > clipped[:, 0]) & (
            clipped[:, 3] > clipped[:, 1]
        )
        if not np.any(valid):
            return CutMix._empty_rows(bboxes, bboxes.shape[1])
        return normalize_bboxes(clipped[valid], (height, width))

    @staticmethod
    def _filter_bboxes_by_visibility(
        bboxes: np.ndarray,
        height: int,
        width: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        min_visibility: float,
    ) -> np.ndarray:
        if bboxes.size == 0 or x1 == x2 or y1 == y2:
            return bboxes

        denormalized = denormalize_bboxes(bboxes.copy(), (height, width))
        bbox_width = denormalized[:, 2] - denormalized[:, 0]
        bbox_height = denormalized[:, 3] - denormalized[:, 1]
        original_area = bbox_width * bbox_height

        overlap_width = np.clip(
            np.minimum(denormalized[:, 2], x2)
            - np.maximum(denormalized[:, 0], x1),
            0,
            None,
        )
        overlap_height = np.clip(
            np.minimum(denormalized[:, 3], y2)
            - np.maximum(denormalized[:, 1], y1),
            0,
            None,
        )
        intersection_area = overlap_width * overlap_height

        with np.errstate(divide="ignore", invalid="ignore"):
            visible_fraction = np.where(
                original_area > 0,
                1.0 - intersection_area / original_area,
                0.0,
            )
        return bboxes[visible_fraction >= min_visibility]

    @staticmethod
    def _mark_keypoints_in_patch(
        keypoints: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        visible_inside: bool,
    ) -> np.ndarray:
        if keypoints.size == 0:
            return keypoints

        inside = (
            (keypoints[:, 0] >= x1)
            & (keypoints[:, 0] < x2)
            & (keypoints[:, 1] >= y1)
            & (keypoints[:, 1] < y2)
        )
        invisible = ~inside if visible_inside else inside
        keypoints[invisible, -1] = 0
        return keypoints

    @staticmethod
    def _match_dimensions(
        data: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        if reference.ndim == 2 and data.ndim == 3 and data.shape[-1] == 1:
            return data[..., 0]
        if reference.ndim == 3 and data.ndim == 2:
            return data[..., None]
        return data

    @staticmethod
    def _empty_rows(array: np.ndarray, fallback_width: int) -> np.ndarray:
        width = array.shape[1] if array.ndim == 2 else fallback_width
        return np.zeros((0, width), dtype=array.dtype)
