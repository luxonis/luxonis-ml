import random
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override

from luxonis_ml.data.augmentations.batch_transform import BatchTransform
from luxonis_ml.data.augmentations.custom import LetterboxResize


class MixUp(BatchTransform):
    r"""Batch-based augmentation that blends two images together.

    Blending is performed by a convex combination of the two images
    based on a mixing coefficient :math:`\alpha` sampled from a specified
    distribution. The resulting image is computed as:

        .. math:: \tilde{x} = \alpha x_i + \left(1 - \alpha\right) x_j

    If the images have different sizes, the second image is resized to
    match the first one.

    See:
        `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.
    """

    def __init__(
        self,
        alpha: float | tuple[float, float] = 0.5,
        keep_aspect_ratio: bool = True,
        p: float = 0.5,
    ):
        r"""Create a MixUp augmentation.

        Args:
            alpha: Mixing coefficient or range to uniformly sample from.
            keep_aspect_ratio: Whether to preserve the second image's
                aspect ratio when resizing.
            p: Probability of applying the transform.

        """
        super().__init__(batch_size=2, p=p)

        self._alpha = (
            alpha if isinstance(alpha, list | tuple) else (alpha, alpha)
        )
        if keep_aspect_ratio:
            self._resize_transform = LetterboxResize(1, 1)
        else:
            self._resize_transform = A.Resize(1, 1)

        if not 0 <= self._alpha[0] <= 1 or not 0 <= self._alpha[1] <= 1:
            raise ValueError("Alpha must be in range [0, 1].")

        if self._alpha[0] > self._alpha[1]:
            raise ValueError("Alpha range must be in ascending order.")

    @override
    def apply(
        self,
        image_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        alpha: float,
        **_,
    ) -> np.ndarray:
        r"""Apply MixUp to a batch of images.

        Args:
            image_batch: Images to transform. Each image should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            image_shapes: Shapes of the original images.
            alpha: Mixing coefficient.

        Returns:
            A single image of shape :math:`\left(H_{out}, W_{out}, C\right)` or
            :math:`\left(H_{out}, W_{out}\right)` resulting from blending
            the input images.

        """
        image1 = image_batch[0]
        image2 = self._resize(image_batch[1], image_shapes, "image", alpha)

        mixup_img = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0.0)
        if mixup_img.ndim == 2:
            mixup_img = mixup_img[..., None]
        return mixup_img

    @override
    def apply_to_mask(
        self,
        masks_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        alpha: float,
        **_,
    ) -> np.ndarray:
        r"""Apply MixUp to a batch of semantic segmentation masks.

        Blends masks together. In case of a conflict, the class from the
        mask associated with the higher :math:`\alpha` is chosen.

        Args:
            masks_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            image_shapes: Shapes of the original images.
            alpha: Mixing coefficient.

        Returns:
            A single segmentation mask of shape
            :math:`\left(H_{out}, W_{out}, C\right)` or
            :math:`\left(H_{out}, W_{out}\right)`.

        """
        mask1, mask2 = masks_batch
        if mask2.size > 0:
            mask2 = self._resize(mask2, image_shapes, "mask")
            if mask2.ndim == 2:
                mask2 = mask2[..., None]

        if mask1.size == 0:
            return mask2
        if mask2.size == 0:
            return mask1

        if alpha >= 0.5:
            mask1[(mask1 == 0) & (mask2 != 0)] = mask2[
                (mask1 == 0) & (mask2 != 0)
            ]
            return mask1

        mask2[(mask2 == 0) & (mask1 != 0)] = mask1[(mask2 == 0) & (mask1 != 0)]
        return mask2

    @override
    def apply_to_instance_mask(
        self,
        masks_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        **_,
    ) -> np.ndarray:
        r"""Apply MixUp to a batch of instance segmentation masks.

        Args:
            masks_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, N\right)`, where :math:`N`
                is the number of instances.
            image_shapes: Shapes of the original images.

        Returns:
            A single instance masks of shape
            :math:`\left(H_{out}, W_{out}, N\right)`.

        """
        mask1, mask2 = masks_batch
        if mask2.size > 0:
            mask2 = self._resize(mask2, image_shapes, "mask")
            if mask2.ndim == 2:
                mask2 = mask2[..., None]
            masks_batch[1] = mask2
        if mask1.size == 0:
            return mask2
        if mask2.size == 0:
            return mask1

        return np.concatenate(masks_batch, axis=-1)

    @override
    def apply_to_bboxes(
        self,
        bboxes_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        **_,
    ) -> np.ndarray:
        """Apply MixUp to a batch of bounding boxes.

        Args:
            bboxes_batch: Bounding boxes to transform.
            image_shapes: Original image shapes.

        Returns:
            Transformed bounding boxes.

        """
        for i in range(len(bboxes_batch)):
            bbox = bboxes_batch[i]
            if bbox.size == 0:  # pragma: no cover
                bboxes_batch[i] = np.zeros((0, 6), dtype=bbox.dtype)

        bboxes_batch[1] = self._resize(
            bboxes_batch[1],
            image_shapes,
            "bboxes",
            orig_height=image_shapes[0][0],
            orig_width=image_shapes[0][1],
        )

        return np.concatenate(bboxes_batch, axis=0)

    @override
    def apply_to_keypoints(
        self,
        keypoints_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        **_,
    ) -> np.ndarray:
        """Apply MixUp to a batch of keypoints.

        Args:
            keypoints_batch: Keypoints to transform.
            image_shapes: Original image shapes.

        Returns:
            Transformed keypoints.

        """
        for i in range(len(keypoints_batch)):
            if keypoints_batch[i].size == 0:  # pragma: no cover
                keypoints_batch[i] = np.zeros(
                    (0, 5), dtype=keypoints_batch[i].dtype
                )

        keypoints_batch[1] = self._resize(
            keypoints_batch[1],
            image_shapes,
            "keypoints",
            shape=image_shapes[1],
            orig_height=image_shapes[1][0],
            orig_width=image_shapes[1][1],
        )
        return np.concatenate(keypoints_batch, axis=0)

    @override
    def get_params(self) -> dict[str, Any]:
        """Sample a mixing coefficient from the specified distribution.

        Returns:
            Dictionary containing ``"alpha"``
            key with the sampled mixing coefficient.

        """
        alpha = random.uniform(*self._alpha)
        return {"alpha": alpha}

    def _resize(
        self,
        data: np.ndarray,
        shapes: list[tuple[int, int]],
        target_type: Literal["image", "mask", "bboxes", "keypoints"],
        alpha: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        out_height, out_width = shapes[0]
        orig_height, orig_width = shapes[1]
        self._resize_transform.height = out_height
        self._resize_transform.width = out_width
        padding = []
        if isinstance(self._resize_transform, LetterboxResize):
            if alpha is not None:
                self._resize_transform._image_fill_value = (
                    int(255 * alpha),
                    int(255 * alpha),
                    int(255 * alpha),
                )

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
