import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override

from luxonis_ml.data.augmentations.batch_transform import BatchBasedTransform
from luxonis_ml.data.augmentations.custom import LetterboxResize


class MixUp(BatchBasedTransform):
    def __init__(
        self,
        alpha: Union[float, Tuple[float, float]] = 0.5,
        min_alpha_mask_threshold: float = 0.1,
        max_alpha_mask_threshold: float = 0.9,
        keep_aspect_ratio: bool = True,
        p: float = 0.5,
    ):
        """MixUp augmentation that merges two images and their
        annotations into one. If images are not of same size then second
        one is first resized to match the first one.

        @type alpha: Union[float, Tuple[float, float]]
        @param alpha: Mixing coefficient, either a single float or a
            tuple representing the range. Defaults to C{0.5}.
        @type min_alpha_mask_threshold: float
        @param min_alpha_mask_threshold: Minimum threshold for alpha
            when applying the transform to semantic segmentations. If
            alpha is less than this threshold, the first mask is
            discarded. Defaults to C{0.1}.
        @type max_alpha_mask_threshold: float
        @param max_alpha_mask_threshold: Maximum threshold for alpha
            when applying the transform to semantic segmentations. If
            alpha is greater than this threshold, the second mask is
            discarded. Defaults to C{0.9}.
        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep the aspect ratio of
            the second image when resizing. Defaults to C{True}.
        @type p: float, optional
        @param p: Probability of applying the transform. Defaults to
            C{0.5}.
        """
        super().__init__(batch_size=2, p=p)

        self.min_alpha_mask_threshold = min_alpha_mask_threshold
        self.max_alpha_mask_threshold = max_alpha_mask_threshold

        self.alpha = (
            alpha if isinstance(alpha, (list, tuple)) else (alpha, alpha)
        )
        if keep_aspect_ratio:
            self.resize_transform = LetterboxResize(1, 1)
        else:
            self.resize_transform = A.Resize(1, 1)
        self._check_alpha()

    def _check_alpha(self) -> None:
        if not 0 <= self.alpha[0] <= 1 or not 0 <= self.alpha[1] <= 1:
            raise ValueError("Alpha must be in range [0, 1].")

        if self.alpha[0] > self.alpha[1]:
            raise ValueError("Alpha range must be in ascending order.")

    def resize(
        self,
        data: np.ndarray,
        shapes: List[Tuple[int, int]],
        target_type: Literal["image", "mask", "bboxes", "keypoints"],
        alpha: Optional[float] = None,
        **kwargs,
    ) -> np.ndarray:
        out_height, out_width = shapes[0]
        orig_height, orig_width = shapes[1]
        self.resize_transform.height = out_height
        self.resize_transform.width = out_width
        padding = []
        if isinstance(self.resize_transform, LetterboxResize):
            if alpha is not None:
                self.resize_transform.image_fill_value = (
                    int(255 * alpha),
                    int(255 * alpha),
                    int(255 * alpha),
                )

            padding = LetterboxResize.compute_padding(
                orig_height, orig_width, out_height, out_width
            )

        if target_type == "image":
            return self.resize_transform.apply(data, *padding, **kwargs)
        elif target_type == "mask":
            return self.resize_transform.apply_to_mask(
                data, *padding, **kwargs
            )
        elif target_type == "bboxes":
            return self.resize_transform.apply_to_bboxes(
                data, *padding, **kwargs
            )
        elif target_type == "keypoints":
            return self.resize_transform.apply_to_keypoints(
                data, *padding, **kwargs
            )

    @override
    def apply(
        self,
        image_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        alpha: float,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of images.

        @type image_batch: List[np.ndarray]
        @param image_batch: Batch of input images to which the
            transformation is applied.
        @type image_shapes: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @rtype: List[np.ndarray]
        @return: List of transformed images.
        """

        image1 = image_batch[0]
        image2 = self.resize(image_batch[1], image_shapes, "image", alpha)

        return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0.0)

    @override
    def apply_to_mask(
        self,
        mask_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        alpha: float,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of masks.

        Uses several heuristics to merge the massk:
            - If the alpha value is greater than the C{max_alpha_mask_threshold},
                then only the first mask is chosen.
            - If the alpha value is less than the C{min_alpha_mask_threshold},
                then only the second mask is chosen.
            - In other cases, the masks are merged by choosing the non-zero
                values from both masks. In case of a conflict (both masks have
                a non-zero value at the same position), the value is chosen
                from the mask with the higher alpha value.

        @type mask_batch: List[np.ndarray]
        @param mask_batch: Batch of input masks to which the
            transformation is applied.
        @type image_shapes: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @type alpha: float
        @param alpha: Mixing coefficient.
        @rtype: List[np.ndarray]
        @return: List of transformed masks.
        """
        mask1, mask2 = mask_batch
        if mask2.size > 0:
            mask2 = self.resize(mask2, image_shapes, "mask")
        if mask1.size == 0:
            return mask2
        elif mask2.size == 0:
            return mask1

        if alpha > self.max_alpha_mask_threshold:
            return mask1
        if alpha < self.min_alpha_mask_threshold:
            return mask2

        if alpha >= 0.5:
            mask1[(mask1 == 0) & (mask2 != 0)] = mask2[
                (mask1 == 0) & (mask2 != 0)
            ]
            return mask1

        mask2[(mask2 == 0) & (mask1 != 0)] = mask1[(mask2 == 0) & (mask1 != 0)]
        return mask2

    @override
    def apply_to_instance_masks(
        self,
        masks_batch: List[List[np.ndarray]],
        **_,
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of instance masks.

        @type masks_batch: List[List[np.ndarray]]
        @param masks_batch: Batch of input instance masks to which the
            transformation is applied.
        @rtype: List[np.ndarray]
        @return: Transformed instance masks.
        """
        return masks_batch[0] + masks_batch[1]

    @override
    def apply_to_bboxes(
        self,
        bboxes_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        rows: int,
        cols: int,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of bboxes.

        @type bboxes_batch: List[np.ndarray]
        @param bboxes_batch: Batch of input bboxes to which the
            transformation is applied.
        @rtype: np.ndarray
        @return: Transformed bboxes.
        """
        for i in range(len(bboxes_batch)):
            bbox = bboxes_batch[i]
            if bbox.size == 0:  # pragma: no cover
                bboxes_batch[i] = np.zeros((0, 6), dtype=bbox.dtype)

        bboxes_batch[1] = self.resize(
            bboxes_batch[1], image_shapes, "bboxes", rows=rows, cols=cols
        )

        return np.concatenate(bboxes_batch, axis=0)

    @override
    def apply_to_keypoints(
        self,
        keypoints_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        cols: int,
        rows: int,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of keypoints.

        @type keypoints_batch: List[np.ndarray]
        @param keypoints_batch: Batch of input keypoints to which the
            transformation is applied.
        @rtype: np.ndarray
        @return: Transformed keypoints.
        """
        for i in range(len(keypoints_batch)):
            if keypoints_batch[i].size == 0:  # pragma: no cover
                keypoints_batch[i] = np.zeros(
                    (0, 5), dtype=keypoints_batch[i].dtype
                )
        keypoints_batch[1] = self.resize(
            keypoints_batch[1],
            image_shapes,
            "keypoints",
            shape=image_shapes[0],
            cols=cols,
            rows=rows,
        )
        return np.concatenate(keypoints_batch, axis=0)

    @override
    def get_params(self) -> Dict[str, Any]:
        """Update parameters.

        @param params: Dictionary containing parameters.
        @type params: Dict[str, Any]
        @return: Dictionary containing updated parameters.
        @rtype: Dict[str, Any]
        """
        alpha = random.uniform(*self.alpha)
        return {"alpha": alpha}

    @override
    def get_params_dependent_on_data(
        self, params: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get parameters dependent on the targets.

        @param params: Dictionary containing parameters.
        @type params: Dict[str, Any]
        @return: Dictionary containing parameters dependent on the
            targets.
        @rtype: Dict[str, Any]
        """
        params = super().get_params_dependent_on_data(params, data)
        image_batch = data["image"]
        return {"image_shapes": [image.shape[:2] for image in image_batch]}
