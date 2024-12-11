import random
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from typing_extensions import override

from luxonis_ml.data.augmentations.batch_transform import BatchBasedTransform
from luxonis_ml.data.augmentations.custom import LetterboxResize


class MixUp(BatchBasedTransform):
    def __init__(
        self, alpha: Union[float, Tuple[float, float]] = 0.5, p: float = 0.5
    ):
        """MixUp augmentation that merges two images and their
        annotations into one. If images are not of same size then second
        one is first resized to match the first one.

        @type alpha: Union[float, Tuple[float, float]]
        @param alpha: Mixing coefficient, either a single float or a
            tuple representing the range. Defaults to C{0.5}.
        @type p: float, optional
        @param p: Probability of applying the transform. Defaults to
            C{0.5}.
        """
        super().__init__(batch_size=2, p=p)

        self.alpha = alpha if isinstance(alpha, tuple) else (alpha, alpha)
        self.letterbox = LetterboxResize(0, 0)
        self._check_alpha()

    def _check_alpha(self) -> None:
        if not 0 <= self.alpha[0] <= 1 or not 0 <= self.alpha[1] <= 1:
            raise ValueError("Alpha must be in range [0, 1].")

        if self.alpha[0] > self.alpha[1]:
            raise ValueError("Alpha range must be in ascending order.")

    @override
    def apply(
        self,
        image_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
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
        alpha = random.uniform(self.alpha[0], self.alpha[1])

        padding = self._update_letterbox_params(image_shapes)
        self.letterbox._image_fill_value = (
            int(255 * alpha),
            int(255 * alpha),
            int(255 * alpha),
        )

        image1 = image_batch[0]
        image2 = self.letterbox.apply(image_batch[1], *padding)

        return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0.0)

    @override
    def apply_to_mask(
        self,
        mask_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of masks.

        @type mask_batch: List[np.ndarray]
        @param mask_batch: Batch of input masks to which the
            transformation is applied.
        @type image_shapes: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @rtype: List[np.ndarray]
        @return: List of transformed masks.
        """
        padding = self._update_letterbox_params(image_shapes)
        mask1 = mask_batch[0]
        mask2 = self.letterbox.apply_to_mask(mask_batch[1], *padding)
        return np.minimum(mask1 + mask2, 1)

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
            if bbox.shape[1] == 4:
                bboxes_batch[i] = np.zeros((0, 6), dtype=bbox.dtype)

        padding = self._update_letterbox_params(image_shapes)
        bboxes_batch[1] = self.letterbox.apply_to_bboxes(
            bboxes_batch[1], *padding, rows=rows, cols=cols
        )

        return np.concatenate(bboxes_batch, axis=0)

    @override
    def apply_to_keypoints(
        self,
        keypoints_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of keypoints.

        @type keypoints_batch: List[np.ndarray]
        @param keypoints_batch: Batch of input keypoints to which the
            transformation is applied.
        @rtype: np.ndarray
        @return: Transformed keypoints.
        """
        padding = self._update_letterbox_params(image_shapes)
        rows, cols = image_shapes[1]
        keypoints_batch[1] = self.letterbox.apply_to_keypoints(
            keypoints_batch[1],
            *padding,
            rows=rows,
            cols=cols,
        )
        return np.concatenate(keypoints_batch, axis=0)

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

    def _update_letterbox_params(
        self, image_shapes: List[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        out_height, out_width = image_shapes[0]
        orig_height, orig_width = image_shapes[1]
        self.letterbox._height = out_height
        self.letterbox._width = out_width
        return LetterboxResize.compute_padding(
            orig_height, orig_width, out_height, out_width
        )
