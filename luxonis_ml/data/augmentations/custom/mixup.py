import random
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from typing_extensions import override

from ..batch_transform import BatchBasedTransform


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
        self._check_alpha()

    def _check_alpha(self) -> None:
        if not 0 <= self.alpha[0] <= 1 or not 0 <= self.alpha[1] <= 1:
            raise ValueError("Alpha must be in range [0, 1].")

        if self.alpha[0] > self.alpha[1]:
            raise ValueError("Alpha range must be in ascending order.")

    @override
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Gets the default arguments for the mixup augmentation.

        @rtype: Tuple[str, ...]
        @return: The string keywords of the arguments.
        """
        return ("alpha",)

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
        image1 = image_batch[0]
        image2 = cv2.resize(
            image_batch[1], (image_shapes[0][1], image_shapes[0][0])
        )

        curr_alpha = random.uniform(
            max(self.alpha[0], 0), min(self.alpha[1], 1)
        )
        return cv2.addWeighted(image1, curr_alpha, image2, 1 - curr_alpha, 0.0)

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
        mask1 = mask_batch[0]
        mask2 = cv2.resize(
            mask_batch[1],
            (image_shapes[0][1], image_shapes[0][0]),
            interpolation=cv2.INTER_NEAREST,
        )
        out_mask = mask1 + mask2
        # if masks intersect keep the one from the first image
        mask_inter = mask1 > 0
        out_mask[mask_inter] = mask1[mask_inter]
        return out_mask

    @override
    def apply_to_bboxes(
        self,
        bboxes_batch: List[np.ndarray],
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

        return np.concatenate(bboxes_batch, axis=0)

    @override
    def apply_to_keypoints(
        self,
        keypoints_batch: List[np.ndarray],
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of keypoints.

        @type keypoints_batch: List[np.ndarray]
        @param keypoints_batch: Batch of input keypoints to which the
            transformation is applied.
        @rtype: np.ndarray
        @return: Transformed keypoints.
        """
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
        image_batch = data["image"]
        return {"image_shapes": [image.shape[:2] for image in image_batch]}
