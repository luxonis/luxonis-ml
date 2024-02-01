import random
from typing import Any, Dict, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from albumentations import BoxType, KeypointType

from ..batch_transform import BatchBasedTransform
from ..utils import AUGMENTATIONS


@AUGMENTATIONS.register_module()
class MixUp(BatchBasedTransform):
    def __init__(
        self,
        alpha: Union[float, Tuple[float, float]] = 0.5,
        out_batch_size: int = 1,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        """MixUp augmentation that merges two images and their annotations into one. If
        images are not of same size then second one is first resized to match the first
        one.

        @param alpha: Mixing coefficient, either a single float or a tuple representing
            the range. Defaults to 0.5.
        @type alpha: Union[float, Tuple[float, float]], optional
        @param out_batch_size: Number of output images in the batch. Defaults to 1.
        @type out_batch_size: int, optional
        @param always_apply: Whether to always apply the transform. Defaults to False.
        @type always_apply: bool, optional
        @param p: Probability of applying the transform. Defaults to 0.5.
        @type p: float, optional
        """
        super().__init__(batch_size=2, always_apply=always_apply, p=p)

        self.alpha = alpha
        self.out_batch_size = out_batch_size

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Gets the default arguments for the mixup augmentation.

        @return: The string keywords of the arguments.
        @rtype: Tuple[str, ...]
        """
        return ("alpha", "out_batch_size")

    @property
    def targets_as_params(self) -> List[str]:
        """List of augmentation targets.

        @return: Output list of augmentation targets.
        @rtype: List[str]
        """
        return ["image_batch"]

    def apply_to_image_batch(
        self,
        image_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of images.

        @param image_batch: Batch of input images to which the transformation is
            applied.
        @type image_batch: List[np.ndarray]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed images.
        @rtype: List[np.ndarray]
        """
        image1 = image_batch[0]
        # resize second image to size of the first one
        image2 = cv2.resize(image_batch[1], (image_shapes[0][1], image_shapes[0][0]))

        if isinstance(self.alpha, float):
            curr_alpha = np.clip(self.alpha, 0, 1)
        else:
            curr_alpha = random.uniform(max(self.alpha[0], 0), min(self.alpha[1], 1))
        img_out = cv2.addWeighted(image1, curr_alpha, image2, 1 - curr_alpha, 0.0)
        return [img_out]

    def apply_to_mask_batch(
        self,
        mask_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of masks.

        @param image_batch: Batch of input masks to which the transformation is applied.
        @type image_batch: List[np.ndarray]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed masks.
        @rtype: List[np.ndarray]
        """
        mask1 = mask_batch[0]
        mask2 = cv2.resize(
            mask_batch[1],
            (image_shapes[0][1], image_shapes[0][0]),
            interpolation=cv2.INTER_NEAREST,
        )
        out_mask = mask1 + mask2
        # if masks intersect keep one present in first image
        mask_inter = mask1 > 0
        out_mask[mask_inter] = mask1[mask_inter]
        return [out_mask]

    def apply_to_bboxes_batch(
        self, bboxes_batch: List[BoxType], image_shapes: List[Tuple[int, int]], **params
    ) -> List[BoxType]:
        """Applies the transformation to a batch of bboxes.

        @param image_batch: Batch of input bboxes to which the transformation is
            applied.
        @type image_batch: List[BoxType]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed bboxes.
        @rtype: List[BoxType]
        """
        return [bboxes_batch[0] + bboxes_batch[1]]

    def apply_to_keypoints_batch(
        self,
        keypoints_batch: List[KeypointType],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[KeypointType]:
        """Applies the transformation to a batch of keypoints.

        @param image_batch: Batch of input keypoints to which the transformation is
            applied.
        @type image_batch: List[BoxType]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed keypoints.
        @rtype: List[BoxType]
        """
        scaled_kpts2 = []
        scale_x = image_shapes[0][1] / image_shapes[1][1]
        scale_y = image_shapes[0][0] / image_shapes[1][0]
        for kpt in keypoints_batch[1]:
            new_kpt = A.augmentations.geometric.functional.keypoint_scale(
                keypoint=kpt, scale_x=scale_x, scale_y=scale_y
            )
            scaled_kpts2.append(new_kpt + kpt[4:])
        return [keypoints_batch[0] + scaled_kpts2]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters dependent on the targets.

        @param params: Dictionary containing parameters.
        @type params: Dict[str, Any]
        @return: Dictionary containing parameters dependent on the targets.
        @rtype: Dict[str, Any]
        """
        image_batch = params["image_batch"]
        return {"image_shapes": [image.shape[:2] for image in image_batch]}
