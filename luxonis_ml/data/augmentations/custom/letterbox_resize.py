from typing import Any, Dict, Tuple

import cv2
import numpy as np
from albumentations import BoxType, DualTransform, KeypointType
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox

from ..utils import AUGMENTATIONS


@AUGMENTATIONS.register_module()
class LetterboxResize(DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_LINEAR,
        border_value: int = 0,
        mask_value: int = 0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        """Augmentation to apply letterbox resizing to images. Also transforms masks,
        bboxes and keypoints to correct shape.

        @param height: Desired height of the output.
        @type height: int
        @param width: Desired width of the output.
        @type width: int
        @param interpolation: Cv2 flag to specify interpolation used when resizing.
            Defaults to cv2.INTER_LINEAR.
        @type interpolation: int, optional
        @param border_value: Padding value for images. Defaults to 0.
        @type border_value: int, optional
        @param mask_value: Padding value for masks. Defaults to 0.
        @type mask_value: int, optional
        @param always_apply: Whether to always apply the transform. Defaults to False.
        @type always_apply: bool, optional
        @param p: Probability of applying the transform. Defaults to 1.0.
        @type p: float, optional
        """

        super().__init__(always_apply, p)

        if not (0 <= border_value <= 255):
            raise ValueError("Border value must be in range [0,255].")

        if not (0 <= mask_value <= 255):
            raise ValueError("Mask value must be in range [0,255].")

        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.border_value = border_value
        self.mask_value = mask_value

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Updates augmentation parameters with the necessary metadata.

        @param params: The existing augmentation parameters dictionary.
        @type params: Dict[str, Any]
        @param kwargs: Additional keyword arguments to add the parameters.
        @type kwargs: Any
        @return: Updated dictionary containing the merged parameters.
        @rtype: Dict[str, Any]
        """

        params = super().update_params(params, **kwargs)

        img_height = params["rows"]
        img_width = params["cols"]

        ratio = min(self.height / img_height, self.width / img_width)
        new_height = int(img_height * ratio)
        new_width = int(img_width * ratio)

        # only supports center alignment
        pad_top = (self.height - new_height) // 2
        pad_bottom = pad_top

        pad_left = (self.width - new_width) // 2
        pad_right = pad_left

        params.update(
            {
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
            }
        )

        return params

    def apply(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **kwargs,
    ) -> np.ndarray:
        """Applies the letterbox augmentation to an image.

        @param img: Input image to which resize is applied.
        @type img: np.ndarray
        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int
        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int
        @param params: Additional parameters for the padding operation.
        @type params: Any
        @return: Image with applied letterbox resize.
        @rtype: np.ndarray
        """

        resized_img = cv2.resize(
            img,
            (self.width - pad_left - pad_right, self.height - pad_top - pad_bottom),
            interpolation=self.interpolation,
        )
        img_out = cv2.copyMakeBorder(
            resized_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            self.border_value,
        )
        img_out = img_out.astype(img.dtype)
        return img_out

    def apply_to_mask(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params,
    ) -> np.ndarray:
        """Applies letterbox augmentation to the input mask.

        @param img: Input mask to which resize is applied.
        @type img: np.ndarray
        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int
        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int
        @param params: Additional parameters for the padding operation.
        @type params: Any
        @return: Mask with applied letterbox resize.
        @rtype: np.ndarray
        """

        resized_img = cv2.resize(
            img,
            (self.width - pad_left - pad_right, self.height - pad_top - pad_bottom),
            interpolation=cv2.INTER_NEAREST,
        )
        img_out = cv2.copyMakeBorder(
            resized_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            self.mask_value,
        )
        img_out = img_out.astype(img.dtype)
        return img_out

    def apply_to_bbox(
        self,
        bbox: BoxType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params,
    ) -> BoxType:
        """Applies letterbox augmentation to the bounding box.

        @param img: Bounding box to which resize is applied.
        @type img: BoxType
        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int
        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int
        @param params: Additional parameters for the padding operation.
        @type params: Any
        @return: Bounding box with applied letterbox resize.
        @rtype: BoxType
        """

        x_min, y_min, x_max, y_max = denormalize_bbox(
            bbox, self.height - pad_top - pad_bottom, self.width - pad_left - pad_right
        )[:4]
        bbox = np.array(
            [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
        )
        # clip bbox to image, ignoring padding
        bbox = bbox.clip(
            min=[pad_left, pad_top] * 2,
            max=[self.width - pad_left, self.height - pad_top] * 2,
        ).tolist()
        return normalize_bbox(bbox, self.height, self.width)

    def apply_to_keypoint(
        self,
        keypoint: KeypointType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params,
    ) -> KeypointType:
        """Applies letterbox augmentation to the keypoint.

        @param img: Keypoint to which resize is applied.
        @type img: KeypointType
        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int
        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int
        @param params: Additional parameters for the padding operation.
        @type params: Any
        @return: Keypoint with applied letterbox resize.
        @rtype: KeypointType
        """

        x, y, angle, scale = keypoint[:4]
        scale_x = (self.width - pad_left - pad_right) / params["cols"]
        scale_y = (self.height - pad_top - pad_bottom) / params["rows"]
        new_x = (x * scale_x) + pad_left
        new_y = (y * scale_y) + pad_top
        # if keypoint is in the padding then set coordinates to -1
        out_keypoint = (
            new_x
            if not self._out_of_bounds(new_x, pad_left, self.width - pad_left)
            else -1,
            new_y
            if not self._out_of_bounds(new_y, pad_top, self.height - pad_top)
            else -1,
            angle,
            scale * max(scale_x, scale_y),
        )
        return out_keypoint

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Gets the default arguments for the letterbox augmentation.

        @return: The string keywords of the arguments.
        @rtype: Tuple[str, ...]
        """

        return ("height", "width", "interpolation", "border_value", "mask_value")

    def _out_of_bounds(self, value: float, min_limit: float, max_limit: float) -> bool:
        """ "Check if the given value is outside the specified limits.

        @param value: The value to be checked.
        @type value: float
        @param min_limit: Minimum limit.
        @type min_limit: float
        @param max_limit: Maximum limit.
        @type max_limit: float
        @return: True if the value is outside the specified limits, False otherwise.
        @rtype: bool
        """
        return value < min_limit or value > max_limit
