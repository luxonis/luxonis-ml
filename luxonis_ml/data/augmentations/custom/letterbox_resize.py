from typing import Any, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from typing_extensions import override

from luxonis_ml.data.utils.visualizations import resolve_color
from luxonis_ml.typing import Color


class LetterboxResize(A.DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_LINEAR,
        image_fill_value: Color = "black",
        mask_fill_value: int = 0,
        p: float = 1.0,
    ):
        """Augmentation to apply letterbox resizing to images. Also
        transforms masks, bboxes and keypoints to correct shape.

        @type height: int
        @param height: Desired height of the output.
        @type width: int
        @param width: Desired width of the output.
        @type interpolation: int
        @param interpolation: cv2 flag to specify interpolation used
            when resizing. Defaults to C{cv2.INTER_LINEAR}.
        @type image_fill_value: int
        @param image_fill_value: Padding value for images. Defaults to
            "black".
        @type mask_fill_value: int
        @param mask_fill_value: Padding value for masks. Must be an
            integer representing the class label. Defaults to C{0}
            (background class).
        @type p: float
        @param p: Probability of applying the transform. Defaults to
            C{1.0}.
        """

        super().__init__(p)

        self._height = height
        self._width = width
        self._interpolation = interpolation
        self._image_fill_value = resolve_color(image_fill_value)
        self._mask_fill_value = resolve_color(mask_fill_value)

    @override
    def update_params(
        self, params: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Updates augmentation parameters with the necessary metadata.

        @param params: The existing augmentation parameters dictionary.
        @type params: Dict[str, Any]
        @param kwargs: Additional keyword arguments to add the
            parameters.
        @type kwargs: Any
        @return: Updated dictionary containing the merged parameters.
        @rtype: Dict[str, Any]
        """

        params = super().update_params(params, **kwargs)

        height = params["rows"]
        width = params["cols"]

        ratio = min(self._height / height, self._width / width)
        new_height = int(height * ratio)
        new_width = int(width * ratio)

        # only supports center alignment
        pad_top = (self._height - new_height) // 2
        pad_bottom = pad_top

        pad_left = (self._width - new_width) // 2
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

    @override
    def apply(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **_,
    ) -> np.ndarray:
        """Applies the letterbox augmentation to an image.

        @type img: np.ndarray
        @param img: Input image to which resize is applied.
        @type pad_top: int
        @param pad_top: Number of pixels to pad at the top.
        @type pad_bottom: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_left: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_right: int
        @param pad_right: Number of pixels to pad on the right.
        @rtype: np.ndarray
        @return: Image with applied letterbox resize.
        """
        return self._apply_to_image_data(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self._interpolation,
            self._image_fill_value,
        )

    @override
    def apply_to_mask(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **_,
    ) -> np.ndarray:
        """Applies letterbox augmentation to the input mask.

        @type img: np.ndarray
        @param img: Input mask to which resize is applied.
        @type pad_top: int
        @param pad_top: Number of pixels to pad at the top.
        @type pad_bottom: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_left: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_right: int
        @param pad_right: Number of pixels to pad on the right.
        @type params: Any
        @param params: Additional parameters for the padding operation.
        @rtype: np.ndarray
        @return: Mask with applied letterbox resize.
        """
        return self._apply_to_image_data(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.INTER_NEAREST,
            self._mask_fill_value,
        )

    @override
    def apply_to_bboxes(
        self,
        bbox: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **_,
    ) -> np.ndarray:
        """Applies letterbox augmentation to the bounding box.

        @type bbox: np.ndarray
        @param bbox: Bounding box to which resize is applied.
            Shape: (N, 6) where N is the number of bounding boxes.
            The bbox format is (x1, y1, x2, y2, class_id, instance_id).
        @type pad_top: int
        @param pad_top: Number of pixels to pad at the top.
        @type pad_bottom: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_left: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_right: int
        @param pad_right: Number of pixels to pad on the right.
        @rtype: np.ndarray
        @return: Bounding box with applied letterbox resize.
        """

        if bbox.shape[0] == 0:
            return bbox

        bbox = denormalize_bboxes(
            bbox,
            (
                self._height - pad_top - pad_bottom,
                self._width - pad_left - pad_right,
            ),
        )
        bbox[..., :4] += np.array([pad_left, pad_top] * 2)
        bbox[..., :4] = bbox[..., :4].clip(
            min=[pad_left, pad_top] * 2,
            max=[self._width - pad_left, self._height - pad_top] * 2,
        )

        return normalize_bboxes(bbox, (self._height, self._width))

    @override
    def apply_to_keypoints(
        self,
        keypoint: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        cols: int,
        rows: int,
        **_,
    ) -> np.ndarray:
        """Applies letterbox augmentation to the keypoint.

        @type keypoint: np.ndarray
        @param keypoint: Keypoint to which resize is applied.
            Shape: (N, 5) where N is the number of keypoints.
            The keypoint format is (x, y, angle, scale, visibility).
        @type pad_top: int
        @param pad_top: Number of pixels to pad at the top.
        @type pad_bottom: int
        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_left: int
        @param pad_left: Number of pixels to pad on the left.
        @type pad_right: int
        @param pad_right: Number of pixels to pad on the right.
        @type kwargs: Any
        @param kwargs: Additional parameters for the padding operation.
        @rtype: np.ndarray
        @return: Keypoint with applied letterbox resize.
        """

        if keypoint.shape[0] == 0:
            return keypoint

        scale_x = (self._width - pad_left - pad_right) / cols
        scale_y = (self._height - pad_top - pad_bottom) / rows
        keypoint[:, 0] *= scale_x
        keypoint[:, 0] += pad_left

        keypoint[:, 1] *= scale_y
        keypoint[:, 1] += pad_top

        out_of_bounds_x = np.logical_or(
            keypoint[:, 0] < pad_left, keypoint[:, 0] > self._width - pad_right
        )
        out_of_bounds_y = np.logical_or(
            keypoint[:, 1] < pad_top,
            keypoint[:, 1] > self._height - pad_bottom,
        )
        keypoint[out_of_bounds_x | out_of_bounds_y, :2] = -1

        return keypoint

    @override
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Gets the default arguments for the letterbox augmentation.

        @rtype: Tuple[str, ...]
        @return: The string keywords of the arguments.
        """

        return (
            "height",
            "width",
            "interpolation",
            "border_value",
            "mask_value",
        )

    def _apply_to_image_data(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        interpolation: int,
        fill_value: Tuple[int, int, int],
    ) -> np.ndarray:
        resized_img = cv2.resize(
            img,
            (
                self._width - pad_left - pad_right,
                self._height - pad_top - pad_bottom,
            ),
            interpolation=interpolation,
        )
        return cv2.copyMakeBorder(
            resized_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=fill_value,
        ).astype(img.dtype)
