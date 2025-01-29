from typing import Any, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override

from luxonis_ml.data.utils.visualizations import resolve_color
from luxonis_ml.typing import RGB, Color


class LetterboxResize(A.DualTransform):
    mask_fill_value: RGB

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

        super().__init__(p=p)

        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.image_fill_value = resolve_color(image_fill_value)
        self.mask_fill_value = resolve_color(mask_fill_value)

    @property
    @override
    def targets(self) -> Dict[str, Any]:
        targets = super().targets
        targets["instance_mask"] = self.apply_to_mask
        return targets

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
        pad_top, pad_bottom, pad_left, pad_right = self.compute_padding(
            params["rows"], params["cols"], self.height, self.width
        )

        params.update(
            {
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
                "pad_left": pad_left,
                "pad_right": pad_right,
            }
        )

        return params

    @staticmethod
    def compute_padding(
        orig_height: int, orig_width: int, out_height: int, out_width: int
    ) -> Tuple[int, int, int, int]:
        """Computes the padding required to resize an image to a
        letterbox format.

        @type orig_height: int
        @param orig_height: Original height of the image.
        @type orig_width: int
        @param orig_width: Original width of the image.
        @type out_height: int
        @param out_height: Desired height of the output.
        @type out_width: int
        @param out_width: Desired width of the output.
        @rtype: Tuple[int, int, int, int]
        @return: Padding values for the top, bottom, left and right
            sides of the image.
        """
        ratio = min(out_height / orig_height, out_width / orig_width)
        new_height = int(orig_height * ratio)
        new_width = int(orig_width * ratio)

        pad_top = (out_height - new_height) // 2
        pad_bottom = pad_top

        pad_left = (out_width - new_width) // 2
        pad_right = pad_left
        return pad_top, pad_bottom, pad_left, pad_right

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
            self.interpolation,
            self.image_fill_value,
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
        """Applies letterbox augmentation to the input mask."""
        return self._apply_to_image_data(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.INTER_NEAREST,
            self.mask_fill_value,
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
        """Applies letterbox augmentation to the bounding box."""

        if bbox.size == 0:
            return bbox

        pad_left_norm = pad_left / self.width
        pad_right_norm = pad_right / self.width
        pad_top_norm = pad_top / self.height
        pad_bottom_norm = pad_bottom / self.height

        bbox[:, [0, 2]] *= 1 - pad_left_norm - pad_right_norm
        bbox[:, [0, 2]] += pad_left_norm

        bbox[:, [1, 3]] *= 1 - pad_top_norm - pad_bottom_norm
        bbox[:, [1, 3]] += pad_top_norm

        np.clip(
            bbox[:, [0, 2]],
            pad_left_norm,
            1 - pad_right_norm,
            out=bbox[:, [0, 2]],
        )
        np.clip(
            bbox[:, [1, 3]],
            pad_top_norm,
            1 - pad_bottom_norm,
            out=bbox[:, [1, 3]],
        )

        return bbox

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
        """Applies letterbox augmentation to the keypoint."""

        if keypoint.size == 0:
            return keypoint

        scale_x = (self.width - pad_left - pad_right) / cols
        scale_y = (self.height - pad_top - pad_bottom) / rows
        keypoint[:, 0] *= scale_x
        keypoint[:, 0] += pad_left

        keypoint[:, 1] *= scale_y
        keypoint[:, 1] += pad_top

        out_of_bounds_x = np.logical_or(
            keypoint[:, 0] < pad_left, keypoint[:, 0] > self.width - pad_right
        )
        out_of_bounds_y = np.logical_or(
            keypoint[:, 1] < pad_top, keypoint[:, 1] > self.height - pad_bottom
        )
        keypoint[out_of_bounds_x | out_of_bounds_y, :2] = -1

        return keypoint

    def _apply_to_image_data(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        interpolation: int,
        fill_value: RGB,
    ) -> np.ndarray:
        resized_img = cv2.resize(
            img,
            (
                self.width - pad_left - pad_right,
                self.height - pad_top - pad_bottom,
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
