from typing import Any

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override

from luxonis_ml.data.utils.visualizations import resolve_color
from luxonis_ml.typing import RGB, Color


class LetterboxResize(A.DualTransform):
    """Augmentation that resizes an image with padding to
    maintain the aspect ratio.

    Attributes:
        height: The desired height of the output image.
        width: The desired width of the output image.

    """

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_LINEAR,
        image_fill_value: Color = "black",
        mask_fill_value: int = 0,
        p: float = 1.0,
    ):
        """Create a ``LetterboxResize`` augmentation.

        Args:
            height: The desired height of the output image
            width: The desired width of the output image
            interpolation: ``cv2`` flag to specify interpolation used
                when resizing. Defaults to ``cv2.INTER_LINEAR``.
            image_fill_value: Padding value for images.
                Can be a string color name or an RGB tuple.
                Defaults to ``"black"``.
            mask_fill_value: Padding value for masks. Must be an integer
                representing a class label. Defaults to ``0`` (background).
            p: The probability of applying the transform. Defaults to ``1.0``.

        """

        super().__init__(p=p)

        self.height = height
        self.width = width

        self._interpolation = interpolation
        self._image_fill_value = resolve_color(image_fill_value)
        self._mask_fill_value = resolve_color(mask_fill_value)

    @property
    @override
    def targets(self) -> dict[str, Any]:
        """Define the targets the augmentation will be applied to."""
        targets = super().targets
        targets["instance_mask"] = self.apply_to_mask
        return targets

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Return parameters dependent on input.

        Args:
            params: The existing augmentation parameters dictionary.
            data: The dictionary with input data.

        Returns:
            A dictionary with extra parameters required for the
            augmentation, such as padding values and original image dimensions.

        """
        orig_height, orig_width, _ = params["shape"]

        pad_top, pad_bottom, pad_left, pad_right = self.compute_padding(
            orig_height, orig_width, self._height, self._width
        )
        return {
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "orig_width": orig_width,
            "orig_height": orig_height,
        }

    @staticmethod
    def compute_padding(
        orig_height: int, orig_width: int, out_height: int, out_width: int
    ) -> tuple[int, int, int, int]:
        """Compute the padding required to resize an image to the
        letterbox format.

        Args:
            orig_height: Original height of the image.
            orig_width: Original width of the image.
            out_height: Desired height of the output.
            out_width: Desired width of the output.

        Returns:
            Padding values for the top, bottom, left and right
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
        r"""Apply the letterbox augmentation to an image.

        Args:
            img: The input image of shape
                :math:`\left(\rightH, W, \ldots\right)` to which the
                letterbox resize will be applied.

            pad_top: The number of pixels to pad at the top of the image.
            pad_bottom: The number of pixels to pad at the bottom of the image.
            pad_left: The number of pixels to pad on the left side of the image.
            pad_right: The number of pixels to pad on the right
                side of the image.

        Returns:
            Resized and padded image.

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
        mask: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **_,
    ) -> np.ndarray:
        r"""Apply letterbox augmentation to the input mask.

        Args:
            mask: The input mask of shape :math:`\left(H, W, \ldots\right)`
                to which the letterbox resize will be applied.
            pad_top: The number of pixels to pad at the top of the mask.
            pad_bottom: The number of pixels to pad at the bottom of the mask.
            pad_left: The number of pixels to pad on the left side of the mask.
            pad_right: The number of pixels to pad on the right
                side of the mask.

        Returns:
            Resized and padded mask.

        """
        return self._apply_to_image_data(
            mask,
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
        r"""Apply letterbox augmentation to the bounding box.

        Args:
            bbox: The input bounding boxes of shape :math:`\left(N, 4\right)`
                to which the letterbox resize will be applied.
                Individual bounding boxes should be in the format
                :math:`\left(x_{min}, y_{min}, x_{max}, y_{max}\right)`
                and normalized to the range :math:`\left[0, 1\right]`.

            pad_top: The number of pixels to pad at the top of the image.
            pad_bottom: The number of pixels to pad at the bottom of the image.
            pad_left: The number of pixels to pad on the left side of the image.
            pad_right: The number of pixels to pad on the right
                side of the image.

        Returns:
            Transformed bounding boxes in the same format and normalization
            as the input.

        """

        if bbox.size == 0:
            return bbox

        pad_left_norm = pad_left / self._width
        pad_right_norm = pad_right / self._width
        pad_top_norm = pad_top / self._height
        pad_bottom_norm = pad_bottom / self._height

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
        orig_height: int,
        orig_width: int,
        **_,
    ) -> np.ndarray:
        r"""Apply letterbox augmentation to the keypoint.

        Args:
            keypoint: The input keypoints of shape :math:`\left(N, 2+\right)`
                to which the letterbox resize will be applied.
                Individual keypoints should be in the format
                :math:`\left(x, y, \ldots\right)` and normalized to the range
                :math:`\left[0, 1\right]`.

            pad_top: The number of pixels to pad at the top of the image.
            pad_bottom: The number of pixels to pad at the bottom of the image.
            pad_left: The number of pixels to pad on the left side of the image.
            pad_right: The number of pixels to pad on the right
                side of the image.

            orig_height: Original height of the image before resizing.
            orig_width: Original width of the image before resizing.

        Returns:
            Transformed keypoints in the same format and normalization
            as the input. Keypoints that fall outside the image boundaries
            after transformation will have their coordinates set to :math:`-1`.

        """

        if keypoint.size == 0:
            return keypoint

        scale_x = (self._width - pad_left - pad_right) / orig_width
        scale_y = (self._height - pad_top - pad_bottom) / orig_height
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
                self._width - pad_left - pad_right,
                self._height - pad_top - pad_bottom,
            ),
            interpolation=interpolation,
        )
        padded_img = cv2.copyMakeBorder(
            resized_img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=fill_value,
        ).astype(img.dtype)

        if padded_img.ndim == 2:
            padded_img = padded_img[..., None]

        return padded_img
