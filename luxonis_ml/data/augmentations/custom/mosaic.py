import math
import random
from typing import Any

import cv2
import numpy as np
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from typing_extensions import override

from luxonis_ml.data.augmentations.batch_transform import BatchTransform
from luxonis_ml.utils.logging import deprecated


class Mosaic4(BatchTransform):
    r"""Batch-based augmentation that creates a mosaic of four images.

    The transform arranges four images in a deterministic :math:`2 \times 2`
    grid:

    .. table::
       :widths: auto
       :align: center

       ==== ====
        1    2
        3    4
       ==== ====

    Images may have different sizes, but they must have
    the same number of channels. The result is cropped around the
    mosaic center to ``out_width`` by ``out_height`` and padded when
    needed with the specified fill values.

    .. figure::
       https://github.com/luxonis/luxonis-ml/blob/58fb10adc38a393640b9dc889fdefd1db81f9900/luxonis_ml/data/augmentations/media/mosaic4.png
       :height: 300px
       :width: 400px
       :loading: embed

       An example of the Mosaic4 augmentation.

    See:
        `Dynamic Scale Training for Object Detection`_

    .. _Dynamic Scale Training for Object Detection:
        https://arxiv.org/abs/2004.12432
    """

    @deprecated(
        "out_height",
        "out_width",
        "value",
        "mask_value",
        suggest={
            "out_height": "height",
            "out_width": "width",
            "value": "image_fill_value",
            "mask_value": "mask_fill_value",
        },
    )
    def __init__(
        self,
        height: int | None = None,
        width: int | None = None,
        image_fill_value: float | list[int] | list[float] | None = None,
        mask_fill_value: float | list[int] | list[float] | None = None,
        p: float = 0.5,
        out_height: int | None = None,
        out_width: int | None = None,
        value: float | list[int] | list[float] | None = None,
        mask_value: float | list[int] | list[float] | None = None,
    ):
        """Create a Mosaic4 augmentation.


        Args:
            height: Output image height.
            width: Output image width.
            image_fill_value: Padding value for images.
            mask_fill_value: Padding value for masks.
            p: Probability of applying the transform.
            out_height:
              .. deprecated:: 0.9.0
                    Use ``height`` instead.
            out_width:
              .. deprecated:: 0.9.0
                    Use ``width`` instead.
            value:
              .. deprecated:: 0.9.0
                    Use ``image_fill_value`` instead.
            mask_value:
              .. deprecated:: 0.9.0
                    Use ``mask_fill_value`` instead.

        """
        super().__init__(batch_size=4, p=p)
        height = height if height is not None else out_height
        width = width if width is not None else out_width

        if height is None or height <= 0:
            raise ValueError(
                f"`out_height` must be larger than 0, got {height}"
            )
        if width is None or width <= 0:
            raise ValueError(f"`out_width` must be larger than 0, got {width}")

        self._height = height
        self._width = width
        self._image_fill_value = value or image_fill_value
        self._mask_fill_value = mask_value or mask_fill_value

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get parameters dependent on the targets.

        Args:
            params: Existing augmentation parameters.
            data: Input data.

        Returns:
            Parameters derived from the input targets.

        """
        x_crop, y_crop = self.generate_random_crop_center()
        return {
            "x_crop": x_crop,
            "y_crop": y_crop,
            "out_width": self._width,
            "out_height": self._height,
        }

    def generate_random_crop_center(self) -> tuple[int, int]:
        """Generate a random crop center within the bounds of the mosaic
        image size.
        """
        crop_x = random.randint(0, max(0, self._width))
        crop_y = random.randint(0, max(0, self._height))
        return crop_x, crop_y

    @override
    def apply(
        self, image_batch: list[np.ndarray], x_crop: int, y_crop: int, **_
    ) -> np.ndarray:
        r"""Apply mosaic augmentation to a batch of images.

        Args:
            image_batch: Images to transform. Each image should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            x_crop: X-coordinate of the crop start point.
            y_crop: Y-coordinate of the crop start point.

        Returns:
            A single image of shape :math:`\left(H_{out}, W_{out}, C\right)` or
            :math:`\left(H_{out}, W_{out}\right)`.

        """
        return self._apply_mosaic4_to_images(
            image_batch,
            self._height,
            self._width,
            x_crop,
            y_crop,
            self._image_fill_value,
        )

    @override
    def apply_to_mask(
        self,
        mask_batch: list[np.ndarray],
        x_crop: int,
        y_crop: int,
        out_height: int,
        out_width: int,
        **_,
    ) -> np.ndarray:
        r"""Apply mosaic augmentation to a batch of semantic segmentation masks.

        Args:
            mask_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, C\right)` or :math:`\left(H, W\right)`.
            x_crop: X-coordinate of the crop start point.
            y_crop: Y-coordinate of the crop start point.
            out_height: The expected height of the output mask.
            out_width: The expected width of the output mask.

        Returns:
            A single segmentation mask of shape
            :math:`\left(H_{out}, W_{out}, C\right)` or
            :math:`\left(H_{out}, W_{out}\right)`.

        """
        for i in range(len(mask_batch)):
            mask = mask_batch[i]
            if mask.size == 0:
                if len(mask.shape) == 2:
                    mask_batch[i] = np.zeros(
                        (out_width, out_height), dtype=mask.dtype
                    )
                else:
                    mask_batch[i] = np.zeros(
                        (out_width, out_height, mask.shape[-1]),
                        dtype=mask.dtype,
                    )
        return self._apply_mosaic4_to_images(
            mask_batch,
            self._height,
            self._width,
            x_crop,
            y_crop,
            self._mask_fill_value,
        )

    @override
    def apply_to_instance_mask(
        self, masks_batch: list[np.ndarray], x_crop: int, y_crop: int, **_
    ) -> np.ndarray:
        r"""Apply mosaic augmentation to a batch of instance segmentation masks.

        Args:
            masks_batch: Masks to transform. Each mask should be of shape
                :math:`\left(H, W, N\right)`, where :math:`N`
                is the number of instances.
            x_crop: X-coordinate of the crop start point.
            y_crop: Y-coordinate of the crop start point.

        Returns:
            A single instance masks of shape
            :math:`\left(H_{out}, W_{out}, N\right)`.


        """
        return self._apply_mosaic4_to_instance_masks(
            masks_batch,
            self._height,
            self._width,
            x_crop,
            y_crop,
            self._image_fill_value,
        )

    @override
    def apply_to_bboxes(
        self,
        bboxes_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        x_crop: int,
        y_crop: int,
        **_,
    ) -> np.ndarray:
        """Apply mosaic augmentation to a batch of bounding boxes.

        Args:
            bboxes_batch: Bounding boxes to transform.
            image_shapes: Original image shapes.
            x_crop: X-coordinate of the crop start point.
            y_crop: Y-coordinate of the crop start point.

        Returns:
            Transformed bounding boxes.

        """
        new_bboxes = []
        for i, (bboxes, (orig_height, orig_width)) in enumerate(
            zip(bboxes_batch, image_shapes, strict=True)
        ):
            if bboxes.size == 0:  # pragma: no cover
                bboxes = np.zeros((0, 6), dtype=bboxes.dtype)

            bbox = self._apply_mosaic4_to_bboxes(
                bboxes,
                orig_height,
                orig_width,
                i,
                self._height,
                self._width,
                x_crop,
                y_crop,
            )
            new_bboxes.append(bbox)

        return np.concatenate(new_bboxes, axis=0)

    @override
    def apply_to_keypoints(
        self,
        keypoints_batch: list[np.ndarray],
        image_shapes: list[tuple[int, int]],
        x_crop: int,
        y_crop: int,
        **_,
    ) -> np.ndarray:
        """Apply mosaic augmentation to a batch of keypoints.

        Args:
            keypoints_batch: Keypoints to transform.
            image_shapes: Original image shapes.
            x_crop: X-coordinate of the crop start point.
            y_crop: Y-coordinate of the crop start point.

        Returns:
            Transformed keypoints.

        """
        new_keypoints = []
        for i, (keypoints, (orig_height, orig_width)) in enumerate(
            zip(keypoints_batch, image_shapes, strict=True)
        ):
            if keypoints.size == 0:
                keypoints = np.zeros((0, 6), dtype=keypoints.dtype)

            new_keypoint = self._apply_mosaic4_to_keypoints(
                keypoints,
                orig_height,
                orig_width,
                i,
                self._height,
                self._width,
                x_crop,
                y_crop,
            )
            new_keypoints.append(new_keypoint)
        return np.concatenate(new_keypoints, axis=0)

    @staticmethod
    def _compute_mosaic4_corners(
        quadrant: int,
        out_height: int,
        out_width: int,
        in_height: int,
        in_width: int,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        if quadrant == 0:
            x1a, y1a, x2a, y2a = (
                max(out_width - in_width, 0),
                max(out_height - in_height, 0),
                out_width,
                out_height,
            )
            x1b, y1b, x2b, y2b = (
                in_width - (x2a - x1a),
                in_height - (y2a - y1a),
                in_width,
                in_height,
            )

        elif quadrant == 1:
            x1a, y1a, x2a, y2a = (
                out_width,
                max(out_height - in_height, 0),
                min(out_width + in_width, out_width * 2),
                out_height,
            )
            x1b, y1b, x2b, y2b = (
                0,
                in_height - (y2a - y1a),
                min(in_width, x2a - x1a),
                in_height,
            )
        elif quadrant == 2:
            x1a, y1a, x2a, y2a = (
                max(out_width - in_width, 0),
                out_height,
                out_width,
                min(out_height * 2, out_height + in_height),
            )
            x1b, y1b, x2b, y2b = (
                in_width - (x2a - x1a),
                0,
                in_width,
                min(y2a - y1a, in_height),
            )
        else:
            x1a, y1a, x2a, y2a = (
                out_width,
                out_height,
                min(out_width + in_width, out_width * 2),
                min(out_height * 2, out_height + in_height),
            )
            x1b, y1b, x2b, y2b = (
                0,
                0,
                min(in_width, x2a - x1a),
                min(y2a - y1a, in_height),
            )
        return (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b)

    @staticmethod
    def _apply_mosaic4_to_instance_masks(
        masks_batch: list[np.ndarray],
        out_height: int,
        out_width: int,
        x_crop: int,
        y_crop: int,
        value: float | list[int] | list[float] | None = None,
    ) -> np.ndarray:
        out_masks = []
        out_shape = [out_height * 2, out_width * 2]

        if not any(m.size for m in masks_batch):
            return np.zeros((out_height, out_width, 0), dtype=np.uint8)

        for quadrant, masks in enumerate(masks_batch):
            if masks.size == 0:
                continue

            for i in range(masks.shape[-1]):
                mask = masks[..., i]
                imgs = max(out_height, out_width)
                h, w = mask.shape
                r = imgs / max(h, w)
                if r != 1:
                    w, h = (
                        min(math.ceil(w * r), imgs),
                        min(math.ceil(h * r), imgs),
                    )
                    mask = cv2.resize(
                        mask, (w, h), interpolation=cv2.INTER_NEAREST
                    )

                combined_mask = np.full(
                    out_shape,
                    value if value is not None else 0,
                    dtype=masks.dtype,
                )
                (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b) = (
                    Mosaic4._compute_mosaic4_corners(
                        quadrant, out_height, out_width, h, w
                    )
                )

                combined_region = combined_mask[y1a:y2a, x1a:x2a]
                mask_region = mask[y1b:y2b, x1b:x2b]

                combined_height, combined_width = combined_region.shape[:2]
                img_h, img_w = mask_region.shape[:2]

                min_h = min(combined_height, img_h)
                min_w = min(combined_width, img_w)

                combined_mask[y1a : y1a + min_h, x1a : x1a + min_w] = mask[
                    y1b : y1b + min_h, x1b : x1b + min_w
                ]

                combined_mask = combined_mask[
                    y_crop : y_crop + out_height, x_crop : x_crop + out_width
                ]
                out_masks.append(combined_mask)

        if not out_masks:
            return np.zeros((out_height, out_width, 0), dtype=np.uint8)

        return np.stack(out_masks, axis=-1)

    @staticmethod
    def _apply_mosaic4_to_images(
        image_batch: list[np.ndarray],
        out_height: int,
        out_width: int,
        x_crop: int,
        y_crop: int,
        padding: float | list[int] | list[float] | None = None,
    ) -> np.ndarray:
        r"""Arrange the images in a :math:`2 \times 2` grid layout.

        The input images should have the same number of channels but can
        have different widths and heights. The gaps are filled by the
        padding value.
        """
        if len(image_batch[0].shape) == 2:
            out_shape = [out_height * 2, out_width * 2]
        else:
            out_shape = [
                out_height * 2,
                out_width * 2,
                image_batch[0].shape[2],
            ]

        combined_image = np.full(
            out_shape,
            padding if padding is not None else 0,
            dtype=image_batch[0].dtype,
        )

        for quadrant, img in enumerate(image_batch):
            h, w = img.shape[:2]
            imgsz = max(out_height, out_width)
            r = imgsz / max(h, w)
            if r != 1:
                w, h = (
                    min(math.ceil(w * r), imgsz),
                    min(math.ceil(h * r), imgsz),
                )
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                if img.ndim == 2:
                    img = img.reshape(h, w, 1)

            (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b) = (
                Mosaic4._compute_mosaic4_corners(
                    quadrant, out_height, out_width, h, w
                )
            )

            combined_region = combined_image[y1a:y2a, x1a:x2a]
            img_region = img[y1b:y2b, x1b:x2b]

            combined_height, combined_width = combined_region.shape[:2]
            img_h, img_w = img_region.shape[:2]

            min_h = min(combined_height, img_h)
            min_w = min(combined_width, img_w)

            combined_image[y1a : y1a + min_h, x1a : x1a + min_w] = img[
                y1b : y1b + min_h, x1b : x1b + min_w
            ]

        return combined_image[
            y_crop : y_crop + out_height, x_crop : x_crop + out_width
        ]

    @staticmethod
    def _compute_shifts_for_quadrant(
        position_index: int,
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
    ) -> tuple[int, int]:
        if position_index == 0:
            shift_x = out_width - in_width
            shift_y = out_height - in_height
        elif position_index == 1:
            shift_x = out_width
            shift_y = out_height - in_height
        elif position_index == 2:
            shift_x = out_width - in_width
            shift_y = out_height
        elif position_index == 3:
            shift_x = out_width
            shift_y = out_height

        return shift_x, shift_y

    @staticmethod
    def _apply_mosaic4_to_bboxes(
        bbox: np.ndarray,
        in_height: int,
        in_width: int,
        position_index: int,
        out_height: int,
        out_width: int,
        x_crop: int,
        y_crop: int,
    ) -> np.ndarray:
        bbox = denormalize_bboxes(bbox, (in_height, in_width))

        imgs = max(out_height, out_width)
        r = imgs / max(in_height, in_width)
        if r != 1:
            in_width, in_height = (
                min(math.ceil(in_width * r), imgs),
                min(math.ceil(in_height * r), imgs),
            )
            bbox[:, :4] = bbox[:, :4] * r

        shift_x, shift_y = Mosaic4._compute_shifts_for_quadrant(
            position_index, in_height, in_width, out_height, out_width
        )

        bbox[:, 0] += shift_x - x_crop
        bbox[:, 2] += shift_x - x_crop

        bbox[:, 1] += shift_y - y_crop
        bbox[:, 3] += shift_y - y_crop

        return normalize_bboxes(bbox, (out_height, out_width))

    @staticmethod
    def _apply_mosaic4_to_keypoints(
        keypoints: np.ndarray,
        in_height: int,
        in_width: int,
        position_index: int,
        out_height: int,
        out_width: int,
        x_crop: int,
        y_crop: int,
    ) -> np.ndarray:
        imgs = max(out_height, out_width)
        r = imgs / max(in_height, in_width)
        if r != 1:
            in_width, in_height = (
                min(math.ceil(in_width * r), imgs),
                min(math.ceil(in_height * r), imgs),
            )
            keypoints[:, 0] = keypoints[:, 0] * r
            keypoints[:, 1] = keypoints[:, 1] * r

        shift_x, shift_y = Mosaic4._compute_shifts_for_quadrant(
            position_index, in_height, in_width, out_height, out_width
        )

        keypoints[:, 0] += shift_x - x_crop
        keypoints[:, 1] += shift_y - y_crop

        mask_invalid = (
            (keypoints[:, 0] < 0)
            | (keypoints[:, 0] > out_width)
            | (keypoints[:, 1] < 0)
            | (keypoints[:, 1] > out_height)
        )

        keypoints[:, -1] = np.where(mask_invalid, 0, keypoints[:, -1])
        return keypoints
