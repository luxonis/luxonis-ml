import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from typing_extensions import override

from luxonis_ml.data.augmentations.batch_transform import BatchTransform


class Mosaic4(BatchTransform):
    def __init__(
        self,
        out_height: int,
        out_width: int,
        value: Optional[Union[int, float, List[int], List[float]]] = None,
        mask_value: Optional[Union[int, float, List[int], List[float]]] = None,
        p: float = 0.5,
    ):
        """Mosaic augmentation arranges selected four images into single
        image in a 2x2 grid layout. This is done in deterministic way
        meaning first image in the batch will always be in top left. The
        input images should have the same number of channels but can
        have different widths and heights. The output is cropped around
        the intersection point of the four images with the size
        (out_with x out_height). If the mosaic image is smaller than
        width x height, the gap is filled by the C{fill_value}.

        @type out_height: int
        @param out_height: Output image height. The mosaic image is
            cropped by this height around the mosaic center. If the size
            of the mosaic image is smaller than this value the gap is
            filled by the C{value}.
        @type out_width: int
        @param out_width: Output image width. The mosaic image is
            cropped by this height around the mosaic center. If the size
            of the mosaic image is smaller than this value the gap is
            filled by the C{value}.
        @type value: Optional[Union[int, float, List[int], List[float]]]
        @param value: Padding value. Defaults to C{None}.
        @type mask_value: Optional[Union[int, float, List[int],
            List[float]]]
        @param mask_value: Padding value for masks. Defaults to C{None}.
        @type p: float
        @param p: Probability of applying the transform. Defaults to
            C{0.5}.
        """

        super().__init__(batch_size=4, p=p)

        if out_height <= 0:
            raise ValueError(
                f"`out_height` must be larger than 0, got {out_height}"
            )
        if out_width <= 0:
            raise ValueError(
                f"`out_width` must be larger than 0, got {out_width}"
            )

        self.out_height = out_height
        self.out_width = out_width
        self.value = value
        self.mask_value = mask_value

    def generate_random_crop_center(self) -> Tuple[int, int]:
        """Generate a random crop center within the bounds of the mosaic
        image size."""
        crop_x = random.randint(0, max(0, self.out_width))
        crop_y = random.randint(0, max(0, self.out_height))
        return crop_x, crop_y

    @override
    def apply(
        self, image_batch: List[np.ndarray], x_crop: int, y_crop: int, **_
    ) -> np.ndarray:
        """Applies the transformation to a batch of images.

        @type image_batch: List[np.ndarray]
        @param image_batch: Batch of input images to which the
            transformation is applied.
        @type x_crop: int
        @param x_crop: x-coordinate of the croping start point
        @type y_crop: int
        @param y_crop: y-coordinate of the croping start point
        @rtype: np.ndarray
        @return: Transformed images.
        """
        return apply_mosaic4_to_images(
            image_batch,
            self.out_height,
            self.out_width,
            x_crop,
            y_crop,
            self.value,
        )

    @override
    def apply_to_mask(
        self,
        mask_batch: List[np.ndarray],
        x_crop: int,
        y_crop: int,
        cols: int,
        rows: int,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of masks.

        @type mask_batch: List[np.ndarray]
        @param mask_batch: Batch of input masks to which the
            transformation is applied.
        @type x_crop: int
        @param x_crop: x-coordinate of the croping start point
        @type y_crop: int
        @param y_crop: y-coordinate of the croping start point
        @rtype: np.ndarray
        @return: Transformed masks.
        """
        for i in range(len(mask_batch)):
            mask = mask_batch[i]
            if mask.size == 0:
                if len(mask.shape) == 2:
                    mask_batch[i] = np.zeros((rows, cols), dtype=mask.dtype)
                else:
                    mask_batch[i] = np.zeros(
                        (rows, cols, mask.shape[-1]), dtype=mask.dtype
                    )
        return apply_mosaic4_to_images(
            mask_batch,
            self.out_height,
            self.out_width,
            x_crop,
            y_crop,
            self.mask_value,
        )

    @override
    def apply_to_instance_mask(
        self, masks_batch: List[np.ndarray], x_crop: int, y_crop: int, **_
    ) -> np.ndarray:
        """Applies the transformation to a batch of instance masks.

        @type mask_batch: List[np.ndarray]
        @param mask_batch: Batch of input masks to which the
            transformation is applied.
        @type x_crop: int
        @param x_crop: x-coordinate of the croping start point
        @type y_crop: int
        @param y_crop: y-coordinate of the croping start point
        @rtype: np.ndarray
        @return: Transformed masks.
        """
        return apply_mosaic4_to_instance_masks(
            masks_batch,
            self.out_height,
            self.out_width,
            x_crop,
            y_crop,
            self.value,
        )

    @override
    def apply_to_bboxes(
        self,
        bboxes_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        x_crop: int,
        y_crop: int,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of bboxes.

        @type bboxes_batch: List[np.ndarray]
        @param bboxes_batch: Batch of input bboxes to which the
            transformation is applied.
        @type indices: List[Tuple[int, int]]
        @param indices: Indices of images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @type params: Any
        @param params: Additional parameters for the transformation.
        @type x_crop: int
        @param x_crop: x-coordinate of the croping start point
        @type y_crop: int
        @param y_crop: y-coordinate of the croping start point
        @rtype: List[np.ndarray]
        @return: List of transformed bboxes.
        """
        new_bboxes = []
        for i, (bboxes, (rows, cols)) in enumerate(
            zip(bboxes_batch, image_shapes)
        ):
            if bboxes.size == 0:  # pragma: no cover
                bboxes = np.zeros((0, 6), dtype=bboxes.dtype)

            bbox = apply_mosaic4_to_bboxes(
                bboxes,
                rows,
                cols,
                i,
                self.out_height,
                self.out_width,
                x_crop,
                y_crop,
            )
            new_bboxes.append(bbox)

        return np.concatenate(new_bboxes, axis=0)

    @override
    def apply_to_keypoints(
        self,
        keypoints_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        x_crop: int,
        y_crop: int,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of keypoints.

        @type keypoints_batch: List[KeypointType]
        @param keypoints_batch: Batch of input keypoints to which the
            transformation is applied.
        @type indices: List[Tuple[int, int]]
        @param indices: Indices of images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @type params: Any
        @param params: Additional parameters for the transformation.
        @type x_crop: int
        @param x_crop: x-coordinate of the croping start point
        @type y_crop: int
        @param y_crop: y-coordinate of the croping start point
        @rtype: List[KeypointType]
        @return: List of transformed keypoints.
        """
        new_keypoints = []
        for i, (keypoints, (rows, cols)) in enumerate(
            zip(keypoints_batch, image_shapes)
        ):
            if keypoints.size == 0:
                keypoints = np.zeros((0, 5), dtype=keypoints.dtype)

            new_keypoint = apply_mosaic4_to_keypoints(
                keypoints,
                rows,
                cols,
                i,
                self.out_height,
                self.out_width,
                x_crop,
                y_crop,
            )
            new_keypoints.append(new_keypoint)
        return np.concatenate(new_keypoints, axis=0)

    @override
    def get_params_dependent_on_data(
        self, params: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get parameters dependent on the targets.

        @type params: Dict[str, Any]
        @param params: Dictionary containing parameters.
        @rtype: Dict[str, Any]
        @return: Dictionary containing parameters dependent on the
            targets.
        """
        additional_params = super().get_params_dependent_on_data(params, data)
        image_batch = data["image"]
        image_shapes = [tuple(image.shape[:2]) for image in image_batch]
        x_crop, y_crop = self.generate_random_crop_center()
        additional_params.update(
            {"image_shapes": image_shapes, "x_crop": x_crop, "y_crop": y_crop}
        )
        return additional_params


def compute_mosaic4_corners(
    quadrant: int,
    out_height: int,
    out_width: int,
    in_height: int,
    in_width: int,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
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


def apply_mosaic4_to_instance_masks(
    masks_batch: List[np.ndarray],
    out_height: int,
    out_width: int,
    x_crop: int,
    y_crop: int,
    value: Optional[Union[int, float, List[int], List[float]]] = None,
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
            imgsz = max(out_height, out_width)
            h, w = mask.shape
            r = imgsz / max(h, w)
            if r != 1:
                w, h = (
                    min(math.ceil(w * r), imgsz),
                    min(math.ceil(h * r), imgsz),
                )
                mask = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST
                )

            combined_mask = np.full(
                out_shape, value if value is not None else 0, dtype=masks.dtype
            )
            (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b) = (
                compute_mosaic4_corners(quadrant, out_height, out_width, h, w)
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


def apply_mosaic4_to_images(
    image_batch: List[np.ndarray],
    out_height: int,
    out_width: int,
    x_crop: int,
    y_crop: int,
    padding: Optional[Union[int, float, List[int], List[float]]] = None,
) -> np.ndarray:
    """Arrange the images in a 2x2 grid layout.

    The input images should have the same number of channels but can
    have different widths and heights. The gaps are filled by the
    padding value.
    """

    if len(image_batch[0].shape) == 2:
        out_shape = [out_height * 2, out_width * 2]
    else:
        out_shape = [out_height * 2, out_width * 2, image_batch[0].shape[2]]

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
            w, h = (min(math.ceil(w * r), imgsz), min(math.ceil(h * r), imgsz))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            if img.ndim == 2:
                img = img.reshape(h, w, 1)

        (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b) = compute_mosaic4_corners(
            quadrant, out_height, out_width, h, w
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


def apply_mosaic4_to_bboxes(
    bbox: np.ndarray,
    in_height: int,
    in_width: int,
    position_index: int,
    out_height: int,
    out_width: int,
    x_crop: int,
    y_crop: int,
) -> np.ndarray:
    """Adjust bounding box coordinates to account for mosaic grid
    position.

    This function modifies bounding boxes according to their placement
    in a 2x2 grid mosaic, shifting their coordinates based on the tile's
    relative position within the mosaic.
    """

    bbox = denormalize_bboxes(bbox, (in_height, in_width))

    imgsz = max(out_height, out_width)
    r = imgsz / max(in_height, in_width)
    if r != 1:
        in_width, in_height = (
            min(math.ceil(in_width * r), imgsz),
            min(math.ceil(in_height * r), imgsz),
        )
        bbox[:, :4] = bbox[:, :4] * r

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

    bbox[:, 0] += shift_x - x_crop
    bbox[:, 2] += shift_x - x_crop

    bbox[:, 1] += shift_y - y_crop
    bbox[:, 3] += shift_y - y_crop

    return normalize_bboxes(bbox, (out_height, out_width))


def apply_mosaic4_to_keypoints(
    keypoints: np.ndarray,
    in_height: int,
    in_width: int,
    position_index: int,
    out_height: int,
    out_width: int,
    x_crop: int,
    y_crop: int,
) -> np.ndarray:
    """Adjust keypoint coordinates based on mosaic grid position.

    This function adjusts the keypoint coordinates by placing them in
    one of the 2x2 mosaic grid cells, with shifts relative to the mosaic
    center.
    """
    imgsz = max(out_height, out_width)
    r = imgsz / max(in_height, in_width)
    if r != 1:
        in_width, in_height = (
            min(math.ceil(in_width * r), imgsz),
            min(math.ceil(in_height * r), imgsz),
        )
        keypoints[:, 0] = keypoints[:, 0] * r
        keypoints[:, 1] = keypoints[:, 1] * r

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
