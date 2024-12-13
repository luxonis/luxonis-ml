import random
from typing import Any, Dict, List, Optional, Tuple, Union

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
                f"out_height should be larger than 0, got {out_height}"
            )
        if out_width <= 0:
            raise ValueError(
                f"out_width should be larger than 0, got {out_width}"
            )

        self.out_height = out_height
        self.out_width = out_width
        self.value = value
        self.mask_value = mask_value

    def _generate_random_crop_center(self) -> Tuple[int, int]:
        """Generate a random crop center within the bounds of the mosaic
        image size."""
        crop_x = random.randint(0, max(0, self.out_width))
        crop_y = random.randint(0, max(0, self.out_height))
        return crop_x, crop_y

    @override
    def apply(
        self,
        image_batch: List[np.ndarray],
        x_crop: int,
        y_crop: int,
        **_,
    ) -> np.ndarray:
        """Applies the transformation to a batch of images.

        @type image_batch: List[np.ndarray]
        @param image_batch: Batch of input images to which the
            transformation is applied.
        @type indices: List[Tuple[int, int]]
        @param indices: Indices of images in the batch.
        @type params: Any
        @param params: Additional parameters for the transformation.
        @type x_crop: int
        @param x_crop: x-coordinate of the croping start point
        @type y_crop: int
        @param y_crop: y-coordinate of the croping start point
        @rtype: List[np.ndarray]
        @return: List of transformed images.
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
        @type indices: List[Tuple[int, int]]
        @param indices: Indices of images in the batch.
        @type params: Any
        @param params: Additional parameters for the transformation.
        @type x_crop: int
        @param x_crop: x-coordinate of the croping start point
        @type y_crop: int
        @param y_crop: y-coordinate of the croping start point
        @rtype: List[np.ndarray]
        @return: List of transformed masks.
        """
        for i in range(len(mask_batch)):
            if mask_batch[i].size == 0:
                mask_batch[i] = np.zeros(
                    (rows, cols), dtype=mask_batch[i].dtype
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
        x_crop, y_crop = self._generate_random_crop_center()
        additional_params.update(
            {
                "image_shapes": image_shapes,
                "x_crop": x_crop,
                "y_crop": y_crop,
            }
        )
        return additional_params


def apply_mosaic4_to_instance_masks(
    masks_batch: List[np.ndarray],
    height: int,
    width: int,
    x_crop: int,
    y_crop: int,
    value: Optional[Union[int, float, List[int], List[float]]] = None,
) -> np.ndarray:
    out_masks = []
    dtype = masks_batch[0].dtype
    out_shape = [height * 2, width * 2]

    for i, masks in enumerate(masks_batch):
        if masks.size == 0:
            continue

        for j in range(masks.shape[-1]):
            mask = masks[..., j]
            mask4 = np.full(
                out_shape,
                value if value is not None else 0,
                dtype=dtype,
            )

            h, w = mask.shape[:2]

            if i == 0:  # top left
                x1a, y1a, x2a, y2a = (
                    max(width - w, 0),
                    max(height - h, 0),
                    width,
                    height,
                )
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = (
                    width,
                    max(height - h, 0),
                    min(width + w, width * 2),
                    height,
                )
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = (
                    max(width - w, 0),
                    height,
                    width,
                    min(height * 2, height + h),
                )
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = (
                    width,
                    height,
                    min(width + w, width * 2),
                    min(height * 2, height + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4_region = mask4[y1a:y2a, x1a:x2a]
            img_region = mask[y1b:y2b, x1b:x2b]

            img4_h, img4_w = img4_region.shape[:2]
            img_h, img_w = img_region.shape[:2]

            min_h = min(img4_h, img_h)
            min_w = min(img4_w, img_w)

            mask4[y1a : y1a + min_h, x1a : x1a + min_w] = mask[
                y1b : y1b + min_h, x1b : x1b + min_w
            ]

            mask4 = mask4[y_crop : y_crop + height, x_crop : x_crop + width]
            out_masks.append(mask4)

    return np.stack(out_masks, axis=-1)


def apply_mosaic4_to_images(
    image_batch: List[np.ndarray],
    height: int,
    width: int,
    x_crop: int,
    y_crop: int,
    value: Optional[Union[int, float, List[int], List[float]]] = None,
) -> np.ndarray:
    """Arrange the images in a 2x2 grid layout. The input images should
    have the same number of channels but can have different widths and
    heights. The gaps are filled by the value.

    @type image_batch: List[np.ndarray]
    @param image_batch: Image list. The length should be four. Each
        image can has different size.
    @type height: int
    @param height: Height of output mosaic image
    @type width: int
    @param width: Width of output mosaic image
    @type value: Optional[int]
    @param value: Padding value
    @type x_crop: int
    @param x_crop: x-coordinate of the croping start point
    @type y_crop: int
    @param y_crop: y-coordinate of the croping start point
    @rtype: np.ndarray
    @return: Final output image
    """
    if len(image_batch) != 4:
        raise ValueError(
            f"Length of image_batch should be 4. Got {len(image_batch)}"
        )

    for i in range(3):
        if image_batch[0].shape[2:] != image_batch[i + 1].shape[2:]:
            raise ValueError(
                "All images should have the same number of channels."
            )

    if len(image_batch[0].shape) == 2:
        out_shape = [height * 2, width * 2]
    else:
        out_shape = [height * 2, width * 2, image_batch[0].shape[2]]

    dtype = image_batch[0].dtype

    img4 = np.full(
        out_shape,
        value if value is not None else 0,
        dtype=dtype,
    )

    for i, img in enumerate(image_batch):
        (h, w) = img.shape[:2]

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = (
                max(width - w, 0),
                max(height - h, 0),
                width,
                height,
            )
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = (
                width,
                max(height - h, 0),
                min(width + w, width * 2),
                height,
            )
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = (
                max(width - w, 0),
                height,
                width,
                min(height * 2, height + h),
            )
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom right
            x1a, y1a, x2a, y2a = (
                width,
                height,
                min(width + w, width * 2),
                min(height * 2, height + h),
            )
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4_region = img4[y1a:y2a, x1a:x2a]
        img_region = img[y1b:y2b, x1b:x2b]

        img4_h, img4_w = img4_region.shape[:2]
        img_h, img_w = img_region.shape[:2]

        min_h = min(img4_h, img_h)
        min_w = min(img4_w, img_w)

        img4[y1a : y1a + min_h, x1a : x1a + min_w] = img[
            y1b : y1b + min_h, x1b : x1b + min_w
        ]

    img4 = img4[y_crop : y_crop + height, x_crop : x_crop + width]

    return img4


def apply_mosaic4_to_bboxes(
    bbox: np.ndarray,
    rows: int,
    cols: int,
    position_index: int,
    height: int,
    width: int,
    x_crop: int,
    y_crop: int,
) -> np.ndarray:
    """Adjust bounding box coordinates to account for mosaic grid
    position.

    This function modifies bounding boxes according to their placement
    in a 2x2 grid mosaic, shifting their coordinates based on the tile's
    relative position within the mosaic.

    @type bbox: np.ndarray
    @param bbox: Bounding box coordinates to be transformed.
    @type rows: int
    @param rows: Height of the original image.
    @type cols: int
    @param cols: Width of the original image.
    @type position_index: int
    @param position_index: Position of the image in the 2x2 grid. (0 =
        top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right).
    @type height: int
    @param height: Height of the final output mosaic image.
    @type width: int
    @param width: Width of the final output mosaic image.
    @type x_crop: int
    @param x_crop: x-coordinate of the croping start point
    @type y_crop: int
    @param y_crop: y-coordinate of the croping start point
    @rtype: np.ndarray
    @return: Transformed bounding box coordinates.
    """

    # TODO: remove normalization
    bbox = denormalize_bboxes(bbox, (rows, cols))

    if position_index == 0:
        shift_x = width - cols
        shift_y = height - rows
    elif position_index == 1:
        shift_x = width
        shift_y = height - rows
    elif position_index == 2:
        shift_x = width - cols
        shift_y = height
    elif position_index == 3:
        shift_x = width
        shift_y = height

    bbox[:, 0] += shift_x - x_crop
    bbox[:, 1] += shift_y - y_crop
    bbox[:, 2] += shift_x - x_crop
    bbox[:, 3] += shift_y - y_crop

    bbox = normalize_bboxes(bbox, (height, width))

    return bbox


def apply_mosaic4_to_keypoints(
    keypoints: np.ndarray,
    rows: int,
    cols: int,
    position_index: int,
    height: int,
    width: int,
    x_crop: int,
    y_crop: int,
) -> np.ndarray:
    """Adjust keypoint coordinates based on mosaic grid position.

    This function adjusts the keypoint coordinates by placing them in
    one of the 2x2 mosaic grid cells, with shifts relative to the mosaic
    center.

    @type keypoints: np.ndarray
    @param keypoint: Keypoint coordinates and attributes (x, y).
    @type rows: int
    @param rows: Height of the original image.
    @type cols: int
    @param cols: Width of the original image.
    @type position_index: int
    @param position_index: Position of the image in the 2x2 grid. (0 =
        top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right).
    @type height: int
    @param height: Height of the final output mosaic image.
    @type width: int
    @param width: Width of the final output mosaic image.
    @type x_crop: int
    @param x_crop: x-coordinate of the croping start point
    @type y_crop: int
    @param y_crop: y-coordinate of the croping start point
    @rtype: np.ndarray
    @return: Adjusted keypoint coordinates.
    """
    if position_index == 0:
        shift_x = width - cols
        shift_y = height - rows
    elif position_index == 1:
        shift_x = width
        shift_y = height - rows
    elif position_index == 2:
        shift_x = width - cols
        shift_y = height
    elif position_index == 3:
        shift_x = width
        shift_y = height

    keypoints[:, 0] += shift_x - x_crop
    keypoints[:, 1] += shift_y - y_crop

    return keypoints
