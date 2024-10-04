import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from albumentations import BoxType, KeypointType
from albumentations.core.bbox_utils import (
    denormalize_bbox,
    normalize_bbox,
)
from albumentations.core.transforms_interface import (
    BoxInternalType,
    KeypointInternalType,
)

from ..batch_transform import BatchBasedTransform
from ..utils import AUGMENTATIONS


@AUGMENTATIONS.register_module()
class Mosaic4(BatchBasedTransform):
    def __init__(
        self,
        out_height: int,
        out_width: int,
        value: Optional[Union[int, float, List[int], List[float]]] = None,
        out_batch_size: int = 1,
        mask_value: Optional[Union[int, float, List[int], List[float]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        """Mosaic augmentation arranges selected four images into single
        image in a 2x2 grid layout. This is done in deterministic way
        meaning first image in the batch will always be in top left. The
        input images should have the same number of channels but can
        have different widths and heights. The output is cropped around
        the intersection point of the four images with the size
        (out_with x out_height). If the mosaic image is smaller than
        width x height, the gap is filled by the fill_value.

        @param out_height: Output image height. The mosaic image is cropped by this height around the mosaic center.
        If the size of the mosaic image is smaller than this value the gap is filled by the `value`.
        @type out_height: int

        @param out_width: Output image width. The mosaic image is cropped by this height around the mosaic center.
        If the size of the mosaic image is smaller than this value the gap is filled by the `value`.
        @type out_width: int

        @param value: Padding value. Defaults to None.
        @type value: Optional[Union[int, float, List[int], List[float]]], optional

        @param out_batch_size: Number of output images in the batch. Defaults to 1.
        @type out_batch_size: int, optional

        @param mask_value: Padding value for masks. Defaults to None.
        @type mask_value: Optional[Union[int, float, List[int], List[float]]], optional

        @param always_apply: Whether to always apply the transform. Defaults to False.
        @type always_apply: bool, optional

        @param p: Probability of applying the transform. Defaults to 0.5.
        @type p: float, optional
        """

        super().__init__(batch_size=4, always_apply=always_apply, p=p)

        if out_height <= 0:
            raise ValueError(
                f"out_height should be larger than 0, got {out_height}"
            )
        if out_width <= 0:
            raise ValueError(
                f"out_width should be larger than 0, got {out_width}"
            )
        if out_batch_size <= 0:
            raise ValueError(
                f"out_batch_size should be larger than 0, got {out_batch_size}"
            )

        self.n_tiles = self.batch_size  # 4: 2x2
        self.out_height = out_height
        self.out_width = out_width
        self.value = value
        self.mask_value = mask_value
        self.out_batch_size = out_batch_size

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Gets the default arguments for the mixup augmentation.

        @return: The string keywords of the arguments.
        @rtype: Tuple[str, ...]
        """
        return (
            "out_height",
            "out_width",
            "replace",
            "value",
            "out_batch_size",
            "mask_value",
        )

    def _generate_random_crop_center(self) -> Tuple[int, int]:
        """Generate a random crop center within the bounds of the mosaic
        image size."""
        crop_x = random.randint(0, max(0, self.out_width))
        crop_y = random.randint(0, max(0, self.out_height))
        return crop_x, crop_y

    @property
    def targets_as_params(self):
        """List of augmentation targets.

        @return: Output list of augmentation targets.
        @rtype: List[str]
        """
        return ["image_batch"]

    def apply_to_image_batch(
        self,
        image_batch: List[np.ndarray],
        indices: List[int],
        x_crop: int,
        y_crop: int,
        **params,
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of images.

        @param image_batch: Batch of input images to which the
            transformation is applied.
        @type image_batch: List[np.ndarray]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @param x_crop: x-coordinate of the croping start point
        @type x_crop: int
        @param y_crop: y-coordinate of the croping start point
        @type y_crop: int
        @return: List of transformed images.
        @rtype: List[np.ndarray]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[
                self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)
            ]
            image_chunk = [image_batch[i] for i in idx_chunk]
            mosaiced = mosaic4(
                image_chunk,
                self.out_height,
                self.out_width,
                x_crop,
                y_crop,
                self.value,
            )
            output_batch.append(mosaiced)
        return output_batch

    def apply_to_mask_batch(
        self,
        mask_batch: List[np.ndarray],
        indices: List[int],
        x_crop: int,
        y_crop: int,
        **params,
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of masks.

        @param mask_batch: Batch of input masks to which the
            transformation is applied.
        @type mask_batch: List[np.ndarray]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @param x_crop: x-coordinate of the croping start point
        @type x_crop: int
        @param y_crop: y-coordinate of the croping start point
        @type y_crop: int
        @return: List of transformed masks.
        @rtype: List[np.ndarray]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[
                self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)
            ]
            mask_chunk = [mask_batch[i] for i in idx_chunk]
            mosaiced = mosaic4(
                mask_chunk,
                self.out_height,
                self.out_width,
                x_crop,
                y_crop,
                self.mask_value,
            )
            output_batch.append(mosaiced)
        return output_batch

    def apply_to_bboxes_batch(
        self,
        bboxes_batch: List[BoxType],
        indices: List[int],
        image_shapes: List[Tuple[int, int]],
        x_crop: int,
        y_crop: int,
        **params,
    ) -> List[BoxType]:
        """Applies the transformation to a batch of bboxes.

        @param bboxes_batch: Batch of input bboxes to which the
            transformation is applied.
        @type bboxes_batch: List[BboxType]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @param x_crop: x-coordinate of the croping start point
        @type x_crop: int
        @param y_crop: y-coordinate of the croping start point
        @type y_crop: int
        @return: List of transformed bboxes.
        @rtype: List[BoxType]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[
                self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)
            ]
            bboxes_chunk = [bboxes_batch[i] for i in idx_chunk]
            shape_chunk = [image_shapes[i] for i in idx_chunk]
            new_bboxes = []
            for i in range(self.n_tiles):
                bboxes = bboxes_chunk[i]
                rows, cols = shape_chunk[i]
                for bbox in bboxes:
                    new_bbox = bbox_mosaic4(
                        bbox[:4],
                        rows,
                        cols,
                        i,
                        self.out_height,
                        self.out_width,
                        x_crop,
                        y_crop,
                    )
                    new_bboxes.append(tuple(new_bbox) + tuple(bbox[4:]))
            output_batch.append(new_bboxes)
        return output_batch

    def apply_to_keypoints_batch(
        self,
        keyboints_batch: List[KeypointType],
        indices: List[int],
        image_shapes: List[Tuple[int, int]],
        x_crop: int,
        y_crop: int,
        **params,
    ) -> List[KeypointType]:
        """Applies the transformation to a batch of keypoints.

        @param keypoints_batch: Batch of input keypoints to which the
            transformation is applied.
        @type keypoints_batch: List[KeypointType]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @param x_crop: x-coordinate of the croping start point
        @type x_crop: int
        @param y_crop: y-coordinate of the croping start point
        @type y_crop: int
        @return: List of transformed keypoints.
        @rtype: List[KeypointType]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[
                self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)
            ]
            keypoints_chunk = [keyboints_batch[i] for i in idx_chunk]
            shape_chunk = [image_shapes[i] for i in idx_chunk]
            new_keypoints = []
            for i in range(self.n_tiles):
                keypoints = keypoints_chunk[i]
                rows, cols = shape_chunk[i]
                for keypoint in keypoints:
                    new_keypoint = keypoint_mosaic4(
                        keypoint[:4],
                        rows,
                        cols,
                        i,
                        self.out_height,
                        self.out_width,
                        x_crop,
                        y_crop,
                    )
                    new_keypoints.append(new_keypoint + tuple(keypoint[4:]))
            output_batch.append(new_keypoints)
        return output_batch

    def get_params_dependent_on_targets(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get parameters dependent on the targets.

        @param params: Dictionary containing parameters.
        @type params: Dict[str, Any]
        @return: Dictionary containing parameters dependent on the
            targets.
        @rtype: Dict[str, Any]
        """
        image_batch = params["image_batch"]
        n = len(image_batch)
        if self.n_tiles * self.out_batch_size > n:
            raise ValueError(
                f"The batch size (= {n}) should be larger than "
                + f"{self.n_tiles} x out_batch_size (= {self.n_tiles * self.out_batch_size})"
            )
        indices = [0, 1, 2, 3]
        image_shapes = [tuple(image.shape[:2]) for image in image_batch]
        x_crop, y_crop = self._generate_random_crop_center()
        return {
            "indices": indices,
            "image_shapes": image_shapes,
            "x_crop": x_crop,
            "y_crop": y_crop,
        }


def mosaic4(
    image_batch: List[np.ndarray],
    height: int,
    width: int,
    x_crop: int,
    y_crop: int,
    value: Optional[int] = None,
) -> np.ndarray:
    """Arrange the images in a 2x2 grid layout. The input images should
    have the same number of channels but can have different widths and
    heights. The gaps are filled by the value.

    @param image_batch: Image list. The length should be four. Each
        image can has different size.
    @type image_batch: List[np.ndarray]
    @param height: Height of output mosaic image
    @type height: int
    @param width: Width of output mosaic image
    @type width: int
    @param value: Padding value
    @type value: Optional[int]
    @param x_crop: x-coordinate of the croping start point
    @type x_crop: int
    @param y_crop: y-coordinate of the croping start point
    @type y_crop: int
    @return: Final output image
    @rtype: np.ndarray
    """
    N_TILES = 4
    if len(image_batch) != N_TILES:
        raise ValueError(
            f"Length of image_batch should be 4. Got {len(image_batch)}"
        )

    for i in range(N_TILES - 1):
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
        elif i == 3:  # bottom right
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


def bbox_mosaic4(
    bbox: BoxInternalType,
    rows: int,
    cols: int,
    position_index: int,
    height: int,
    width: int,
    x_crop: int,
    y_crop: int,
) -> BoxInternalType:
    """Adjust bounding box coordinates to account for mosaic grid
    position.

    This function modifies bounding boxes according to their placement
    in a 2x2 grid mosaic, shifting their coordinates based on the tile's
    relative position within the mosaic.

    @param bbox: Bounding box coordinates to be transformed.
    @type bbox: BoxInternalType
    @param rows: Height of the original image.
    @type rows: int
    @param cols: Width of the original image.
    @type cols: int
    @param position_index: Position of the image in the 2x2 grid. (0 =
        top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right).
    @type position_index: int
    @param height: Height of the final output mosaic image.
    @type height: int
    @param width: Width of the final output mosaic image.
    @type width: int
    @param x_crop: x-coordinate of the croping start point
    @type x_crop: int
    @param y_crop: y-coordinate of the croping start point
    @type y_crop: int
    @return: Transformed bounding box coordinates.
    @rtype: BoxInternalType
    """

    bbox = denormalize_bbox(bbox, rows, cols)

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

    bbox = (
        bbox[0] + shift_x - x_crop,
        bbox[1] + shift_y - y_crop,
        bbox[2] + shift_x - x_crop,
        bbox[3] + shift_y - y_crop,
    )

    bbox = normalize_bbox(bbox, height, width)

    return bbox


def keypoint_mosaic4(
    keypoint: KeypointInternalType,
    rows: int,
    cols: int,
    position_index: int,
    height: int,
    width: int,
    x_crop: int,
    y_crop: int,
) -> KeypointInternalType:
    """Adjust keypoint coordinates based on mosaic grid position.

    This function adjusts the keypoint coordinates by placing them in
    one of the 2x2 mosaic grid cells, with shifts relative to the mosaic
    center.

    @param keypoint: Keypoint coordinates and attributes (x, y, angle,
        scale).
    @type keypoint: KeypointInternalType
    @param rows: Height of the original image.
    @type rows: int
    @param cols: Width of the original image.
    @type cols: int
    @param position_index: Position of the image in the 2x2 grid. (0 =
        top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right).
    @type position_index: int
    @param height: Height of the final output mosaic image.
    @type height: int
    @param width: Width of the final output mosaic image.
    @type width: int
    @param x_crop: x-coordinate of the croping start point
    @type x_crop: int
    @param y_crop: y-coordinate of the croping start point
    @type y_crop: int
    @return: Adjusted keypoint coordinates.
    @rtype: KeypointInternalType
    """
    x, y, angle, scale = keypoint

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

    return x + shift_x - x_crop, y + shift_y - y_crop, angle, scale
