from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from albumentations import BoxType, KeypointType
from albumentations.core.bbox_utils import (
    denormalize_bbox,
    normalize_bbox,
)
from albumentations.core.transforms_interface import (
    BoxInternalType,
    ImageColorType,
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
        """Mosaic augmentation arranges selected four images into single image in a 2x2
        grid layout. This is done in deterministic way meaning first image in the batch
        will always be in top left. The input images should have the same number of
        channels but can have different widths and heights. The output is cropped around
        the intersection point of the four images with the size (out_with x out_height).
        If the mosaic image is smaller than with x height, the gap is filled by the
        fill_value.

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
            raise ValueError(f"out_height should be larger than 0, got {out_height}")
        if out_width <= 0:
            raise ValueError(f"out_width should be larger than 0, got {out_width}")
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

    @property
    def targets_as_params(self):
        """List of augmentation targets.

        @return: Output list of augmentation targets.
        @rtype: List[str]
        """
        return ["image_batch"]

    def apply_to_image_batch(
        self, image_batch: List[np.ndarray], indices: List[int], **params
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of images.

        @param image_batch: Batch of input images to which the transformation is
            applied.
        @type image_batch: List[np.ndarray]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed images.
        @rtype: List[np.ndarray]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)]
            image_chunk = [image_batch[i] for i in idx_chunk]
            mosaiced = mosaic4(image_chunk, self.out_height, self.out_width, self.value)
            output_batch.append(mosaiced)
        return output_batch

    def apply_to_mask_batch(
        self, mask_batch: List[np.ndarray], indices: List[int], **params
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of masks.

        @param mask_batch: Batch of input masks to which the transformation is applied.
        @type mask_batch: List[np.ndarray]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed masks.
        @rtype: List[np.ndarray]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)]
            mask_chunk = [mask_batch[i] for i in idx_chunk]
            mosaiced = mosaic4(
                mask_chunk, self.out_height, self.out_width, self.mask_value
            )
            output_batch.append(mosaiced)
        return output_batch

    def apply_to_bboxes_batch(
        self,
        bboxes_batch: List[BoxType],
        indices: List[int],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[BoxType]:
        """Applies the transformation to a batch of bboxes.

        @param bboxes_batch: Batch of input bboxes to which the transformation is
            applied.
        @type bboxes_batch: List[BboxType]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed bboxes.
        @rtype: List[BoxType]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)]
            bboxes_chunk = [bboxes_batch[i] for i in idx_chunk]
            shape_chunk = [image_shapes[i] for i in idx_chunk]
            new_bboxes = []
            for i in range(self.n_tiles):
                bboxes = bboxes_chunk[i]
                rows, cols = shape_chunk[i]
                for bbox in bboxes:
                    new_bbox = bbox_mosaic4(
                        bbox[:4], rows, cols, i, self.out_height, self.out_width
                    )
                    new_bboxes.append(tuple(new_bbox) + tuple(bbox[4:]))
            output_batch.append(new_bboxes)
        return output_batch

    def apply_to_keypoints_batch(
        self,
        keyboints_batch: List[KeypointType],
        indices: List[int],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[KeypointType]:
        """Applies the transformation to a batch of keypoints.

        @param keypoints_batch: Batch of input keypoints to which the transformation is
            applied.
        @type keypoints_batch: List[KeypointType]
        @param indices: Indices of images in the batch.
        @type indices: List[Tuple[int, int]]
        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]
        @param params: Additional parameters for the transformation.
        @type params: Any
        @return: List of transformed keypoints.
        @rtype: List[KeypointType]
        """
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)]
            keypoints_chunk = [keyboints_batch[i] for i in idx_chunk]
            shape_chunk = [image_shapes[i] for i in idx_chunk]
            new_keypoints = []
            for i in range(self.n_tiles):
                keypoints = keypoints_chunk[i]
                rows, cols = shape_chunk[i]
                for keypoint in keypoints:
                    new_keypoint = keypoint_mosaic4(
                        keypoint[:4], rows, cols, i, self.out_height, self.out_width
                    )
                    new_keypoints.append(new_keypoint + tuple(keypoint[4:]))
            output_batch.append(new_keypoints)
        return output_batch

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters dependent on the targets.

        @param params: Dictionary containing parameters.
        @type params: Dict[str, Any]
        @return: Dictionary containing parameters dependent on the targets.
        @rtype: Dict[str, Any]
        """
        image_batch = params["image_batch"]
        n = len(image_batch)
        if self.n_tiles * self.out_batch_size > n:
            raise ValueError(
                f"The batch size (= {n}) should be larger than "
                + f"{self.n_tiles} x out_batch_size (= {self.n_tiles * self.out_batch_size})"
            )
        indices = np.random.choice(
            range(n), size=self.n_tiles * self.out_batch_size, replace=False
        ).tolist()
        image_shapes = [tuple(image.shape[:2]) for image in image_batch]
        return {
            "indices": indices,
            "image_shapes": image_shapes,
        }


def mosaic4(
    image_batch: List[np.ndarray],
    height: int,
    width: int,
    value: Optional[ImageColorType] = None,
) -> np.ndarray:
    """Arrange the images in a 2x2 grid layout.
    The input images should have the same number of channels but can have different widths and heights.
    The output is cropped around the intersection point of the four images with the size (with x height).
    If the mosaic image is smaller than with x height, the gap is filled by the fill_value.
    This implementation is based on YOLOv5 with some modification:
    https://github.com/ultralytics/yolov5/blob/932dc78496ca532a41780335468589ad7f0147f7/utils/datasets.py#L648

    @param image_batch: Image list. The length should be four. Each image can has different size.
    @type image_batch: List[np.ndarray]
    @param height: Height of output mosaic image
    @type height: int
    @param width: Width of output mosaic image
    @type width: int
    @param value: Padding value
    @type value: Optional[ImageColorType]
    @return: Final output image
    @rtype: np.ndarray
    """
    N_TILES = 4
    if len(image_batch) != N_TILES:
        raise ValueError(f"Length of image_batch should be 4. Got {len(image_batch)}")

    for i in range(N_TILES - 1):
        if image_batch[0].shape[2:] != image_batch[i + 1].shape[2:]:
            raise ValueError(
                "All images should have the same number of channels."
                + f" Got the shapes {image_batch[0].shape} and {image_batch[i + 1].shape}"
            )

        if image_batch[0].dtype != image_batch[i + 1].dtype:
            raise ValueError(
                "All images should have the same dtype."
                + f" Got the dtypes {image_batch[0].dtype} and {image_batch[i + 1].dtype}"
            )

    if len(image_batch[0].shape) == 2:
        out_shape = [height, width]
    else:
        out_shape = [height, width, image_batch[0].shape[2]]

    dtype = image_batch[0].dtype
    img4 = np.zeros(out_shape, dtype=dtype)  # base image with 4 tiles

    value = 0 if value is None else value
    if isinstance(value, (tuple, list, np.ndarray)):
        if out_shape[2] != len(value):
            ValueError(
                "value parameter should has the same lengh as the output channel."
                + f" value: ({value}), output shape: {out_shape}"
            )
        for i in range(len(value)):
            img4[:, :, i] = value[i]
    else:
        img4[:] = value

    center_x = width // 2
    center_y = height // 2
    for i, img in enumerate(image_batch):
        (h, w) = img.shape[:2]

        # place img in img4
        # this based on the yolo5's implementation
        #
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = (
                max(center_x - w, 0),
                max(center_y - h, 0),
                center_x,
                center_y,
            )  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = (
                w - (x2a - x1a),
                h - (y2a - y1a),
                w,
                h,
            )  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = (
                center_x,
                max(center_y - h, 0),
                min(center_x + w, width),
                center_y,
            )
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = (
                max(center_x - w, 0),
                center_y,
                center_x,
                min(height, center_y + h),
            )
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = (
                center_x,
                center_y,
                min(center_x + w, width),
                min(height, center_y + h),
            )
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

    return img4


def bbox_mosaic4(
    bbox: BoxInternalType,
    rows: int,
    cols: int,
    position_index: int,
    height: int,
    width: int,
) -> BoxInternalType:
    """Put the given bbox in one of the cells of the 2x2 grid.

    @param bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
    @type bbox: BoxInternalType
    @param rows: Height of input image that corresponds to one of the mosaic cells
    @type rows: int
    @param cols: Width of input image that corresponds to one of the mosaic cells
    @type cols: int
    @param position_index: Index of the mosaic cell. 0: top left, 1: top right, 2:
        bottom left, 3: bottom right
    @type position_index: int
    @param height: Height of output mosaic image
    @type height: int
    @param width: Width of output mosaic image
    @type width: int
    @return: Transformed bbox
    @rtype: BoxInternalType
    """
    bbox = denormalize_bbox(bbox, rows, cols)
    center_x = width // 2
    center_y = height // 2
    if position_index == 0:  # top left
        shift_x = center_x - cols
        shift_y = center_y - rows
    elif position_index == 1:  # top right
        shift_x = center_x
        shift_y = center_y - rows
    elif position_index == 2:  # bottom left
        shift_x = center_x - cols
        shift_y = center_y
    elif position_index == 3:  # bottom right
        shift_x = center_x
        shift_y = center_y
    bbox = (
        bbox[0] + shift_x,
        bbox[1] + shift_y,
        bbox[2] + shift_x,
        bbox[3] + shift_y,
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
) -> KeypointInternalType:
    """Put the given bbox in one of the cells of the 2x2 grid.

    @param keypoint: A keypoint `(x, y, angle, scale)`.
    @type bbox: KeypointInternalType
    @param rows: Height of input image that corresponds to one of the mosaic cells
    @type rows: int
    @param cols: Width of input image that corresponds to one of the mosaic cells
    @type cols: int
    @param position_index: Index of the mosaic cell. 0: top left, 1: top right, 2:
        bottom left, 3: bottom right
    @type position_index: int
    @param height: Height of output mosaic image
    @type height: int
    @param width: Width of output mosaic image
    @type width: int
    @return: Transformed keypoint
    @rtype: KeypointInternalType
    """
    x, y, angle, scale = keypoint

    center_x = width // 2
    center_y = height // 2
    if position_index == 0:  # top left
        shift_x = center_x - cols
        shift_y = center_y - rows
    elif position_index == 1:  # top right
        shift_x = center_x
        shift_y = center_y - rows
    elif position_index == 2:  # bottom left
        shift_x = center_x - cols
        shift_y = center_y
    elif position_index == 3:  # bottom right
        shift_x = center_x
        shift_y = center_y
    return x + shift_x, y + shift_y, angle, scale
