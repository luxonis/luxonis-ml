import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from albumentations.core.transforms_interface import (
    BoxInternalType,
    ImageColorType,
    KeypointInternalType,
)
from albumentations.core.bbox_utils import (
    denormalize_bbox,
    normalize_bbox,
)
from .batch_transform import BatchBasedTransform


class Mosaic4(BatchBasedTransform):
    """Mosaic augmentation arranges randomly selected four images into single image in a 2x2 grid layout.
    The input images should have the same number of channels but can have different widths and heights.
    The output is cropped around the intersection point of the four images with the size (out_with x out_height).
    If the mosaic image is smaller than with x height, the gap is filled by the fill_value.
    Args:
        out_height (int)): output image height.
            The mosaic image is cropped by this height around the mosaic center.
            If the size of the mosaic image is smaller than this value the gap is filled by the `value`.
        out_width (int): output image width.
            The mosaic image is cropped by this height around the mosaic center.
            If the size of the mosaic image is smaller than this value the gap is filled by the `value`.
        value (int, float, list of ints, list of float): padding value. Default 0 (None).
        replace (bool): whether to allow replacement in sampling or not. When the value is `True`, the same image
            can be selected multiple times. When False, the batch size of the input should be at least four.
        out_batch_size(int): output batch size. If the replace = False,
            the input batch size should be 4 * out_batch_size.
        mask_value (int, float, list of ints, list of float): padding value for masks. Default 0 (None).
    Targets:
        image_batch, mask_batch, bboxes_batch
    [Bochkovskiy] Bochkovskiy A, Wang CY, Liao HYM. (2020) "YOLOv 4 : Optimal speed and accuracy of object detection.",
    https://arxiv.org/pdf/2004.10934.pdf
    """

    def __init__(
        self,
        out_height,
        out_width,
        value=None,
        replace=True,
        out_batch_size=1,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        if out_height <= 0:
            raise ValueError(f"out_height should be larger than 0, got {out_height}")
        if out_width <= 0:
            raise ValueError(f"out_width should be larger than 0, got {out_width}")
        if out_batch_size <= 0:
            raise ValueError(
                f"out_batch_size should be larger than 0, got {out_batch_size}"
            )

        self.n_tiles = 4  # 2x2
        self.out_height = out_height
        self.out_width = out_width
        self.replace = replace
        self.value = value
        self.mask_value = mask_value
        self.out_batch_size = out_batch_size

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
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
        return ["image_batch"]

    def apply_to_image_batch(self, image_batch, indices, **params):
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)]
            image_chunk = [image_batch[i] for i in idx_chunk]
            mosaiced = mosaic4(image_chunk, self.out_height, self.out_width, self.value)
            output_batch.append(mosaiced)
        return output_batch

    def apply_to_mask_batch(self, mask_batch, indices, **params):
        output_batch = []
        for i_batch in range(self.out_batch_size):
            idx_chunk = indices[self.n_tiles * i_batch : self.n_tiles * (i_batch + 1)]
            mask_chunk = [mask_batch[i] for i in idx_chunk]
            mosaiced = mosaic4(
                mask_chunk, self.out_height, self.out_width, self.mask_value
            )
            output_batch.append(mosaiced)
        return output_batch

    def apply_to_bboxes_batch(self, bboxes_batch, indices, image_shapes, **params):
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
        self, keyboints_batch, indices, image_shapes, **params
    ):
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
        image_batch = params["image_batch"]
        n = len(image_batch)
        if not self.replace and self.n_tiles * self.out_batch_size > n:
            raise ValueError(
                f"If replace == False, the batch size (= {n}) should be larger than "
                + f"{self.n_tiles} x out_batch_size (= {self.n_tiles * self.out_batch_size})"
            )
        indices = np.random.choice(
            range(n), size=self.n_tiles * self.out_batch_size, replace=self.replace
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
    Args:
        image_batch (List[np.ndarray]): image list. The length should be four. Each image can has different size.
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
        value (int, float, list of ints, list of float): padding value
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
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Height of input image that corresponds to one of the mosaic cells
        cols (int): Width of input image that corresponds to one of the mosaic cells
        position_index (int): Index of the mosaic cell. 0: top left, 1: top right, 2: bottom left, 3: bottom right
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
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
    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        rows (int): Height of input image that corresponds to one of the mosaic cells
        cols (int): Width of input image that corresponds to one of the mosaic cells
        position_index (int): Index of the mosaic cell. 0: top left, 1: top right, 2: bottom left, 3: bottom right
        height (int): Height of output mosaic image
        width (int): Width of output mosaic image
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
