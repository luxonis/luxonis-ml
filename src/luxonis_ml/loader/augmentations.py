import numpy as np
import cv2
import warnings
import albumentations as A
from albumentations import *
from typing import Any, Dict, List, Tuple
from .loader import LabelType


class Augmentations:
    def __init__(self, train_rgb: bool = True):
        """Base class for augmentations that are used in LuxonisLoader

        Args:
            train_rgb (bool, optional): Weather should use RGB or BGR images. Defaults to True.
        """

        self.train_rgb = train_rgb

        self.is_batched = False
        self.aug_batch_size = 1

    def _parse_cfg(
        self,
        image_size: List[int],
        augmentations: List[Dict[str, Any]],
        keep_aspect_ratio: bool = True,
    ) -> Tuple[A.BatchCompose, A.Compose]:
        """Parses provided config and returns Albumentations BatchedCompose object and Compose object for default transforms

        Args:
            image_size (List[int]): Desired image size [H,W]
            augmentations (List[Dict[str, Any]]): List of augmentations to use and their params
            keep_aspect_ratio (bool, optional): Weather should use resize that keeps aspect ratio of original image. Defaults to True.

        Returns:
            Tuple[A.BatchCompose, A.Compose]: Objects for batched and spatial transforms
        """

        image_size = image_size

        # Always perform Resize
        if keep_aspect_ratio:
            resize = LetterboxResize(height=image_size[0], width=image_size[1])
        else:
            resize = A.Resize(image_size[0], image_size[1])

        pixel_augs = []
        spatial_augs = []
        batched_augs = []
        if augmentations:
            for aug in augmentations:
                # use our implementaiton
                if aug["name"] == "Mosaic4":
                    aug["name"] == "DeterministicMosaic4"
                    params = aug.get("params", {})
                    params["replace"] = False
                    warnings.warn(
                        "Our Mosaic4 implementation doesn't support replace, setting it to False"
                    )

                curr_aug = eval(aug["name"])(**aug.get("params", {}))
                if isinstance(curr_aug, A.ImageOnlyTransform):
                    pixel_augs.append(curr_aug)
                elif isinstance(curr_aug, A.DualTransform):
                    spatial_augs.append(curr_aug)
                elif isinstance(curr_aug, A.BatchBasedTransform):
                    self.is_batched = True
                    self.aug_batch_size = max(self.aug_batch_size, curr_aug.n_tiles)
                    batched_augs.append(curr_aug)
        spatial_augs.append(resize)  # always perform resize last

        batch_transform = A.BatchCompose(
            [
                A.ForEach(pixel_augs),
                *batched_augs,
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["bboxes_classes_batch", "bboxes_visibility_batch"],
            ),
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoints_visibility_batch", "keypoints_classes_batch"],
                remove_invisible=False,
            ),
        )

        spatial_transform = A.Compose(
            spatial_augs,
            bbox_params=A.BboxParams(
                format="coco", label_fields=["bboxes_classes", "bboxes_visibility"]
            ),
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoints_visibility", "keypoints_classes"],
                remove_invisible=False,
            ),
        )

        return batch_transform, spatial_transform

    def __call__(
        self,
        data: List[Tuple[np.ndarray, Dict[LabelType, np.ndarray]]],
        nc: int = 1,
        ns: int = 1,
        nk: int = 1,
    ) -> Tuple[np.ndarray, Dict[LabelType, np.ndarray]]:
        """Performs augmentations on provided data

        Args:
            data (List[Tuple[np.ndarray, Dict[LabelType, np.ndarray]]]): Data with list of input images and their annotations
            nc (int, optional): Number of classes. Defaults to 1.
            ns (int, optional): Number of segmentation classes. Defaults to 1.
            nk (int, optional): Number of keypoints per instance. Defaults to 1.

        Returns:
            Tuple[np.ndarray, Dict[LabelType, np.ndarray]]: Output image and its annotations
        """

        image_batch = []
        mask_batch = []
        bboxes_batch = []
        bboxes_visibility_batch = []
        bboxes_classes_batch = []
        keypoints_batch = []
        keypoints_visibility_batch = []
        keypoints_classes_batch = []

        present_annotations = set()
        bbox_counter = 0
        for img, annotations in data:
            present_annotations.update(annotations.keys())
            (
                classes,
                mask,
                bboxes_points,
                bboxes_classes,
                keypoints_points,
                keypoints_visibility,
                keypoints_classes,
            ) = self.prepare_img_annotations(annotations, *img.shape[:-1])

            image_batch.append(img)
            mask_batch.append(mask)

            bboxes_batch.append(bboxes_points)
            bboxes_visibility_batch.append(
                [i + bbox_counter for i in range(bboxes_points.shape[0])]
            )
            bboxes_classes_batch.append(bboxes_classes)
            bbox_counter += bboxes_points.shape[0]

            keypoints_batch.append(keypoints_points)
            keypoints_visibility_batch.append(keypoints_visibility)
            keypoints_classes_batch.append(keypoints_classes)

        # Apply transforms
        # NOTE: All keys (including label_fields) must have _batch suffix when using BatchCompose
        transformed = self.batch_transform(
            image_batch=image_batch,
            mask_batch=mask_batch,
            bboxes_batch=bboxes_batch,
            bboxes_visibility_batch=bboxes_visibility_batch,
            bboxes_classes_batch=bboxes_classes_batch,
            keypoints_batch=keypoints_batch,
            keypoints_visibility_batch=keypoints_visibility_batch,
            keypoints_classes_batch=keypoints_classes_batch,
        )

        # convert to numpy arrays
        for key in transformed:
            transformed[key] = np.array(transformed[key][0])

        transformed = self.spatial_transform(
            image=transformed["image_batch"],
            mask=transformed["mask_batch"],
            bboxes=transformed["bboxes_batch"],
            bboxes_visibility=transformed["bboxes_visibility_batch"],
            bboxes_classes=transformed["bboxes_classes_batch"],
            keypoints=transformed["keypoints_batch"],
            keypoints_visibility=transformed["keypoints_visibility_batch"],
            keypoints_classes=transformed["keypoints_classes_batch"],
        )

        out_image, out_mask, out_bboxes, out_keypoints = self.post_transform_process(
            transformed,
            ns=ns,
            nk=nk,
            filter_kpts_by_bbox=(LabelType.BOUNDINGBOX in present_annotations)
            and (LabelType.KEYPOINT in present_annotations),
        )

        out_annotations = {}
        for key in present_annotations:
            if key == LabelType.CLASSIFICATION:
                out_annotations[LabelType.CLASSIFICATION] = classes
            elif key == LabelType.SEGMENTATION:
                out_annotations[LabelType.SEGMENTATION] = out_mask
            elif key == LabelType.BOUNDINGBOX:
                out_annotations[LabelType.BOUNDINGBOX] = out_bboxes
            elif key == LabelType.KEYPOINT:
                out_annotations[LabelType.KEYPOINT] = out_keypoints

        return out_image, out_annotations

    def prepare_img_annotations(
        self, annotations: Dict[LabelType, np.ndarray], ih: int, iw: int
    ) -> Tuple[np.ndarray]:
        """Prepare annotations to be compatible with albumentations

        Args:
            annotations (Dict[LabelType, np.ndarray]): Dict with annotations
            ih (int): Input image height
            iw (int): Input image width

        Returns:
            Tuple[np.ndarray]: All the data needed for albumentations input
        """

        classes = annotations.get(LabelType.CLASSIFICATION, np.zeros(1))

        seg = annotations.get(LabelType.SEGMENTATION, np.zeros((1, ih, iw)))
        mask = np.argmax(seg, axis=0) + 1
        mask[np.sum(seg, axis=0) == 0] = 0  # only background has value 0

        # COCO format in albumentations is [x,y,w,h] non-normalized
        bboxes = annotations.get(LabelType.BOUNDINGBOX, np.zeros((0, 5)))
        bboxes_points = bboxes[:, 1:]
        bboxes_points[:, 0::2] *= iw
        bboxes_points[:, 1::2] *= ih
        bboxes_points = self.check_bboxes(bboxes_points)
        bboxes_classes = bboxes[:, 0]

        # albumentations expects list of keypoints e.g. [(x,y),(x,y),(x,y),(x,y)]
        keypoints = annotations.get(LabelType.KEYPOINT, np.zeros((1, 3 + 1)))
        keypoints_unflat = np.reshape(keypoints[:, 1:], (-1, 3))
        keypoints_points = keypoints_unflat[:, :2]
        keypoints_points[:, 0] *= iw
        keypoints_points[:, 1] *= ih
        keypoints_visibility = keypoints_unflat[:, 2]
        # albumentations expects classes to be same length as keypoints
        # (use case: each kpt separate class - not supported in LuxonisDataset)
        nk = int((keypoints.shape[1] - 1) / 3)
        keypoints_classes = np.repeat(keypoints[:, 0], nk)

        return (
            classes,
            mask,
            bboxes_points,
            bboxes_classes,
            keypoints_points,
            keypoints_visibility,
            keypoints_classes,
        )

    def post_transform_process(
        self,
        transformed_data: Dict[str, np.ndarray],
        ns: int,
        nk: int,
        filter_kpts_by_bbox: bool,
    ) -> Tuple[np.ndarray]:
        """Postprocessing of albumentations output to LuxonisLoader format

        Args:
            transformed_data (Dict[str, np.ndarray]): Output data from albumentations
            ns (int): Number of segmentation classes
            nk (int): Number of keypoints per instance
            filter_kpts_by_bbox (bool): If True removes keypoint instances if its bounding box was removed.

        Returns:
            Tuple[np.ndarray]: Postprocessed annotations
        """

        out_image = transformed_data["image"].astype(np.float32)
        ih, iw, _ = out_image.shape
        if not self.train_rgb:
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)

        transformed_mask = transformed_data["mask"]
        out_mask = np.zeros((ns, *transformed_mask.shape))
        for key in np.unique(transformed_mask):
            if key != 0:
                out_mask[int(key) - 1, ...] = transformed_mask == key
        out_mask[out_mask > 0] = 1

        if len(transformed_data["bboxes"]):
            transformed_bboxes_classes = np.expand_dims(
                transformed_data["bboxes_classes"], axis=-1
            )
            out_bboxes = np.concatenate(
                (transformed_bboxes_classes, transformed_data["bboxes"]), axis=1
            )
        else:  # if no bboxes after transform
            out_bboxes = np.zeros((0, 5))
        out_bboxes[:, 1::2] /= iw
        out_bboxes[:, 2::2] /= ih

        transformed_keypoints_vis = np.expand_dims(
            transformed_data["keypoints_visibility"], axis=-1
        )
        out_keypoints = np.concatenate(
            (transformed_data["keypoints"], transformed_keypoints_vis), axis=1
        )
        out_keypoints = self.mark_invisible_keypoints(out_keypoints, ih, iw)
        out_keypoints[..., 0] /= iw
        out_keypoints[..., 1] /= ih
        if nk == 0:
            nk = 1  # done for easier postprocessing
        out_keypoints = np.reshape(out_keypoints, (-1, nk * 3))
        keypoints_classes = transformed_data["keypoints_classes"]
        # keypoints classes are repeated so take one per instance
        keypoints_classes = keypoints_classes[0::nk]
        keypoints_classes = np.expand_dims(keypoints_classes, axis=-1)
        out_keypoints = np.concatenate((keypoints_classes, out_keypoints), axis=1)
        if filter_kpts_by_bbox:
            out_keypoints = out_keypoints[
                transformed_data["bboxes_visibility"]
            ]  # keep only keypoints of visible instances

        return out_image, out_mask, out_bboxes, out_keypoints

    def check_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        """Check bbox annotations and correct those with width or height 0"""
        for i in range(bboxes.shape[0]):
            if bboxes[i, 2] == 0:
                bboxes[i, 2] = 1
            if bboxes[i, 3] == 0:
                bboxes[i, 3] = 1
        return bboxes

    def mark_invisible_keypoints(
        self, keypoints: np.ndarray, ih: int, iw: int
    ) -> np.ndarray:
        """Mark invisible keypoints with label == 0"""
        for kp in keypoints:
            if not (0 <= kp[0] < iw and 0 <= kp[1] < ih):
                kp[2] = 0
            if kp[2] == 0:  # per COCO format invisible points have x=y=0
                kp[0] = kp[1] = 0
        return keypoints


class TrainAugmentations(Augmentations):
    def __init__(
        self,
        image_size: List[int],
        augmentations: List[Dict[str, Any]],
        train_rgb: bool = True,
        keep_aspect_ratio: bool = True,
    ):
        """Class for train augmentations

        Args:
            image_size (List[int]): Desired image size
            augmentations (List[Dict[str, Any]]): List of dictionaries like {name: str, params: dict} for augmentation init
            train_rgb (bool, optional): Weather use RGB images. Defaults to True.
            keep_aspect_ratio (bool, optional): Weather to perform resize that original aspect ratio of the image. Defaults to True.
        """
        super().__init__(train_rgb=train_rgb)

        self.batch_transform, self.spatial_transform = self._parse_cfg(
            image_size=image_size,
            augmentations=augmentations,
            keep_aspect_ratio=keep_aspect_ratio,
        )


class ValAugmentations(Augmentations):
    def __init__(
        self,
        image_size: List[int],
        augmentations: List[Dict[str, Any]],
        train_rgb: bool = True,
        keep_aspect_ratio: bool = True,
    ):
        """Class for validation augmentations which performs only normalization (if present) and resize

        Args:
            image_size (List[int]): Desired image size
            augmentations (List[Dict[str, Any]]): List of dictionaries like {name: str, params: dict} for augmentation init
            train_rgb (bool, optional): Weather use RGB images. Defaults to True.
            keep_aspect_ratio (bool, optional): Weather to perform resize that original aspect ratio of the image. Defaults to True.
        """
        super().__init__(train_rgb=train_rgb)

        self.batch_transform, self.spatial_transform = self._parse_cfg(
            image_size=image_size,
            augmentations=[a for a in augmentations if a["name"] == "Normalize"],
            keep_aspect_ratio=keep_aspect_ratio,
        )


from albumentations.core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    KeypointInternalType,
)
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox


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
        """Augmentation to apply letterbox resizing to images. Alse transforms
        masks, bboxes and keypoints to correct shape.

        Args:
            height (int): Desired height of the output
            width (int): Desired width of the output
            interpolation (int, optional): Cv2 flag to specify interpolation used when resizing. Defaults to cv2.INTER_LINEAR.
            border_value (int, optional): Padding value for images. Defaults to 0.
            mask_value (int, optional): Padding value for masks. Defaults to 0.
            always_apply (bool, optional): Defaults to False.
            p (float, optional): Probability of applying the transform. Defaults to 1.0.
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
        **params
    ) -> np.ndarray:
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
        **params
    ) -> np.ndarray:
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
        bbox: BoxInternalType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params
    ) -> BoxInternalType:
        x_min, y_min, x_max, y_max = denormalize_bbox(
            bbox, self.height - pad_top - pad_bottom, self.width - pad_left - pad_right
        )[:4]
        bbox = [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
        # clip bbox to image, ignoring padding
        for i in range(4):
            pad = pad_left if i % 2 == 0 else pad_top
            img_size = params["cols"] if i % 2 == 0 else params["rows"]
            bbox[i] = self._clip(bbox[i], pad, img_size + pad)
        return normalize_bbox(bbox, self.height, self.width)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params
    ) -> KeypointInternalType:
        x, y, angle, scale = keypoint[:4]
        scale_x = (self.width - pad_left - pad_right) / params["cols"]
        scale_y = (self.height - pad_top - pad_bottom) / params["rows"]
        new_x = (x * scale_x) + pad_left
        new_y = (y * scale_y) + pad_top
        # if keypoint is in the padding then set coordinates to -1
        out_keypoint = (
            new_x
            if not self._out_of_bounds(new_x, pad_left, params["cols"] + pad_left)
            else -1,
            new_y
            if not self._out_of_bounds(new_y, pad_top, params["rows"] + pad_top)
            else -1,
            angle,
            scale * max(scale_x, scale_y),
        )
        return out_keypoint

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("height", "width", "interpolation", "border_value", "mask_value")

    def _clip(self, value: float, min_limit: float, max_limit: float) -> float:
        """Clip value to range"""
        return max(min(value, max_limit), min_limit)

    def _out_of_bounds(self, value: float, min_limit: float, max_limit: float) -> bool:
        """Check if value is out of set range"""
        return value < min_limit or value > max_limit


class DeterministicMosaic4(Mosaic4):
    """Mosaic4 adaptaion that always uses batch images in same order"""

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        target_params = super().get_params_dependent_on_targets(params)
        target_params["indices"] = list(range(self.n_tiles))
        return target_params
