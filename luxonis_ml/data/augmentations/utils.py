from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

from luxonis_ml.utils.registry import Registry

from ..utils.enums import LabelType
from .batch_compose import BatchCompose
from .batch_transform import BatchBasedTransform

AUGMENTATIONS = Registry(name="augmentations")


class Augmentations:
    def __init__(
        self,
        image_size: List[int],
        augmentations: List[Dict[str, Any]],
        train_rgb: bool = True,
        keep_aspect_ratio: bool = True,
        only_normalize: bool = False,
    ):
        """Base class for augmentations that are used in LuxonisLoader.

        @type train_rgb: bool
        @param train_rgb: Whether should use RGB or BGR images.
        """

        self.image_size = image_size
        self.train_rgb = train_rgb
        self.only_normalize = only_normalize

        self.is_batched = False
        self.aug_batch_size = 1

        (
            self.batch_transform,
            self.spatial_transform,
            self.resize_transform,
        ) = self._parse_cfg(
            image_size=image_size,
            augmentations=[a for a in augmentations if a["name"] == "Normalize"]
            if only_normalize
            else augmentations,
            keep_aspect_ratio=keep_aspect_ratio,
        )

    def _parse_cfg(
        self,
        image_size: List[int],
        augmentations: List[Dict[str, Any]],
        keep_aspect_ratio: bool = True,
    ) -> Tuple[BatchCompose, A.Compose]:
        """Parses provided config and returns Albumentations BatchedCompose object and
        Compose object for default transforms.

        @type image_size: List[int]
        @param image_size: Desired image size [H,W]
        @type augmentations: List[Dict[str, Any]]
        @param augmentations: List of augmentations to use and their params
        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether should use resize that keeps aspect ratio of
            original image.
        @rtype: Tuple[BatchCompose, A.Compose]
        @return: Objects for batched and spatial transforms
        """

        # NOTE: Always perform Resize
        if keep_aspect_ratio:
            resize = AUGMENTATIONS.get("LetterboxResize")(
                height=image_size[0], width=image_size[1]
            )
        else:
            resize = A.Resize(image_size[0], image_size[1])

        spatial_augs = []
        batched_augs = []
        if self.only_normalize:
            spatial_augs.append(resize)
        if augmentations:
            for aug in augmentations:
                curr_aug = AUGMENTATIONS.get(aug["name"])(**aug.get("params", {}))
                if isinstance(curr_aug, BatchBasedTransform):
                    self.is_batched = True
                    self.aug_batch_size = max(self.aug_batch_size, curr_aug.batch_size)
                    batched_augs.append(curr_aug)
                else:
                    spatial_augs.append(curr_aug)

        batch_transform = BatchCompose(
            [
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

        resize_transform = A.Compose(
            [resize],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["bboxes_classes", "bboxes_visibility"]
            ),
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["keypoints_visibility", "keypoints_classes"],
                remove_invisible=False,
            ),
        )

        return batch_transform, spatial_transform, resize_transform

    def __call__(
        self,
        data: List[Tuple[np.ndarray, Dict[LabelType, np.ndarray]]],
        ns: int = 1,
        nk: int = 1,
    ) -> Tuple[np.ndarray, Dict[LabelType, np.ndarray]]:
        """Performs augmentations on provided data.

        @type data: List[Tuple[np.ndarray, Dict[LabelType, np.ndarray]]]
        @param data: Data with list of input images and their annotations
        @type nc: int
        @param nc: Number of classes
        @type ns: int
        @param ns: Number of segmentation classes
        @type nk: int
        @param nk: Number of keypoints per instance
        @rtype: Tuple[np.ndarray, Dict[LabelType, np.ndarray]]
        @return: Output image and its annotations
        """

        present_annotations = {
            key for _, annotations in data for key in annotations.keys()
        }
        return_mask = LabelType.SEGMENTATION in present_annotations
        image_batch = []
        mask_batch = []
        bboxes_batch = []
        bboxes_visibility_batch = []
        bboxes_classes_batch = []
        keypoints_batch = []
        keypoints_visibility_batch = []
        keypoints_classes_batch = []

        bbox_counter = 0
        for img, annotations in data:
            (
                classes,
                mask,
                bboxes_points,
                bboxes_classes,
                keypoints_points,
                keypoints_visibility,
                keypoints_classes,
            ) = self.prepare_img_annotations(
                annotations, *img.shape[:-1], nk=nk, return_mask=return_mask
            )

            image_batch.append(img)
            if return_mask:
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

        transform_args = {
            "image_batch": image_batch,
            "bboxes_batch": bboxes_batch,
            "bboxes_visibility_batch": bboxes_visibility_batch,
            "bboxes_classes_batch": bboxes_classes_batch,
            "keypoints_batch": keypoints_batch,
            "keypoints_visibility_batch": keypoints_visibility_batch,
            "keypoints_classes_batch": keypoints_classes_batch,
        }
        if return_mask:
            transform_args["mask_batch"] = mask_batch

        # Apply transforms
        transformed = self.batch_transform(force_apply=False, **transform_args)
        transformed = {key: np.array(value[0]) for key, value in transformed.items()}

        # Prepare the spatial transform arguments
        spatial_transform_args = {
            "image": transformed["image_batch"],
            "bboxes": transformed["bboxes_batch"],
            "bboxes_visibility": transformed["bboxes_visibility_batch"],
            "bboxes_classes": transformed["bboxes_classes_batch"],
            "keypoints": transformed["keypoints_batch"],
            "keypoints_visibility": transformed["keypoints_visibility_batch"],
            "keypoints_classes": transformed["keypoints_classes_batch"],
        }
        if return_mask:
            spatial_transform_args["mask"] = transformed["mask_batch"]

        transformed = self.spatial_transform(
            force_apply=False, **spatial_transform_args
        )

        if (
            transformed["image"].shape[0] != self.image_size[0]
            or transformed["image"].shape[1] != self.image_size[1]
        ):
            resize_transform_args = {
                "image": transformed["image"],
                "bboxes": transformed["bboxes"],
                "bboxes_visibility": transformed["bboxes"],
                "bboxes_classes": transformed["bboxes_classes"],
                "keypoints": transformed["keypoints"],
                "keypoints_visibility": transformed["keypoints_visibility"],
                "keypoints_classes": transformed["keypoints_classes"],
            }

            if return_mask:
                resize_transform_args["mask"] = transformed["mask"]

            transformed = self.resize_transform(
                force_apply=False, **resize_transform_args
            )

        out_image, out_mask, out_bboxes, out_keypoints = self.post_transform_process(
            transformed,
            ns=ns,
            nk=nk,
            filter_kpts_by_bbox=(LabelType.BOUNDINGBOX in present_annotations)
            and (LabelType.KEYPOINTS in present_annotations),
            return_mask=return_mask,
        )

        out_annotations = {}
        for key in present_annotations:
            if key == LabelType.CLASSIFICATION:
                out_annotations[LabelType.CLASSIFICATION] = classes  # type: ignore
            elif key == LabelType.SEGMENTATION:
                out_annotations[LabelType.SEGMENTATION] = out_mask
            elif key == LabelType.BOUNDINGBOX:
                out_annotations[LabelType.BOUNDINGBOX] = out_bboxes
            elif key == LabelType.KEYPOINTS:
                out_annotations[LabelType.KEYPOINTS] = out_keypoints

        return out_image, out_annotations

    def prepare_img_annotations(
        self,
        annotations: Dict[LabelType, np.ndarray],
        ih: int,
        iw: int,
        nk: int,
        return_mask: bool = True,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Prepare annotations to be compatible with albumentations.

        @type annotations: Dict[LabelType, np.ndarray]
        @param annotations: Dict with annotations
        @type ih: int
        @param ih: Input image height
        @type iw: int
        @param iw: Input image width
        @type return_mask: bool
        @param return_mask: Whether to compute and return mask
        @rtype: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray]
        @return: Annotations in albumentations format
        """

        classes = annotations.get(LabelType.CLASSIFICATION, np.zeros(1))

        mask = None
        if return_mask:
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
        keypoints = annotations.get(LabelType.KEYPOINTS, np.zeros((1, nk * 3 + 1)))
        keypoints_unflat = np.reshape(keypoints[:, 1:], (-1, 3))
        keypoints_points = keypoints_unflat[:, :2]
        keypoints_points[:, 0] *= iw
        keypoints_points[:, 1] *= ih
        keypoints_visibility = keypoints_unflat[:, 2]
        # albumentations expects classes to be same length as keypoints
        # (use case: each kpt separate class - not supported in LuxonisDataset)
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
        return_mask: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Postprocessing of albumentations output to LuxonisLoader format.

        @type transformed_data: Dict[str, np.ndarray]
        @param transformed_data: Output data from albumentations
        @type ns: int
        @param ns: Number of segmentation classes
        @type nk: int
        @param nk: Number of keypoints per instance
        @type filter_kpts_by_bbox: bool
        @param filter_kpts_by_bbox: If True removes keypoint instances if its bounding
            box was removed.
        @rtype: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]
        @return: Postprocessed annotations
        """

        out_image = transformed_data["image"]
        ih, iw, _ = out_image.shape
        if not self.train_rgb:
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        out_image = out_image.astype(np.float32)

        out_mask = None
        if return_mask:
            transformed_mask = transformed_data.get("mask")
            out_mask = (
                np.zeros((ns, *transformed_mask.shape))
                if transformed_mask is not None
                else None
            )
            if transformed_mask is not None:
                assert out_mask is not None
                for key in np.unique(transformed_mask):
                    if key != 0:
                        out_mask[int(key) - 1, ...] = transformed_mask == key
                out_mask[out_mask > 0] = 1

        if transformed_data["bboxes"]:
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

        if nk == 0:
            nk = 1  # done for easier postprocessing
        if transformed_data["keypoints"]:
            out_keypoints = np.concatenate(
                (transformed_data["keypoints"], transformed_keypoints_vis), axis=1
            )
        else:
            out_keypoints = np.zeros((0, nk * 3 + 1))

        out_keypoints = self.mark_invisible_keypoints(out_keypoints, ih, iw)
        out_keypoints[..., 0] /= iw
        out_keypoints[..., 1] /= ih
        out_keypoints = np.reshape(out_keypoints, (-1, nk * 3))
        keypoints_classes = transformed_data["keypoints_classes"]
        keypoints_classes = keypoints_classes[0::nk]
        keypoints_classes = np.expand_dims(keypoints_classes, axis=-1)
        out_keypoints = np.concatenate((keypoints_classes, out_keypoints), axis=1)
        if filter_kpts_by_bbox:
            transformed_visibility = transformed_data["bboxes_visibility"]
            if (
                len(transformed_visibility) != len(out_keypoints)
                and not self.batch_transform.transforms
            ):
                raise ValueError(
                    "Number of keypoints and bounding boxes are not equal. "
                    "This can happen when using a dataset with multiple keypoint and boundingbox "
                    "tasks. To avoid this, use the 'detection' type when generating the dataset "
                    "instead of 'keypoints' and 'boundingbox' separately, or use the following "
                    "naming convention for task names: 'task_name-keypoints' and 'task_name-boundingbox."
                )
            out_keypoints = out_keypoints[
                transformed_data["bboxes_visibility"]
            ]  # keep only keypoints of visible instances

        return out_image, out_mask, out_bboxes, out_keypoints

    def check_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        """Check bbox annotations and correct those with width or height 0.

        @type bboxes: np.ndarray
        @param bboxes: A numpy array representing bounding boxes.
        @rtype: np.ndarray
        @return: The same bounding boxes with any out-of-bounds coordinate corrections.
        """

        for i in range(bboxes.shape[0]):
            if bboxes[i, 2] == 0:
                bboxes[i, 2] = 1
            if bboxes[i, 3] == 0:
                bboxes[i, 3] = 1
        return bboxes

    def mark_invisible_keypoints(
        self, keypoints: np.ndarray, ih: int, iw: int
    ) -> np.ndarray:
        """Mark invisible keypoints with label == 0.

        @type keypoints: np.ndarray
        @param keypoints: A numpy array representing keypoints.
        @type ih: int
        @param ih: The image height.
        @type iw: int
        @param iw: The image width.
        @rtype: np.ndarray
        @return: The same keypoints with corrections to mark keypoints out-of-bounds as
            invisible.
        """
        for kp in keypoints:
            if not (0 <= kp[0] < iw and 0 <= kp[1] < ih):
                kp[2] = 0
            if kp[2] == 0:  # per COCO format invisible points have x=y=0
                kp[0] = kp[1] = 0
        return keypoints


# Registering all supported transforms
# Pixel-level transforms
AUGMENTATIONS.register_module(module=A.AdvancedBlur)
AUGMENTATIONS.register_module(module=A.Blur)
AUGMENTATIONS.register_module(module=A.CLAHE)
AUGMENTATIONS.register_module(module=A.ChannelDropout)
AUGMENTATIONS.register_module(module=A.ChannelShuffle)
AUGMENTATIONS.register_module(module=A.ColorJitter)
AUGMENTATIONS.register_module(module=A.Defocus)
AUGMENTATIONS.register_module(module=A.Downscale)
AUGMENTATIONS.register_module(module=A.Emboss)
AUGMENTATIONS.register_module(module=A.Equalize)
AUGMENTATIONS.register_module(module=A.FDA)
AUGMENTATIONS.register_module(module=A.FancyPCA)
AUGMENTATIONS.register_module(module=A.FromFloat)
AUGMENTATIONS.register_module(module=A.GaussNoise)
AUGMENTATIONS.register_module(module=A.GaussianBlur)
AUGMENTATIONS.register_module(module=A.GlassBlur)
AUGMENTATIONS.register_module(module=A.HistogramMatching)
AUGMENTATIONS.register_module(module=A.HueSaturationValue)
AUGMENTATIONS.register_module(module=A.ISONoise)
AUGMENTATIONS.register_module(module=A.ImageCompression)
AUGMENTATIONS.register_module(module=A.InvertImg)
AUGMENTATIONS.register_module(module=A.MedianBlur)
AUGMENTATIONS.register_module(module=A.MotionBlur)
AUGMENTATIONS.register_module(module=A.MultiplicativeNoise)
AUGMENTATIONS.register_module(module=A.Normalize)
AUGMENTATIONS.register_module(module=A.PixelDistributionAdaptation)
AUGMENTATIONS.register_module(module=A.Posterize)
AUGMENTATIONS.register_module(module=A.RGBShift)
AUGMENTATIONS.register_module(module=A.RandomBrightnessContrast)
AUGMENTATIONS.register_module(module=A.RandomFog)
AUGMENTATIONS.register_module(module=A.RandomGamma)
AUGMENTATIONS.register_module(module=A.RandomGravel)
AUGMENTATIONS.register_module(module=A.RandomRain)
AUGMENTATIONS.register_module(module=A.RandomShadow)
AUGMENTATIONS.register_module(module=A.RandomSnow)
AUGMENTATIONS.register_module(module=A.RandomSunFlare)
AUGMENTATIONS.register_module(module=A.RandomToneCurve)
AUGMENTATIONS.register_module(module=A.RingingOvershoot)
AUGMENTATIONS.register_module(module=A.Sharpen)
AUGMENTATIONS.register_module(module=A.Solarize)
AUGMENTATIONS.register_module(module=A.Spatter)
AUGMENTATIONS.register_module(module=A.Superpixels)
AUGMENTATIONS.register_module(module=A.TemplateTransform)
AUGMENTATIONS.register_module(module=A.ToFloat)
AUGMENTATIONS.register_module(module=A.ToGray)
AUGMENTATIONS.register_module(module=A.ToRGB)
AUGMENTATIONS.register_module(module=A.ToSepia)
AUGMENTATIONS.register_module(module=A.UnsharpMask)
AUGMENTATIONS.register_module(module=A.ZoomBlur)

# Spatial.level transforms
# NOTE: only augmentations that are supported for all targets
AUGMENTATIONS.register_module(module=A.Affine)
# AUGMENTATIONS.register_module(module=A.BBoxSafeRandomCrop)
AUGMENTATIONS.register_module(module=A.CenterCrop)
AUGMENTATIONS.register_module(module=A.CoarseDropout)
AUGMENTATIONS.register_module(module=A.Crop)
AUGMENTATIONS.register_module(module=A.CropAndPad)
AUGMENTATIONS.register_module(module=A.CropNonEmptyMaskIfExists)
# AUGMENTATIONS.register_module(module=A.ElasticTransform)
AUGMENTATIONS.register_module(module=A.Flip)
# AUGMENTATIONS.register_module(module=A.GridDistortion)
# AUGMENTATIONS.register_module(module=A.GridDropout)
AUGMENTATIONS.register_module(module=A.HorizontalFlip)
AUGMENTATIONS.register_module(module=A.Lambda)
AUGMENTATIONS.register_module(module=A.LongestMaxSize)
# AUGMENTATIONS.register_module(module=A.MaskDropout)
AUGMENTATIONS.register_module(module=A.NoOp)
# AUGMENTATIONS.register_module(module=A.OpticalDistortion)
AUGMENTATIONS.register_module(module=A.PadIfNeeded)
AUGMENTATIONS.register_module(module=A.Perspective)
AUGMENTATIONS.register_module(module=A.PiecewiseAffine)
AUGMENTATIONS.register_module(module=A.PixelDropout)
AUGMENTATIONS.register_module(module=A.RandomCrop)
AUGMENTATIONS.register_module(module=A.RandomCropFromBorders)
AUGMENTATIONS.register_module(module=A.RandomCropNearBBox)
AUGMENTATIONS.register_module(module=A.RandomGridShuffle)
AUGMENTATIONS.register_module(module=A.RandomResizedCrop)
AUGMENTATIONS.register_module(module=A.RandomRotate90)
AUGMENTATIONS.register_module(module=A.RandomScale)
# AUGMENTATIONS.register_module(module=A.RandomSizedBBoxSafeCrop)
AUGMENTATIONS.register_module(module=A.RandomSizedCrop)
AUGMENTATIONS.register_module(module=A.Resize)
AUGMENTATIONS.register_module(module=A.Rotate)
AUGMENTATIONS.register_module(module=A.SafeRotate)
AUGMENTATIONS.register_module(module=A.ShiftScaleRotate)
AUGMENTATIONS.register_module(module=A.SmallestMaxSize)
AUGMENTATIONS.register_module(module=A.Transpose)
AUGMENTATIONS.register_module(module=A.VerticalFlip)
