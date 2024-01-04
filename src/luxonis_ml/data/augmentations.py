import numpy as np
import cv2
import warnings
import random
import albumentations as A
from typing import Any, Dict, List, Tuple, Union, Optional

from .loader import LabelType
from luxonis_ml.utils.registry import Registry

from albumentations.core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    KeypointInternalType,
    BoxType,
    KeypointType,
)
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox


AUGMENTATIONS = Registry(name="augmentations")


class Augmentations:
    def __init__(self, train_rgb: bool = True):
        """Base class for augmentations that are used in LuxonisLoader.

        Args:
            train_rgb (bool, optional): Whether should use RGB or BGR images. Defaults to True.
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
        """Parses provided config and returns Albumentations BatchedCompose object and
        Compose object for default transforms.

        Args:
            image_size (List[int]): Desired image size [H,W]
            augmentations (List[Dict[str, Any]]): List of augmentations to use and their params
            keep_aspect_ratio (bool, optional): Whether should use resize that keeps aspect ratio of original image. Defaults to True.

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
                curr_aug = AUGMENTATIONS.get(aug["name"])(**aug.get("params", {}))
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
        """Performs augmentations on provided data.

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
        """Prepare annotations to be compatible with albumentations.

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
        """Postprocessing of albumentations output to LuxonisLoader format.

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
        """Check bbox annotations and correct those with width or height 0.

        @type bboxes: np.ndarray @param bboxes: A numpy array
        representing bounding boxes.

        @rtype: np.ndarray @return: The same bounding boxes with any
        out-of-bounds coordinate corrections.
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

        @type keypoints: np.ndarray @param keypoints: A numpy array
        representing keypoints.

        @type ih: int @param ih: The image height.

        @type iw: int @param iw: The image width.

        @rtype: np.ndarray @return: The same keypoints with corrections
        to mark keypoints out-of-bounds as invisible.
        """
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
        """Class for train augmentations.

        Args:
            image_size (List[int]): Desired image size
            augmentations (List[Dict[str, Any]]): List of dictionaries like {name: str, params: dict} for augmentation init
            train_rgb (bool, optional): Whether use RGB images. Defaults to True.
            keep_aspect_ratio (bool, optional): Whether to perform resize that original aspect ratio of the image. Defaults to True.
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
        """Class for validation augmentations which performs only normalization (if
        present) and resize.

        Args:
            image_size (List[int]): Desired image size
            augmentations (List[Dict[str, Any]]): List of dictionaries like {name: str, params: dict} for augmentation init
            train_rgb (bool, optional): Whether use RGB images. Defaults to True.
            keep_aspect_ratio (bool, optional): Whether to perform resize that original aspect ratio of the image. Defaults to True.
        """
        super().__init__(train_rgb=train_rgb)

        self.batch_transform, self.spatial_transform = self._parse_cfg(
            image_size=image_size,
            augmentations=[a for a in augmentations if a["name"] == "Normalize"],
            keep_aspect_ratio=keep_aspect_ratio,
        )


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


@AUGMENTATIONS.register_module()
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
        """Augmentation to apply letterbox resizing to images. Also transforms masks,
        bboxes and keypoints to correct shape.

        @param height: Desired height of the output.
        @type height: int

        @param width: Desired width of the output.
        @type width: int

        @param interpolation: Cv2 flag to specify interpolation used when resizing. Defaults to cv2.INTER_LINEAR.
        @type interpolation: int, optional

        @param border_value: Padding value for images. Defaults to 0.
        @type border_value: int, optional

        @param mask_value: Padding value for masks. Defaults to 0.
        @type mask_value: int, optional

        @param always_apply: Whether to always apply the transform. Defaults to False.
        @type always_apply: bool, optional

        @param p: Probability of applying the transform. Defaults to 1.0.
        @type p: float, optional
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
        """Updates augmentation parameters with the necessary metadata.

        @param params: The existing augmentation parameters dictionary.
        @type params: Dict[str, Any]

        @param kwargs: Additional keyword arguments to add the parameters.
        @type kwargs: Any

        @return: Updated dictionary containing the merged parameters.
        @rtype: Dict[str, Any]
        """

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
        **params,
    ) -> np.ndarray:
        """Applies the letterbox augmentation to an image.

        @param img: Input image to which resize is applied.
        @type img: np.ndarray

        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int

        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int

        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int

        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int

        @param params: Additional parameters for the padding operation.
        @type params: Any

        @return: Image with applied letterbox resize.
        @rtype: np.ndarray
        """

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
        **params,
    ) -> np.ndarray:
        """Applies letterbox augmentation to the input mask.

        @param img: Input mask to which resize is applied.
        @type img: np.ndarray

        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int

        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int

        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int

        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int

        @param params: Additional parameters for the padding operation.
        @type params: Any

        @return: Mask with applied letterbox resize.
        @rtype: np.ndarray
        """

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
        **params,
    ) -> BoxInternalType:
        """Applies letterbox augmentation to the bounding box.

        @param img: Bounding box to which resize is applied.
        @type img: BoxInternalType

        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int

        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int

        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int

        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int

        @param params: Additional parameters for the padding operation.
        @type params: Any

        @return: Bounding box with applied letterbox resize.
        @rtype: BoxInternalType
        """

        x_min, y_min, x_max, y_max = denormalize_bbox(
            bbox, self.height - pad_top - pad_bottom, self.width - pad_left - pad_right
        )[:4]
        bbox = np.array(
            [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
        )
        # clip bbox to image, ignoring padding
        bbox = bbox.clip(
            min=[pad_left, pad_top] * 2,
            max=[params["cols"] + pad_left, params["rows"] + pad_top] * 2,
        ).tolist()
        return normalize_bbox(bbox, self.height, self.width)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params,
    ) -> KeypointInternalType:
        """Applies letterbox augmentation to the keypoint.

        @param img: Keypoint to which resize is applied.
        @type img: KeypointInternalType

        @param pad_top: Number of pixels to pad at the top.
        @type pad_top: int

        @param pad_bottom: Number of pixels to pad at the bottom.
        @type pad_bottom: int

        @param pad_left: Number of pixels to pad on the left.
        @type pad_left: int

        @param pad_right: Number of pixels to pad on the right.
        @type pad_right: int

        @param params: Additional parameters for the padding operation.
        @type params: Any

        @return: Keypoint with applied letterbox resize.
        @rtype: KeypointInternalType
        """

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
        """Gets the default arguments for the letterbox augmentation.

        @return: The string keywords of the arguments.
        @rtype: Tuple[str, ...]
        """

        return ("height", "width", "interpolation", "border_value", "mask_value")

    def _out_of_bounds(self, value: float, min_limit: float, max_limit: float) -> bool:
        """ "Check if the given value is outside the specified limits.

        @param value: The value to be checked.
        @type value: float

        @param min_limit: Minimum limit.
        @type min_limit: float

        @param max_limit: Maximum limit.
        @type max_limit: float

        @return: True if the value is outside the specified limits, False otherwise.
        @rtype: bool
        """
        return value < min_limit or value > max_limit


@AUGMENTATIONS.register_module(name="Mosaic4")
class DeterministicMosaic4(A.Mosaic4):
    def __init__(
        self,
        out_height: int,
        out_width: int,
        value: Optional[Union[int, float, List[int], List[float]]] = None,
        replace: bool = False,
        out_batch_size: int = 1,
        mask_value: Optional[Union[int, float, List[int], List[float]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        """Mosaic augmentation arranges selected four images into single image in a 2x2 grid layout. This
        is done in deterministic way meaning first image in the batch will always be in top left.
        The input images should have the same number of channels but can have different widths and heights.
        The output is cropped around the intersection point of the four images with the size (out_with x out_height).
        If the mosaic image is smaller than with x height, the gap is filled by the fill_value.

        @param out_height: Output image height. The mosaic image is cropped by this height around the mosaic center.
        If the size of the mosaic image is smaller than this value the gap is filled by the `value`.
        @type out_height: int

        @param out_width: Output image width. The mosaic image is cropped by this height around the mosaic center.
        If the size of the mosaic image is smaller than this value the gap is filled by the `value`.
        @type out_width: int

        @param value: Padding value. Defaults to None.
        @type value: Optional[Union[int, float, List[int], List[float]]], optional

        @param replace: Whether to replace the original images in the mosaic. Current implementation
        only supports this set to False. Defaults to False.
        @type replace: bool, optional

        @param out_batch_size: Number of output images in the batch. Defaults to 1.
        @type out_batch_size: int, optional

        @param mask_value: Padding value for masks. Defaults to None.
        @type mask_value: Optional[Union[int, float, List[int], List[float]]], optional

        @param always_apply: Whether to always apply the transform. Defaults to False.
        @type always_apply: bool, optional

        @param p: Probability of applying the transform. Defaults to 0.5.
        @type p: float, optional
        """

        super().__init__(
            out_height,
            out_width,
            value,
            replace,
            out_batch_size,
            mask_value,
            always_apply,
            p,
        )
        warnings.warn(
            "Only deterministic version of Mosaic4 is available, setting replace=False."
        )
        self.replace = False

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters dependent on the targets.

        @param params: Dictionary containing parameters.
        @type params: Dict[str, Any]

        @return: Dictionary containing parameters dependent on the targets.
        @rtype: Dict[str, Any]
        """
        target_params = super().get_params_dependent_on_targets(params)
        target_params["indices"] = list(range(self.n_tiles))
        return target_params


@AUGMENTATIONS.register_module()
class MixUp(A.BatchBasedTransform):
    def __init__(
        self,
        alpha: Union[float, Tuple[float, float]] = 0.5,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        """MixUp augmentation that merges two images and their annotations into
        one. If images are not of same size then second one is first resized to
        match the first one.

        @param alpha: Mixing coefficient, either a single float or a tuple representing the range.
        Defaults to 0.5.
        @type alpha: Union[float, Tuple[float, float]], optional

        @param always_apply: Whether to always apply the transform. Defaults to False.
        @type always_apply: bool, optional

        @param p: Probability of applying the transform. Defaults to 0.5.
        @type p: float, optional
        """
        super().__init__(always_apply=always_apply, p=p)

        self.alpha = alpha
        self.n_tiles = 2
        self.out_batch_size = 1

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Gets the default arguments for the mixup augmentation.

        @return: The string keywords of the arguments.
        @rtype: Tuple[str, ...]
        """
        return ("alpha", "out_batch_size")

    @property
    def targets_as_params(self) -> List[str]:
        """List of augmentation targets

        @return: Output list of augmentation targets.
        @rtype: List[str]
        """
        return ["image_batch"]

    def apply_to_image_batch(
        self,
        image_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of images.

        @param image_batch: Batch of input images to which the transformation is applied.
        @type image_batch: List[np.ndarray]

        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]

        @param params: Additional parameters for the transformation.
        @type params: Any

        @return: List of transformed images.
        @rtype: List[np.ndarray]
        """
        image1 = image_batch[0]
        # resize second image to size of the first one
        image2 = cv2.resize(image_batch[1], (image_shapes[0][1], image_shapes[0][0]))

        if isinstance(self.alpha, float):
            curr_alpha = np.clip(self.alpha, 0, 1)
        else:
            curr_alpha = random.uniform(max(self.alpha[0], 0), min(self.alpha[1], 1))
        img_out = cv2.addWeighted(image1, curr_alpha, image2, 1 - curr_alpha, 0.0)
        return [img_out]

    def apply_to_mask_batch(
        self,
        mask_batch: List[np.ndarray],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[np.ndarray]:
        """Applies the transformation to a batch of masks.

        @param image_batch: Batch of input masks to which the transformation is applied.
        @type image_batch: List[np.ndarray]

        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]

        @param params: Additional parameters for the transformation.
        @type params: Any

        @return: List of transformed masks.
        @rtype: List[np.ndarray]
        """
        mask1 = mask_batch[0]
        mask2 = cv2.resize(
            mask_batch[1],
            (image_shapes[0][1], image_shapes[0][0]),
            interpolation=cv2.INTER_NEAREST,
        )
        out_mask = mask1 + mask2
        # if masks intersect keep one present in first image
        mask_inter = mask1 > 0
        out_mask[mask_inter] = mask1[mask_inter]
        return [out_mask]

    def apply_to_bboxes_batch(
        self, bboxes_batch: List[BoxType], image_shapes: List[Tuple[int, int]], **params
    ) -> List[BoxType]:
        """Applies the transformation to a batch of bboxes.

        @param image_batch: Batch of input bboxes to which the transformation is applied.
        @type image_batch: List[BoxType]

        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]

        @param params: Additional parameters for the transformation.
        @type params: Any

        @return: List of transformed bboxes.
        @rtype: List[BoxType]
        """
        return [bboxes_batch[0] + bboxes_batch[1]]

    def apply_to_keypoints_batch(
        self,
        keypoints_batch: List[KeypointType],
        image_shapes: List[Tuple[int, int]],
        **params,
    ) -> List[KeypointType]:
        """Applies the transformation to a batch of keypoints.

        @param image_batch: Batch of input keypoints to which the transformation is applied.
        @type image_batch: List[BoxType]

        @param image_shapes: Shapes of the input images in the batch.
        @type image_shapes: List[Tuple[int, int]]

        @param params: Additional parameters for the transformation.
        @type params: Any

        @return: List of transformed keypoints.
        @rtype: List[BoxType]
        """
        scaled_kpts2 = []
        scale_x = image_shapes[0][1] / image_shapes[1][1]
        scale_y = image_shapes[0][0] / image_shapes[1][0]
        for kpt in keypoints_batch[1]:
            new_kpt = A.augmentations.geometric.functional.keypoint_scale(
                keypoint=kpt, scale_x=scale_x, scale_y=scale_y
            )
            scaled_kpts2.append(new_kpt + kpt[4:])
        return [keypoints_batch[0] + scaled_kpts2]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters dependent on the targets.

        @param params: Dictionary containing parameters.
        @type params: Dict[str, Any]

        @return: Dictionary containing parameters dependent on the targets.
        @rtype: Dict[str, Any]
        """
        image_batch = params["image_batch"]
        return {"image_shapes": [image.shape[:2] for image in image_batch]}
