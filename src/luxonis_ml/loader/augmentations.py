import numpy as np
import cv2
import albumentations as A
from albumentations import *
from typing import List, Tuple, Optional
from .loader import LabelType


class Augmentations:
    def __init__(self, train_rgb: Optional[bool] = True):
        """Base class for augmentations that are used in LuxonisLoader

        Args:
            train_rgb (Optional[bool], optional): Flag if should use RGB or BGR images. Defaults to True.
        """
        self.train_rgb = train_rgb
        
        self.is_batched = False
        self.aug_batch_size = 1

    def _parse_cfg(
        self, image_size: list, augmentations: dict, keep_aspect_ratio: Optional[bool] = True
    ):
        """Parses provided config and returns Albumentations BatchedCompose object and Compose object for default transforms

        Args:
            image_size (list): Desired image size [H,W]
            augmentations (dict): Dict of augmentations to use and their params
            keep_aspect_ratio (Optional[bool], optional): Flat if should use resize that keeps aspect ratio of original image. Defaults to True.

        Returns:
            Tuple[A.BatchedCompose, A.Compose]: Objects for batched and default transforms
        """
        image_size = image_size

        # Always perform Resize
        if keep_aspect_ratio:
            resize = A.Sequential(
                [
                    A.LongestMaxSize(max_size=max(image_size), interpolation=1),
                    A.PadIfNeeded(
                        min_height=image_size[0],
                        min_width=image_size[1],
                        border_mode=0,
                        value=(0, 0, 0),
                    ),
                ],
                p=1,
            )
        else:
            resize = A.Resize(image_size[0], image_size[1])

        pixel_augs = []
        spatial_augs = []
        batched_augs = []
        if augmentations:
            for aug in augmentations:
                curr_aug = eval(aug["name"])(**aug.get("params", {}))
                if isinstance(curr_aug, A.ImageOnlyTransform):
                    pixel_augs.append(curr_aug)
                elif isinstance(curr_aug, A.DualTransform):
                    spatial_augs.append(curr_aug)
                elif isinstance(curr_aug, A.BatchBasedTransform):
                    self.is_batched = True
                    self.aug_batch_size = max(self.aug_batch_size, curr_aug.n_tiles)
                    batched_augs.append(curr_aug)
        spatial_augs.append(resize) # always perform resize last

        batch_transform = A.BatchCompose(
            [
                A.ForEach(pixel_augs),
                *batched_augs,
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["bboxes_classes_batch"]),
            keypoint_params=A.KeypointParams(
                format="xy", 
                label_fields=["keypoints_visibility_batch", "keypoints_classes_batch"],
                remove_invisible=False
            ),
        )

        spatial_transform = A.Compose(
            spatial_augs,
            bbox_params=A.BboxParams(format="coco", label_fields=["bboxes_classes", "bboxes_visibility"]),
            keypoint_params=A.KeypointParams(
                format="xy", 
                label_fields=["keypoints_visibility", "keypoints_classes"],
                remove_invisible=False
            )
        )

        return batch_transform, spatial_transform


    def __call__(self, data: List[Tuple[np.ndarray, dict]]):
        """Performs augmentations on provided data

        Args:
            data (List[Tuple[np.ndarray, dict]]): Data with list of input images and their annotations

        Returns:
            Tuple[np.ndarray, dict]: Transformed output image and its annotations
        """
        image_batch = []
        mask_batch = []
        bboxes_batch = []
        bboxes_classes_batch = []
        keypoints_batch = []
        keypoints_visibility_batch = []
        keypoints_classes_batch = []

        present_annotations = set()
        for img, annotations in data:
            present_annotations.update(annotations.keys())
            classes, mask, bboxes_points, bboxes_classes, keypoints_points, \
            keypoints_visibility, keypoints_classes, n_kpts_per_instance = self.prepare_img_annotations(annotations, *img.shape[:-1])

            image_batch.append(img)
            mask_batch.append(mask)
            bboxes_batch.append(bboxes_points)
            bboxes_classes_batch.append(bboxes_classes)
            keypoints_batch.append(keypoints_points)
            keypoints_visibility_batch.append(keypoints_visibility)
            keypoints_classes_batch.append(keypoints_classes)

        # Apply transforms
        # NOTE: All keys (including label_fields) must have _batch suffix when using BatchCompose
        transformed = self.batch_transform(
            image_batch=image_batch,
            mask_batch=mask_batch,
            bboxes_batch=bboxes_batch,
            bboxes_classes_batch=bboxes_classes_batch,
            keypoints_batch=keypoints_batch,
            keypoints_visibility_batch=keypoints_visibility_batch,
            keypoints_classes_batch=keypoints_classes_batch
        )

        # convert to numpy arrays
        for key in transformed:
            transformed[key] = np.array(transformed[key][0])

        transformed = self.spatial_transform(
            image=transformed["image_batch"],
            mask=transformed["mask_batch"],
            bboxes=transformed["bboxes_batch"],
            bboxes_classes=transformed["bboxes_classes_batch"],
            bboxes_visibility=[i for i in range(transformed["bboxes_batch"].shape[0])],
            keypoints=transformed["keypoints_batch"],
            keypoints_visibility=transformed["keypoints_visibility_batch"],
            keypoints_classes=transformed["keypoints_classes_batch"]
        )

        out_image, out_mask, out_bboxes, out_keypoints = self.post_transform_process(transformed, n_kpts_per_instance, 
            filter_kpts_by_bbox=LabelType.BOUNDINGBOX in present_annotations
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

    def prepare_img_annotations(self, annotations: dict, ih: int, iw: int):
        """Prepare annotations to be compatible with albumentations

        Args:
            annotations (dict): Dict with annotations
            ih (int): Input image height
            iw (int): Input image width

        Returns:
            Tuple: All the data needed for albumentations input
        """

        classes = annotations.get(LabelType.CLASSIFICATION, np.zeros(1))

        seg = annotations.get(LabelType.SEGMENTATION, np.zeros((1, ih, iw)))
        mask = np.argmax(seg, axis=0) + 1
        mask[np.sum(seg, axis=0)==0] = 0 # only background has value 0
        
        # COCO format in albumentations is [x,y,w,h] non-normalized
        bboxes = annotations.get(LabelType.BOUNDINGBOX, np.zeros((0, 5)))
        bboxes_points = bboxes[:, 1:]
        bboxes_points[:, 0::2] *= iw
        bboxes_points[:, 1::2] *= ih
        bboxes_points = self.check_bboxes(bboxes_points)
        bboxes_classes = bboxes[:, 0]

        # albumentations expects list of keypoints e.g. [(x,y),(x,y),(x,y),(x,y)]
        keypoints = annotations.get(LabelType.KEYPOINT, np.zeros((1, 3 + 1)))
        keypoints_unflat = np.reshape(keypoints[:, 1:], (-1,3))
        keypoints_points = keypoints_unflat[:,:2]
        keypoints_points[:, 0] *= iw
        keypoints_points[:, 1] *= ih
        keypoints_visibility = keypoints_unflat[:,2]
        # albumentations expects classes to be same length as keypoints 
        # (use case: each kpt separate class - not supported in LuxonisDataset)
        n_kpts_per_instance = int((keypoints.shape[1]-1) / 3)
        keypoints_classes = np.repeat(keypoints[:, 0], n_kpts_per_instance)

        return classes, mask, bboxes_points, bboxes_classes, keypoints_points, \
            keypoints_visibility, keypoints_classes, n_kpts_per_instance

    def post_transform_process(self, transformed_data: dict, n_kpts_per_instance: int, filter_kpts_by_bbox: bool):
        """Postprocessing of albumentations output to LuxonisLoader format

        Args:
            transformed_data (dict): Output data from albumentations
            n_kpts_per_instance (int): Number of keypoints per instance
            filter_kpts_by_bbox (bool): If True removes keypoint instances if its bounding box was removed.

        Returns:
            Tuple: Postprocessed annotations
        """
        out_image = transformed_data["image"]
        ih, iw, _ = out_image.shape
        if not self.train_rgb:
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)

        transformed_mask = transformed_data["mask"]
        keys = np.unique(transformed_mask[transformed_mask!=0])
        if len(keys):
            out_mask = np.zeros((len(keys), *transformed_mask.shape))
            for idx, key in enumerate(keys):
                out_mask[idx] = transformed_mask == key
        else: # if mask is just background
            out_mask = np.zeros((1,*transformed_mask.shape))

        if len(transformed_data["bboxes"]):
            transformed_bboxes_classes = np.expand_dims(transformed_data["bboxes_classes"], axis=-1)
            out_bboxes = np.concatenate((transformed_bboxes_classes, transformed_data["bboxes"]), axis=1)
        else: # if no bboxes after transform
            out_bboxes = np.zeros((0,5))
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
        out_keypoints = np.reshape(out_keypoints, (-1, n_kpts_per_instance*3))
        keypoints_classes = transformed_data["keypoints_classes"]
        # keypoints classes are repeated so take one per instance
        keypoints_classes = keypoints_classes[0::n_kpts_per_instance]
        keypoints_classes = np.expand_dims(keypoints_classes, axis=-1)
        out_keypoints = np.concatenate((keypoints_classes, out_keypoints), axis=1)
        if filter_kpts_by_bbox:
            out_keypoints = out_keypoints[transformed_data["bboxes_visibility"]] # keep only keypoints of visible instances

        return out_image, out_mask, out_bboxes, out_keypoints

    def check_bboxes(self, bboxes: np.ndarray):
        """Check bbox annotations and correct those with width or height 0"""
        for i in range(bboxes.shape[0]):
            if bboxes[i, 2] == 0:
                bboxes[i, 2] = 1
            if bboxes[i, 3] == 0:
                bboxes[i, 3] = 1
        return bboxes
    
    def mark_invisible_keypoints(self, keypoints: np.ndarray, ih: int, iw: int):
        """Mark invisible keypoints with label == 0"""
        for kp in keypoints:
            if not (0 <= kp[0] < iw and 0 <= kp[1] < ih):
                kp[2] = 0
            if kp[2] == 0: # per COCO format invisible points have x=y=0
                kp[0] = kp[1] = 0
        return keypoints
    

class TrainAugmentations(Augmentations):
    def __init__(
        self,
        image_size: list,
        augmentations: dict,
        train_rgb: bool = True,
        keep_aspect_ratio: bool = True,
    ):
        """Class for train augmentations"""
        super().__init__(train_rgb=train_rgb)
        self.batch_transform, self.spatial_transform = self._parse_cfg(
            image_size=image_size,
            augmentations=augmentations,
            keep_aspect_ratio=keep_aspect_ratio,
        )


class ValAugmentations(Augmentations):
    def __init__(
        self,
        image_size: list,
        augmentations: dict,
        train_rgb: bool = True,
        keep_aspect_ratio: bool = True,
    ):
        """Class for val augmentations, only performs Normalize augmentation if present"""
        super().__init__(train_rgb=train_rgb)
        self.batch_transform, self.spatial_transform = self._parse_cfg(
            image_size=image_size,
            augmentations=[a for a in augmentations if a["name"] == "Normalize"],
            keep_aspect_ratio=keep_aspect_ratio,
        )