import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import albumentations as A
import cv2
import numpy as np

from luxonis_ml.data.utils import (
    LuxonisLoaderOutput,
    get_task_name,
    get_task_type,
)

from .batch_compose import BatchCompose
from .batch_transform import BatchBasedTransform
from .batch_utils import unbatch_all
from .custom import LetterboxResize, MixUp, Mosaic4


class AugmentationConfiguration(TypedDict):
    name: str
    params: Dict[str, Any]


class BaseAugmentationPipeline(ABC):
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        height: int,
        width: int,
        config: List[AugmentationConfiguration],
        out_rgb: bool,
        keep_aspect_ratio: bool,
        is_validation_pipeline: bool,
    ) -> "BaseAugmentationPipeline":
        """Create augmentation pipeline from configuration.

        @type height: int
        @param height: Target image height
        @type width: int
        @param width: Target image width
        @type config: List[Dict[str, Any]]
        @param config: List of dictionaries with augmentation
            configurations.
        @type out_rgb: bool
        @param out_rgb: Whether to output RGB images
        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep aspect ratio
        @type is_validation_pipeline: bool
        @param is_validation_pipeline: Whether this is a validation
            pipeline (in which case some augmentations are skipped)
        @rtype: BaseAugmentationPipeline
        @return: Initialized augmentation pipeline
        """
        ...

    @abstractmethod
    def apply(
        self, data: List[LuxonisLoaderOutput]
    ) -> LuxonisLoaderOutput: ...

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Getter for the batch size.

        The batch size is the number of images necessary for the
        augmentation pipeline to work in case of batch-based
        augmentations.

        For example, if the augmentation pipeline contains the MixUp
        augmentation, the batch size should be 2.

        If the pipeline requires MixUp and also Mosaic4 augmentations,
        the batch size should be 6 (2 + 4).
        """
        ...

    @property
    def is_batched(self) -> bool:
        return self.batch_size > 1


class Augmentations(BaseAugmentationPipeline):
    def __init__(
        self,
        height: int,
        width: int,
        out_rgb: bool,
        batch_size: int,
        batch_transform: BatchCompose,
        spatial_transform: A.Compose,
        pixel_transform: A.Compose,
        resize_transform: A.Compose,
    ):
        self.image_size = (height, width)
        self.out_rgb = out_rgb
        self._batch_size = batch_size

        self.batch_transform = batch_transform
        self.spatial_transform = spatial_transform
        self.pixel_transform = pixel_transform
        self.resize_transform = resize_transform

    @classmethod
    def from_config(
        cls,
        height: int,
        width: int,
        config: List[AugmentationConfiguration],
        out_rgb: bool = True,
        keep_aspect_ratio: bool = True,
        is_validation_pipeline: bool = False,
    ) -> "Augmentations":
        if keep_aspect_ratio:
            resize = LetterboxResize(height=height, width=width)
        else:
            resize = A.Resize(height=height, width=width)

        if is_validation_pipeline:
            config = [a for a in config if a.get("name") == "Normalize"]

        pixel_augs = []
        spatial_augs = []
        batched_augs = []
        batch_size = 1
        for aug in config:
            curr_aug = cls._get_augmentation(
                aug["name"], **aug.get("params", {})
            )
            if isinstance(curr_aug, A.ImageOnlyTransform):
                pixel_augs.append(curr_aug)
            elif isinstance(curr_aug, A.DualTransform):
                spatial_augs.append(curr_aug)
            elif isinstance(curr_aug, BatchBasedTransform):
                batch_size *= curr_aug.batch_size
                batched_augs.append(curr_aug)

        def _get_params(batch: bool = False):
            suffix = "_batch" if batch else ""
            return {
                "bbox_params": A.BboxParams(
                    format="coco",
                    label_fields=[
                        f"bboxes_classes{suffix}",
                        f"bboxes_visibility{suffix}",
                    ],
                    min_visibility=0.01,
                ),
                "keypoint_params": A.KeypointParams(
                    format="xy",
                    label_fields=[
                        f"keypoints_visibility{suffix}",
                        f"keypoints_classes{suffix}",
                    ],
                    remove_invisible=False,
                ),
            }

        return cls(
            height=height,
            width=width,
            out_rgb=out_rgb,
            batch_size=batch_size,
            batch_transform=BatchCompose(
                batched_augs, **_get_params(batch=True)
            ),
            spatial_transform=A.Compose(spatial_augs, **_get_params()),
            pixel_transform=A.Compose(pixel_augs),
            resize_transform=A.Compose([resize], **_get_params()),
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @staticmethod
    def _get_augmentation(name: str, **kwargs) -> A.BasicTransform:
        # TODO: Registry
        if name == "MixUp":
            return MixUp(**kwargs)
        if name == "Mosaic4":
            return Mosaic4(**kwargs)
        if not hasattr(A, name):
            raise ValueError(
                f"Augmentation {name} not found in Albumentations"
            )
        return getattr(A, name)(**kwargs)

    def apply(self, data: List[LuxonisLoaderOutput]) -> LuxonisLoaderOutput:
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        reference_labels = data[0][1]
        task_names = {get_task_name(task) for task in reference_labels.keys()}
        out_labels = {}

        for task_name in sorted(list(task_names)):
            task_types_to_names = {}
            data_subset: List[LuxonisLoaderOutput] = []
            nk = 0
            ns = 0
            for img, labels in data:
                label_subset = {}
                for task, label in labels.items():
                    if get_task_name(task) == task_name:
                        task_type = get_task_type(task)
                        label_subset[task_type] = label
                        task_types_to_names[task_type] = task
                        if task_type == "keypoints":
                            nk = (label.shape[1] - 1) // 3
                        elif task_type == "segmentation":
                            ns = label.shape[0]
                data_subset.append((img, label_subset))

            random.setstate(random_state)
            np.random.set_state(np_random_state)

            # TODO: Optimize
            aug_img, aug_labels = self._apply(data_subset, ns, nk)
            for task_type, label in aug_labels.items():
                out_labels[task_types_to_names[task_type]] = label

        return aug_img, out_labels

    def _apply(
        self, data: List[LuxonisLoaderOutput], ns: int, nk: int
    ) -> LuxonisLoaderOutput:
        present_annotations = {
            get_task_type(key)
            for _, annotations in data
            for key in annotations.keys()
        }
        return_mask = "segmentation" in present_annotations
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

        transformed: Dict[str, Any] = {
            "image_batch": image_batch,
            "bboxes_batch": bboxes_batch,
            "bboxes_visibility_batch": bboxes_visibility_batch,
            "bboxes_classes_batch": bboxes_classes_batch,
            "keypoints_batch": keypoints_batch,
            "keypoints_visibility_batch": keypoints_visibility_batch,
            "keypoints_classes_batch": keypoints_classes_batch,
        }

        if return_mask:
            transformed["mask_batch"] = mask_batch

        if self.batch_transform.transforms:
            transformed = self.batch_transform(
                force_apply=False, **transformed
            )
        else:
            transformed = unbatch_all(transformed)

        transformed = self.spatial_transform(**transformed, force_apply=False)

        if transformed["image"].shape[:2] != self.image_size:
            transformed = self.resize_transform(
                **transformed, force_apply=False
            )

        transformed["image"] = self.pixel_transform(
            image=transformed["image"], force_apply=False
        )["image"]

        out_image, out_mask, out_bboxes, out_keypoints = (
            self.post_transform_process(
                transformed,
                ns=ns,
                nk=nk,
                filter_kpts_by_bbox="boundingbox" in present_annotations
                and "keypoints" in present_annotations,
                return_mask=return_mask,
            )
        )

        out_annotations = {}
        for key in present_annotations:
            if key == "classification":
                out_annotations["classification"] = classes
            elif key == "segmentation":
                out_annotations["segmentation"] = out_mask
            elif key == "boundingbox":
                out_annotations["boundingbox"] = out_bboxes
            elif key == "keypoints":
                out_annotations["keypoints"] = out_keypoints

        return out_image, out_annotations

    def prepare_img_annotations(
        self,
        annotations: Dict[str, np.ndarray],
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
        @rtype: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        @return: Annotations in albumentations format
        """

        classes = annotations.get("classification", np.zeros(1))

        mask = None
        if return_mask:
            seg = annotations.get("segmentation", np.zeros((1, ih, iw)))
            mask = np.argmax(seg, axis=0) + 1
            mask[np.sum(seg, axis=0) == 0] = 0  # only background has value 0

        # COCO format in albumentations is [x,y,w,h] non-normalized
        bboxes = annotations.get("boundingbox", np.zeros((0, 5)))
        bboxes_points = bboxes[:, 1:]
        bboxes_points[:, 0::2] *= iw
        bboxes_points[:, 1::2] *= ih
        bboxes_points = self.check_bboxes(bboxes_points).astype(np.int32)
        bboxes_classes = bboxes[:, 0].astype(np.int32)

        # albumentations expects list of keypoints e.g. [(x,y),(x,y),(x,y),(x,y)]
        keypoints = annotations.get("keypoints", np.zeros((1, nk * 3 + 1)))
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
        """Postprocessing of albumentations output to LuxonisLoader
        format.

        @type transformed_data: Dict[str, np.ndarray]
        @param transformed_data: Output data from albumentations
        @type ns: int
        @param ns: Number of segmentation classes
        @type nk: int
        @param nk: Number of keypoints per instance
        @type filter_kpts_by_bbox: bool
        @param filter_kpts_by_bbox: If True removes keypoint instances
            if its bounding box was removed.
        @rtype: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray,
            np.ndarray]
        @return: Postprocessed annotations
        """

        out_image = transformed_data["image"]
        ih, iw, _ = out_image.shape
        if not self.out_rgb:
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        out_image = out_image.astype(np.float32)

        out_mask = None
        if return_mask:
            transformed_mask = transformed_data.get("mask")
            assert transformed_mask is not None
            out_mask = np.zeros((ns, *transformed_mask.shape))
            for key in np.unique(transformed_mask):
                if key != 0:
                    out_mask[int(key) - 1, ...] = transformed_mask == key
            out_mask[out_mask > 0] = 1

        if transformed_data["bboxes"].shape[0] > 0:
            transformed_bboxes_classes = np.expand_dims(
                transformed_data["bboxes_classes"], axis=-1
            )
            out_bboxes = np.concatenate(
                (transformed_bboxes_classes, transformed_data["bboxes"]),
                axis=1,
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
        if transformed_data["keypoints"].shape[0] > 0:
            out_keypoints = np.concatenate(
                (transformed_data["keypoints"], transformed_keypoints_vis),
                axis=1,
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
        out_keypoints = np.concatenate(
            (keypoints_classes, out_keypoints), axis=1
        )
        if filter_kpts_by_bbox:
            out_keypoints = out_keypoints[
                [int(v) for v in transformed_data["bboxes_visibility"]]
            ]  # keep only keypoints of visible instances

        return out_image, out_mask, out_bboxes, out_keypoints

    def check_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        """Check bbox annotations and correct those with width or height
        0.

        @type bboxes: np.ndarray
        @param bboxes: A numpy array representing bounding boxes.
        @rtype: np.ndarray
        @return: The same bounding boxes with any out-of-bounds
            coordinate corrections.
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
        @return: The same keypoints with corrections to mark keypoints
            out-of-bounds as invisible.
        """
        for kp in keypoints:
            if not (0 <= kp[0] < iw and 0 <= kp[1] < ih):
                kp[2] = 0
            if kp[2] == 0:  # per COCO format invisible points have x=y=0
                kp[0] = kp[1] = 0
        return keypoints
