import random
from collections import defaultdict
from math import prod
from typing import Any, Dict, List, Set, Tuple

import albumentations as A
import numpy as np
from typing_extensions import override

from luxonis_ml.data.utils import get_qualified_task_name, get_task_type
from luxonis_ml.typing import Labels, LoaderOutput

from .base_pipeline import AugmentationEngine
from .batch_compose import BatchCompose
from .batch_transform import BatchBasedTransform
from .custom import LetterboxResize, MixUp, Mosaic4
from .utils import (
    post_process_bboxes,
    post_process_keypoints,
    post_process_mask,
    prepare_bboxes,
    prepare_keypoints,
    prepare_mask,
)


class Augmentations(AugmentationEngine, register_name="albumentations"):
    def __init__(
        self,
        height: int,
        width: int,
        batch_size: int,
        batch_transform: BatchCompose,
        spatial_transform: A.Compose,
        pixel_transform: A.Compose,
        resize_transform: A.Compose,
    ):
        self.image_size = (height, width)
        self._batch_size = batch_size

        self.batch_transform = batch_transform
        self.spatial_transform = spatial_transform

        def apply_pixel_transform(data: Dict[str, Any]) -> Dict[str, Any]:
            if pixel_transform.transforms:
                data["image"] = pixel_transform(image=data["image"])["image"]
            return data

        self.pixel_transform = apply_pixel_transform
        self.resize_transform = resize_transform

    @classmethod
    @override
    def from_config(
        cls,
        height: int,
        width: int,
        config: List[Dict[str, Any]],
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

        def _get_params():
            return {
                "bbox_params": A.BboxParams(
                    format="coco",
                    label_fields=["bboxes_classes", "bboxes_visibility"],
                    min_visibility=0.01,
                ),
                "keypoint_params": A.KeypointParams(
                    format="xy",
                    label_fields=["keypoints_visibility", "keypoints_classes"],
                    remove_invisible=False,
                ),
            }

        return cls(
            height=height,
            width=width,
            batch_size=batch_size,
            batch_transform=BatchCompose(batched_augs, **_get_params()),
            spatial_transform=A.Compose(spatial_augs, **_get_params()),
            pixel_transform=A.Compose(pixel_augs),
            resize_transform=A.Compose([resize], **_get_params()),
        )

    @property
    @override
    def batch_size(self) -> int:
        return self._batch_size

    @override
    def apply(self, data: List[LoaderOutput]) -> LoaderOutput:
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        task_names = {
            get_qualified_task_name(task)
            for _, label in data
            for task in label
        }
        out_labels = {}

        if not task_names:
            return self._apply(data, 0, 0)

        for task_name in task_names:
            task_types_to_names = {}
            batch_data: List[LoaderOutput] = []
            nk = 0
            ns = 0
            for img, labels in data:
                label_subset = {}
                for task, label in labels.items():
                    if get_qualified_task_name(task) == task_name:
                        task_type = get_task_type(task)
                        if task_type == "metadata":
                            task_type = task.split("/", 1)[1]
                        label_subset[task_type] = label
                        task_types_to_names[task_type] = task
                        if task_type == "keypoints":
                            nk = (label.shape[1] - 1) // 3
                        elif task_type == "segmentation":
                            ns = label.shape[0]
                batch_data.append((img, label_subset))

            random.setstate(random_state)
            np.random.set_state(np_random_state)

            # TODO: Optimize
            aug_img, aug_labels = self._apply(batch_data, ns, nk)
            for task_type, label in aug_labels.items():
                out_labels[task_types_to_names[task_type]] = label

        return aug_img, out_labels

    def _apply(
        self,
        data: List[LoaderOutput],
        n_segmentation_classes: int,
        n_keypoints: int,
    ) -> LoaderOutput:
        present_labels = {key for _, label in data for key in label.keys()}
        return_mask = "segmentation" in present_labels

        bbox_counter = 0
        batch_data = []
        for img, labels in data:
            # TODO: how to deal with classes for batch augmentations?
            classes = labels.get("classification", np.zeros(1))

            t = self.prepare_img_labels(
                labels,
                *img.shape[:-1],
                n_keypoints=n_keypoints,
                return_mask=return_mask,
            )
            t["image"] = img
            t["bboxes_visibility"] = np.arange(
                bbox_counter, t["bboxes"].shape[0] + bbox_counter
            )

            bbox_counter += t["bboxes"].shape[0]
            batch_data.append(t)

        metadata = defaultdict(list)
        for _, ann in data:
            for task, label in ann.items():
                if task.startswith("metadata/"):
                    metadata[task].append(label)
        metadata = {k: np.concatenate(v) for k, v in metadata.items()}

        if self.batch_transform.transforms:
            transformed = self.batch_transform(batch_data)
        else:
            transformed = batch_data[0]

        if self.spatial_transform.transforms:
            transformed = self.spatial_transform(**transformed)

        if transformed["image"].shape[:2] != self.image_size:
            transformed_size = prod(transformed["image"].shape[:2])
            target_size = prod(self.image_size)

            if transformed_size > target_size:
                transformed = self.resize_transform(**transformed)
                transformed = self.pixel_transform(transformed)
            elif transformed_size < target_size:
                transformed = self.pixel_transform(transformed)
                transformed = self.resize_transform(**transformed)
        else:
            transformed = self.pixel_transform(transformed)

        out_image, out_labels = self._post_transform(
            transformed,
            metadata,
            n_segmentation_classes,
            n_keypoints,
            present_labels,
        )

        if "classification" in present_labels:
            out_labels["classification"] = classes

        return out_image, out_labels

    def prepare_img_labels(
        self,
        labels: Labels,
        height: int,
        width: int,
        n_keypoints: int,
        return_mask: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Prepare labels to be compatible with albumentations.

        @type labels: Dict[str, np.ndarray]
        @param labels: Dict with labels
        @type height: int
        @param height: Input image height
        @type width: int
        @param width: Input image width
        @type n_keypoints: int
        @param n_keypoints: Number of keypoints per instance
        @type return_mask: bool
        @param return_mask: Whether to compute and return mask
        @rtype: Tuple[np.ndarray, Optional[np.ndarray], np.ndarray,
            np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        @return: Labels in albumentations format
        """

        mask = None
        if return_mask:
            mask = prepare_mask(labels, height, width)

        bboxes, bboxes_classes = prepare_bboxes(labels, height, width)
        keypoints_points, keypoints_visibility, keypoints_classes = (
            prepare_keypoints(labels, height, width, n_keypoints)
        )

        data = {
            "bboxes": bboxes,
            "bboxes_classes": bboxes_classes,
            "keypoints": keypoints_points,
            "keypoints_visibility": keypoints_visibility,
            "keypoints_classes": keypoints_classes,
        }
        if mask is not None:
            data["mask"] = mask

        return data

    def _post_transform(
        self,
        data: Dict[str, np.ndarray],
        metadata: Dict[str, np.ndarray],
        n_segmentation_classes: int,
        n_keypoints: int,
        present_labels: Set[str],
    ) -> Tuple[
        np.ndarray,
        Dict[str, np.ndarray],
    ]:
        """Postprocessing of albumentations output to LuxonisLoader
        format.

        @type data: Dict[str, np.ndarray]
        @param data: Output data from albumentations
        @type metadata: Dict[str, np.ndarray]
        @param metadata: Metadata
        @type n_segmentation_classes: int
        @param n_segmentation_classes: Number of segmentation classes
        @type n_keypoints: int
        @param n_keypoints: Number of keypoints per instance
        @type present_labels: Set[str]
        @param present_labels: Set of present labels
        @rtype: Tuple[np.ndarray, Dict[str, np.ndarray]]
        @return: Image and labels
        """

        out_image = data["image"]
        image_height, image_width, _ = out_image.shape
        out_image = out_image.astype(np.float32)

        out_mask = None
        if "mask" in data:
            out_mask = post_process_mask(data["mask"], n_segmentation_classes)

        out_bboxes = post_process_bboxes(
            data["bboxes"],
            np.expand_dims(data["bboxes_classes"], axis=-1),
            image_height,
            image_width,
        )
        out_keypoints = post_process_keypoints(
            data["keypoints"],
            data["keypoints_visibility"],
            data["keypoints_classes"],
            n_keypoints,
            image_height,
            image_width,
        )

        visible_bboxes = [int(v) for v in data["bboxes_visibility"]]

        if {"boundingbox", "keypoints"} <= present_labels:
            out_keypoints = out_keypoints[visible_bboxes]

        out_metadata = {}
        for key, value in metadata.items():
            out_metadata[key] = value[visible_bboxes]

        out_labels = {}
        if "boundingbox" in present_labels:
            out_labels["boundingbox"] = out_bboxes
        if "keypoints" in present_labels:
            out_labels["keypoints"] = out_keypoints
        if out_mask is not None:
            out_labels["segmentation"] = out_mask
        out_labels.update(out_metadata)
        return out_image, out_labels

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
