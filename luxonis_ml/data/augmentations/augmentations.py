import warnings
from collections import defaultdict
from math import prod
from typing import Any, Dict, List, Literal, Set, Tuple

import albumentations as A
import numpy as np
from typing_extensions import TypeAlias, override

from luxonis_ml.data.utils.task_utils import get_task_name, task_is_metadata
from luxonis_ml.typing import ConfigItem, LoaderOutput, TaskType

from .base_pipeline import AugmentationEngine
from .batch_compose import BatchCompose
from .batch_transform import BatchBasedTransform
from .custom import LetterboxResize, MixUp, Mosaic4
from .utils import (
    postprocess_bboxes,
    postprocess_keypoints,
    postprocess_mask,
    preprocess_bboxes,
    preprocess_keypoints,
    preprocess_mask,
)

Data: TypeAlias = Dict[str, np.ndarray]


class Augmentations(AugmentationEngine, register_name="albumentations"):
    def __init__(
        self,
        height: int,
        width: int,
        batch_transform: BatchCompose,
        spatial_transform: A.Compose,
        pixel_transform: A.Compose,
        resize_transform: A.Compose,
        targets: Dict[str, str],
        targets_to_tasks: Dict[str, str],
        special_targets: Dict[Literal["metadata", "classification"], Set[str]],
    ):
        self.image_size = (height, width)
        self.special_targets = special_targets
        self.targets = targets
        self.targets_to_tasks = targets_to_tasks

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
        targets: Dict[str, TaskType],
        config: List[ConfigItem],
        keep_aspect_ratio: bool = True,
        is_validation_pipeline: bool = False,
        min_bbox_visibility: float = 0.001,
    ) -> "Augmentations":
        if keep_aspect_ratio:
            resize = LetterboxResize(height=height, width=width)
        else:
            resize = A.Resize(height=height, width=width)

        if is_validation_pipeline:
            config = [a for a in config if a["name"] == "Normalize"]

        pixel_augs = []
        spatial_augs = []
        batched_augs = []
        batch_size = 1
        for aug in config:
            curr_aug = cls._get_transform(aug)
            if isinstance(curr_aug, A.ImageOnlyTransform):
                pixel_augs.append(curr_aug)
            elif isinstance(curr_aug, BatchBasedTransform):
                batch_size *= curr_aug.batch_size
                batched_augs.append(curr_aug)
            elif isinstance(curr_aug, A.DualTransform):
                spatial_augs.append(curr_aug)

        main_task_names = set()
        alb_targets = {}
        alb_targets_to_tasks = {}
        special_targets: Dict[
            Literal["metadata", "classification"], Set[str]
        ] = {
            "metadata": set(),
            "classification": set(),
        }
        for task, task_type in targets.items():
            if task_type == "classification":
                special_targets["classification"].add(task)
                continue
            elif task_is_metadata(task):
                special_targets["metadata"].add(task)
                continue

            if task_type in {"segmentation", "instance_segmentation"}:
                task_type = "mask"
            elif task_type == "boundingbox":
                task_type = "bboxes"

            task_name = get_task_name(task)

            safe_task = task.replace("/", "_").replace("-", "_")
            assert safe_task.isidentifier()
            alb_targets[safe_task] = task_type
            alb_targets_to_tasks[safe_task] = task
            main_task_names.add(task_name)

        def _get_params():
            return {
                "bbox_params": A.BboxParams(
                    format="albumentations",
                    # Bug in albumentations v1.4.18 (the latest installable
                    # on python 3.8) causes the pipeline to eventually
                    # crash when set to 0.
                    min_visibility=min_bbox_visibility,
                ),
                "keypoint_params": A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
                "additional_targets": alb_targets,
            }

        with warnings.catch_warnings(record=True):
            return cls(
                height=height,
                width=width,
                batch_transform=BatchCompose(batched_augs, **_get_params()),
                spatial_transform=A.Compose(spatial_augs, **_get_params()),
                pixel_transform=A.Compose(pixel_augs),
                resize_transform=A.Compose([resize], **_get_params()),
                targets=alb_targets,
                targets_to_tasks=alb_targets_to_tasks,
                special_targets=special_targets,
            )

    @property
    @override
    def batch_size(self) -> int:
        return self.batch_transform.batch_size

    @override
    def apply(self, data: List[LoaderOutput]) -> LoaderOutput:
        new_data = []
        for img, labels in data:
            labels["image"] = img
            new_data.append(labels)

        return self._apply(new_data)

    def _apply(self, data: List[Data]) -> LoaderOutput:
        metadata = defaultdict(list)
        classification = defaultdict(list)
        for labels in data:
            for metadata_task in self.special_targets["metadata"]:
                if metadata_task in labels:
                    metadata[metadata_task].append(labels[metadata_task])
            for classification_task in self.special_targets["classification"]:
                if classification_task in labels:
                    classification[classification_task].append(
                        labels[classification_task]
                    )
        metadata = {k: np.concatenate(v) for k, v in metadata.items()}
        classification = {
            k: np.clip(sum(v), 0, 1) for k, v in classification.items()
        }

        data, n_keypoints, n_mask_classes = self.preprocess(data)

        if self.batch_transform.transforms:
            transformed = self.batch_transform(data)
        else:
            transformed = data[0]

        if self.spatial_transform.transforms:
            transformed = self.spatial_transform(**transformed)

        transformed_size = transformed["image"].shape[:2]

        if transformed_size != self.image_size:
            transformed_size = prod(transformed_size)
            target_size = prod(self.image_size)

            if transformed_size > target_size:
                transformed = self.resize_transform(**transformed)
                transformed = self.pixel_transform(transformed)
            else:
                transformed = self.pixel_transform(transformed)
                transformed = self.resize_transform(**transformed)
        else:
            transformed = self.pixel_transform(transformed)

        return self.postprocess(
            transformed, metadata, classification, n_keypoints, n_mask_classes
        )

    def preprocess_data(
        self, labels: Data, bbox_counters: Dict[str, int]
    ) -> Tuple[Data, Dict[str, int], Dict[str, int]]:
        img = labels.pop("image")
        height, width = img.shape[:2]
        data = {"image": img}
        n_keypoints = {}
        n_segmentation_classes = {}
        for task, task_type in self.targets.items():
            override_name = task
            task = self.targets_to_tasks[task]

            if task not in labels:
                data[override_name] = np.array([])
                continue

            array = labels[task]
            if task_type == "mask":
                n_segmentation_classes[override_name] = array.shape[0]
                data[override_name] = preprocess_mask(array)
            elif task_type == "bboxes":
                data[override_name] = preprocess_bboxes(
                    array, bbox_counters[override_name]
                )
            elif task_type == "keypoints":
                n_keypoints[override_name] = array.shape[1] // 3
                data[override_name] = preprocess_keypoints(
                    array, height, width
                )
        return data, n_keypoints, n_segmentation_classes

    def preprocess(
        self, data: List[Data]
    ) -> Tuple[List[Data], Dict[str, int], Dict[str, int]]:
        batch_data = []
        bbox_counters = defaultdict(int)
        n_keypoints = {}
        n_segmentation_classes = {}

        for d in data:
            d, _n_keypoints, _n_segmentation_classes = self.preprocess_data(
                d, bbox_counters
            )
            n_keypoints.update(_n_keypoints)
            n_segmentation_classes.update(_n_segmentation_classes)
            for task, array in d.items():
                if task == "image":
                    continue
                if self.targets[task] == "bboxes":
                    bbox_counters[task] += array.shape[0]
            batch_data.append(d)
        return batch_data, n_keypoints, n_segmentation_classes

    def postprocess(
        self,
        data: Data,
        metadata: Data,
        classification: Data,
        n_keypoints: Dict[str, int],
        n_segmentation_classes: Dict[str, int],
    ) -> LoaderOutput:
        out_labels = {}
        out_image = data.pop("image")
        image_height, image_width, _ = out_image.shape

        bboxes_orderings = {}
        targets = sorted(
            list(data.keys()), key=lambda x: self.targets[x] != "bboxes"
        )

        for target in targets:
            array = data[target]
            if array.shape[0] == 0:
                continue

            task = self.targets_to_tasks[target]
            task_name = get_task_name(task)
            task_type = self.targets[target]
            if task_type == "mask":
                if target not in n_segmentation_classes:
                    continue
                out_labels[task] = postprocess_mask(
                    array, n_segmentation_classes[target]
                )
            elif task_type == "bboxes":
                out_labels[task], ordering = postprocess_bboxes(array)
                bboxes_orderings[task_name] = ordering
            elif task_type == "keypoints" and task_name in bboxes_orderings:
                out_labels[task] = postprocess_keypoints(
                    array,
                    bboxes_orderings[task_name],
                    image_height,
                    image_width,
                    n_keypoints[target],
                )

        out_metadata = {}
        for task, value in metadata.items():
            task_name = get_task_name(task)
            out_metadata[task] = value[
                bboxes_orderings.get(task_name, np.array([], dtype=np.uint8))
            ]

        out_labels.update(**out_metadata, **classification)
        return out_image, out_labels

    @staticmethod
    def _get_transform(config: ConfigItem) -> A.BasicTransform:
        name = config["name"]
        params = config.get("params", {})
        # TODO: Registry
        if name == "MixUp":
            return MixUp(**params)  # type: ignore
        if name == "Mosaic4":
            return Mosaic4(**params)  # type: ignore
        if not hasattr(A, name):
            raise ValueError(
                f"Augmentation {name} not found in Albumentations"
            )
        return getattr(A, name)(**params)
