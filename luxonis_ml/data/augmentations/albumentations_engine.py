import logging
import warnings
from collections import defaultdict
from math import prod
from typing import Any, Dict, List, Set, Tuple, TypedDict

import albumentations as A
import numpy as np
from typing_extensions import TypeAlias, override

from luxonis_ml.data.utils.task_utils import get_task_name, task_is_metadata
from luxonis_ml.typing import ConfigItem, LoaderOutput, TaskType

from .base_engine import AugmentationEngine
from .batch_compose import BatchCompose
from .batch_transform import BatchTransform
from .custom import TRANSFORMATIONS, LetterboxResize
from .utils import (
    postprocess_bboxes,
    postprocess_keypoints,
    postprocess_mask,
    preprocess_bboxes,
    preprocess_keypoints,
    preprocess_mask,
)

logger = logging.getLogger(__name__)

Data: TypeAlias = Dict[str, np.ndarray]


class SpecialTargets(TypedDict):
    """Dictionary for storing names of tasks that has to be augmented in
    a special way."""

    metadata: Set[str]
    classification: Set[str]
    arrays: Set[str]
    instance_masks: Set[str]


class AlbumentationsEngine(AugmentationEngine, register_name="albumentations"):
    """Augmentation engine using the Albumentations library under the
    hood.

    The order of transformations provided in the configuration is not
    guaranteed to be preserved. The transformations are divided into
    three groups: pixel transformations, spatial transformations, and
    batch transformations.

    Batch transformations are always applied first, followed by spatial
    transformations, and finally pixel transformations.

    Supported Standard Augmentations
    ================================

    All augmentations provided by the Albumentations library are supported.

    Supported Batch Augmentations
    ==============================

    MixUp
    -----

    MixUp is a data augmentation technique that blends 2 source
    images into a single image using a weight coefficient alpha.

    Mosaic4
    -------

    Mosaic4 transformation combines 4 images into a single image
    by placing them in a 2x2 grid.

    Extending the Engine
    ====================

    New transformations can be added by registering them in the
    `TRANSFORMATIONS` registry. All the custom transformations should
    inherit either from `A.BasicTransform` or `BatchBasedTransform`.

    Augmenting unsupported tasks
    ============================

    Metadata
    --------

    Metadata tasks can contain arbitrary data and their semantics are
    unknown to the augmentation engine. Therefore, the only transformation
    applied to metadata is discarding metadata associated with boxes
    falling outside the image.

    Arrays
    ------

    Arrays are dealt with in the same way as metadata.
    The only transformation applied to arrays is discarding
    arrays associated with bboxes falling outside the image.

    Oriented Bounding Boxes
    -----------------------

    (Not implemented yet)

    Oriented bounding boxes are of shape (n_boxes, 5) where
    the last dimension contains the angle of the box.
    This format is not supported by Albumentations, however,
    Albumentations support angle to be part of the keypoints.
    So, the oriented bounding boxes are split into regular
    bounding boxes and a set of keypoints that represent
    the center of the bbox and contain the angle as the third coordinate.

    Both the keypoints and the bboxes are augmented separately.
    At the end, the angle is extracted from the keypoints and added
    back to the bounding boxes. The keypoints are discarded.
    """

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
        special_tasks: SpecialTargets,
    ):
        """Initialize the AlbumentationsEngine.

        @type height: int
        @param height: Height of the output image
        @type width: int
        @param width: Width of the output image
        @type batch_transform: BatchCompose
        @param batch_transform: Batch transformations
        @type spatial_transform: A.Compose
        @param spatial_transform: Spatial transformations
        @type pixel_transform: A.Compose
        @param pixel_transform: Pixel transformations
        @type resize_transform: A.Compose
        @param resize_transform: Resize transformations
        @type targets: Dict[str, str]
        @param targets: Dictionary mapping albumentations
            target names to their albumentations types.
            Example: C{{"detection_boundingbox": "bboxes"}}
        @type targets_to_tasks: Dict[str, str]
        @param targets_to_tasks: Dictionary mapping
            albumenation target names to their respective LDF task names.
            Example: C{{"detection_boundingbox": "detection/boundingbox"}}
            This is necessary because the albumentations target names
            have to be valid python identifiers as opposed to LDF task names.
        @type special_tasks: SpecialTargets
        @param special_tasks: Dictionary containing the names of tasks
            that require special treatment during augmentation.
            The dictionary must contain the following keys:
                - metadata: Set of metadata tasks
                - classification: Set of classification tasks
                - arrays: Set of array tasks
                - instance_masks: Set of instance segmentation tasks
        """
        self.image_size = (height, width)
        self.special_tasks = special_tasks
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
    ) -> "AlbumentationsEngine":
        if keep_aspect_ratio:
            resize = LetterboxResize(height=height, width=width)
        else:
            resize = A.Resize(height=height, width=width)

        if is_validation_pipeline:
            config = [a for a in config if a["name"] == "Normalize"]

        main_task_names = set()
        alb_targets = {}
        alb_targets_to_tasks = {}
        special_tasks: SpecialTargets = {
            "metadata": set(),
            "classification": set(),
            "instance_masks": set(),
            "arrays": set(),
        }
        instance_segmentation_targets = set()
        for task, task_type in targets.items():
            if task_type == "array":
                logger.warning(
                    "Array task detected. The 'array' task can contain "
                    "arbitrary data so it cannot be properly augmented. "
                    "The only applied transformation is discarding arrays "
                    "associated with bboxes falling outside the image."
                )
                special_tasks["arrays"].add(task)
                continue
            elif task_type == "classification":
                special_tasks["classification"].add(task)
                continue
            elif task_is_metadata(task):
                special_tasks["metadata"].add(task)
                logger.warning(
                    "Metadata labels detected. Metadata labels can contain "
                    "arbitrary data so they cannot be properly augmented. "
                    "The only applied transformation is discarding metadata "
                    "associated with bboxes falling outside the image."
                )
                continue

            if task_type in {"segmentation", "instance_segmentation"}:
                if task_type == "instance_segmentation":
                    special_tasks["instance_masks"].add(task)
                    instance_segmentation_targets.add(_task_to_target(task))

                task_type = "mask"

            elif task_type == "boundingbox":
                task_type = "bboxes"

            task_name = get_task_name(task)

            target_name = _task_to_target(task)
            alb_targets[target_name] = task_type
            alb_targets_to_tasks[target_name] = task
            main_task_names.add(task_name)

        def _get_params():
            return {
                "bbox_params": A.BboxParams(
                    format="albumentations", min_visibility=min_bbox_visibility
                ),
                "keypoint_params": A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
                "additional_targets": alb_targets,
            }

        pixel_augmentations = []
        spatial_augmentation = []
        batched_augmentations = []
        batch_size = 1
        for aug in config:
            curr_aug = cls._get_transform(aug)
            if isinstance(curr_aug, A.ImageOnlyTransform):
                pixel_augmentations.append(curr_aug)
            elif isinstance(curr_aug, BatchTransform):
                batch_size *= curr_aug.batch_size
                batched_augmentations.append(curr_aug)
            elif isinstance(curr_aug, A.DualTransform):
                spatial_augmentation.append(curr_aug)

        with warnings.catch_warnings(record=True):
            return cls(
                height=height,
                width=width,
                batch_transform=BatchCompose(
                    batched_augmentations,
                    instance_segmentation_targets=instance_segmentation_targets,
                    **_get_params(),
                ),
                spatial_transform=A.Compose(
                    spatial_augmentation, **_get_params()
                ),
                pixel_transform=A.Compose(pixel_augmentations),
                resize_transform=A.Compose([resize], **_get_params()),
                targets=alb_targets,
                targets_to_tasks=alb_targets_to_tasks,
                special_tasks=special_tasks,
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
        arrays = defaultdict(list)
        classification = defaultdict(list)
        for labels in data:
            for array_task in self.special_tasks["arrays"]:
                if array_task in labels:
                    arrays[array_task].append(labels[array_task])
            for metadata_task in self.special_tasks["metadata"]:
                if metadata_task in labels:
                    metadata[metadata_task].append(labels[metadata_task])
            for classification_task in self.special_tasks["classification"]:
                if classification_task in labels:
                    classification[classification_task].append(
                        labels[classification_task]
                    )

        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        metadata = {k: np.concatenate(v) for k, v in metadata.items()}
        classification = {
            k: np.clip(sum(v), 0, 1) for k, v in classification.items()
        }

        data, n_keypoints = self.preprocess(data)

        if self.batch_transform.transforms:
            transformed = self.batch_transform(data)
        else:
            transformed = data[0]

        # For batch augmentations we need to replace
        # missing labels with empty arrays so the
        # correct batch size is maintained.
        # After the transformation, we remove them
        # so they don't interfere with the rest of the
        # pipeline.
        for key in list(transformed.keys()):
            if transformed[key].size == 0:
                del transformed[key]

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
            transformed,
            metadata,
            classification,
            arrays,
            n_keypoints,
        )

    def preprocess_data(
        self, labels: Data, bbox_counters: Dict[str, int]
    ) -> Tuple[Data, Dict[str, int]]:
        img = labels.pop("image")
        height, width = img.shape[:2]
        data = {"image": img}
        n_keypoints = {}
        for task, task_type in self.targets.items():
            override_name = task
            task = self.targets_to_tasks[task]

            if task not in labels:
                data[override_name] = np.array([])
                continue

            array = labels[task]
            if task_type == "mask":
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
        return data, n_keypoints

    def preprocess(
        self, data: List[Data]
    ) -> Tuple[List[Data], Dict[str, int]]:
        batch_data = []
        bbox_counters = defaultdict(int)
        n_keypoints = {}

        for d in data:
            d, _n_keypoints = self.preprocess_data(d, bbox_counters)
            n_keypoints.update(_n_keypoints)
            for task, array in d.items():
                if task == "image":
                    continue
                if self.targets[task] == "bboxes":
                    bbox_counters[task] += array.shape[0]
            batch_data.append(d)
        return batch_data, n_keypoints

    def postprocess(
        self,
        data: Data,
        metadata: Data,
        classification: Data,
        arrays: Data,
        n_keypoints: Dict[str, int],
    ) -> LoaderOutput:
        out_labels = {}
        out_image = data.pop("image")
        image_height, image_width, _ = out_image.shape

        bboxes_orderings = {}
        targets = sorted(
            list(data.keys()), key=lambda x: self.targets.get(x) != "bboxes"
        )

        for target in targets:
            array = data[target]
            if array.size == 0:
                continue

            task = self.targets_to_tasks[target]
            task_name = get_task_name(task)
            task_type = self.targets[target]
            if task_type == "bboxes":
                out_labels[task], ordering = postprocess_bboxes(array)
                bboxes_orderings[task_name] = ordering

        for target in targets:
            array = data[target]
            if array.size == 0:
                continue

            task = self.targets_to_tasks[target]
            task_name = get_task_name(task)
            task_type = self.targets[target]

            if task_type == "mask":
                mask = postprocess_mask(array)
                if task in self.special_tasks["instance_masks"]:
                    if task_name in bboxes_orderings:
                        out_labels[task] = mask[bboxes_orderings[task_name]]
                else:
                    out_labels[task] = mask

            elif task_type == "keypoints" and task_name in bboxes_orderings:
                out_labels[task] = postprocess_keypoints(
                    array,
                    bboxes_orderings[task_name],
                    image_height,
                    image_width,
                    n_keypoints[target],
                )

        for task, value in metadata.items():
            task_name = get_task_name(task)
            out_labels[task] = value[
                bboxes_orderings.get(task_name, np.array([], dtype=np.uint8))
            ]
        for task, array in arrays.items():
            task_name = get_task_name(task)
            out_labels[task] = array[
                bboxes_orderings.get(task_name, np.array([], dtype=np.uint8))
            ]

        out_labels.update(**classification)
        return out_image, out_labels

    @staticmethod
    def _get_transform(config: ConfigItem) -> A.BasicTransform:
        name = config["name"]
        params = config.get("params", {})
        if hasattr(A, name):
            return getattr(A, name)(**params)
        return TRANSFORMATIONS.get(name)(**params)  # type: ignore


def _task_to_target(task: str) -> str:
    target = task.replace("/", "_").replace("-", "_")
    assert target.isidentifier()
    return target
