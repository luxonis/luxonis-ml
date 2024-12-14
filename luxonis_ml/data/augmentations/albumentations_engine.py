import logging
import warnings
from collections import defaultdict
from math import prod
from typing import Any, Dict, Iterator, List, Literal, Set, Tuple

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
TargetType: TypeAlias = Literal[
    "array",
    "classification",
    "mask",
    "instance_mask",
    "bboxes",
    "keypoints",
    "metadata",
]


class AlbumentationsEngine(AugmentationEngine, register_name="albumentations"):
    """Augmentation engine using the Albumentations library under the
    hood.

    Configuration Format
    ====================

    The configuration is a list of dictionaries, where the dictionaries
    contain the name of the transformation and optionally its parameters.

    The name must be either a valid name of an Albumentations
    transformation (accessible under the `albumentations` namespace),
    or a name of a custom transformation registered in the
    `TRANSFORMATIONS` registry.

    Example::

        [
            {
                "name": "Affine",
                "params": {
                    "rotate": 30,
                    "scale": 0.5,
                    "p": 0.3,
                },
            },
            {
                "name": "MixUp",
                "alpha": [0.3, 0.7],
                "p": 0.5,
            },
        ]

    Transformation Order
    ====================

    The order of transformations provided in the configuration is not
    guaranteed to be preserved. The transformations are divided into
    the following groups and are applied in the same order:

        1. batch transformations: Subclasses of L{BatchTransform}.
            `additional_targets`. These transformations contain
            all the LDF task types, not only those supported
            by Albumentations.

        2. spatial transformations: Subclasses of `A.DualTransform`.
            These transformations will only receive native Albumentations
            task types: 'mask', 'bboxes', and 'keypoints'. 'instance_mask'
            tasks are changed to 'mask'.

        3. pixel transformations: Subclasses of `A.ImageOnlyTransform`.
            These transformations act only on the image.

        4. custom transformations: Subclasses of `A.BasicTransform`,
            but not subclasses of any of more specific base classes above.
            These transformations are applied at the very end and receive
            all the LDF task types and the full set of labels.
            That means they can be implemented to handle
            specific metadata or array tasks.

    Batch transformations are always applied first, followed by spatial
    transformations, and finally pixel transformations.

    Supported Augmentations
    =======================

    Native Augmentations
    --------------------

    All augmentations provided by the Albumentations library are supported.

    Supported Batch Augmentations
    -----------------------------

    MixUp
    ~~~~~

    MixUp is a data augmentation technique that blends 2 source
    images into a single image using a weight coefficient alpha.

    Mosaic4
    ~~~~~~~

    Mosaic4 transformation combines 4 images into a single image
    by placing them in a 2x2 grid.

    Augmenting Unsupported Tasks
    ============================

    Albumentations do not natively support all the tasks supported
    by Luxonis Data Format. This sections describes how
    unsupported tasks are handled.

    Note that the following applies only to officially supported
    augmentations. Custom augmentations can be implemented to handle
    arbitrary tasks.

    Classification
    --------------

    Classification tasks can be properly augmented only for multi-label
    tasks, where each class is tied to a bounding box. In such cases,
    the classes belonging to bboxes falling outside the image are removed.
    In other cases, the classification annotation is kept as is.

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

    (Not yet implemented)

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

    @override
    def __init__(
        self,
        height: int,
        width: int,
        targets: Dict[str, TaskType],
        config: Iterator[ConfigItem],
        keep_aspect_ratio: bool = True,
        is_validation_pipeline: bool = False,
        min_bbox_visibility: float = 0.001,
    ):
        self.targets: Dict[str, TargetType] = {}
        self.targets_to_tasks = {}
        self.image_size = (height, width)

        for task, task_type in targets.items():
            target_name = self.task_to_target_name(task)

            if task_type == "array":
                target_type = "array"
                logger.warning(
                    "Array task detected. The 'array' task can contain "
                    "arbitrary data so it cannot be properly augmented. "
                    "The only applied transformation is discarding arrays "
                    "associated with bboxes falling outside the image."
                )

            elif task_type == "classification":
                target_type = "classification"
                logger.warning(
                    "Classification task detected. Classification tasks "
                    "can be properly augmented only for multi-label tasks, "
                    "where each class is tied to a bounding box. "
                    "In such cases, the classes belonging to bboxes falling "
                    "outside the image are removed. In other cases, "
                    "the annotation is kept as is."
                )

            elif task_is_metadata(task):
                target_type = "metadata"
                logger.warning(
                    "Metadata labels detected. Metadata labels can contain "
                    "arbitrary data so they cannot be properly augmented. "
                    "The only applied transformation is discarding metadata "
                    "associated with bboxes falling outside the image."
                )

            elif task_type == "segmentation":
                target_type = "mask"

            elif task_type == "instance_segmentation":
                target_type = "instance_mask"

            elif task_type == "boundingbox":
                target_type = "bboxes"

            elif task_type == "keypoints":
                target_type = "keypoints"

            else:
                raise ValueError(
                    f"Unsupported task type: '{task_type}'. "
                    f"Only 'array', 'classification', 'segmentation', "
                    f"'instance_segmentation', 'boundingbox', "
                    f"'keypoints', and 'metadata' are supported."
                )

            self.targets[target_name] = target_type
            self.targets_to_tasks[target_name] = task

        # Only 'mask', 'bbox', and 'keypoints'.
        # Instance masks are handled as masks for
        # official Albumentations transformations.
        pure_albumentations_targets = {
            target_name: target_type
            if target_type != "instance_mask"
            else "mask"
            for target_name, target_type in self.targets.items()
            if target_type not in self.special_tasks
        }

        pixel_transforms = []
        spatial_transforms = []
        batch_transforms = []
        custom_transforms = []

        if is_validation_pipeline:
            config = (a for a in config if a["name"] == "Normalize")

        for config_item in config:
            transform = self.create_transformation(config_item)

            if isinstance(transform, A.ImageOnlyTransform):
                pixel_transforms.append(transform)
            elif isinstance(transform, BatchTransform):
                batch_transforms.append(transform)
            elif isinstance(transform, A.DualTransform):
                spatial_transforms.append(transform)
            elif isinstance(transform, A.BasicTransform):
                custom_transforms.append(transform)
            else:
                raise ValueError(
                    f"Unsupported transformation type: '{transform.__name__}'. "
                    f"Only subclasses of `A.BasicTransform` are allowed. "
                )

        if keep_aspect_ratio:
            resize = LetterboxResize(height=height, width=width)
        else:
            resize = A.Resize(height=height, width=width)

        def get_params(*, is_custom: bool = False) -> Dict[str, Any]:
            return {
                "bbox_params": A.BboxParams(
                    format="albumentations", min_visibility=min_bbox_visibility
                ),
                "keypoint_params": A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
                "additional_targets": self.targets
                if is_custom
                else pure_albumentations_targets,
            }

        with warnings.catch_warnings(record=True):
            self.batch_transform = BatchCompose(
                batch_transforms, **get_params(is_custom=True)
            )
            self.spatial_transform = wrap_transform(
                A.Compose(spatial_transforms, **get_params())
            )
            self.pixel_transform = wrap_transform(
                A.Compose(pixel_transforms), is_pixel=True
            )
            self.resize_transform = wrap_transform(
                A.Compose([resize], **get_params())
            )
            self.custom_transform = wrap_transform(
                A.Compose(custom_transforms, **get_params(is_custom=True))
            )

    @property
    @override
    def batch_size(self) -> int:
        return self.batch_transform.batch_size

    @property
    def special_tasks(self) -> Set[str]:
        return {"metadata", "array", "classification"}

    @override
    def apply(self, input_batch: List[LoaderOutput]) -> LoaderOutput:
        metadata, arrays, classification = self.extract_special_tasks(
            input_batch
        )

        data, n_keypoints = self.preprocess_batch(input_batch)

        transformed = self.batch_transform(data)

        for target_name in list(transformed.keys()):
            if transformed[target_name].size == 0:
                del transformed[target_name]

        transformed = self.spatial_transform(**transformed)

        transformed_size = transformed["image"].shape[:2]

        if transformed_size != self.image_size:
            transformed_size = prod(transformed_size)
            target_size = prod(self.image_size)

            if transformed_size > target_size:
                transformed = self.resize_transform(**transformed)
                transformed = self.pixel_transform(**transformed)
            else:
                transformed = self.pixel_transform(**transformed)
                transformed = self.resize_transform(**transformed)
        else:
            transformed = self.pixel_transform(**transformed)

        return self.postprocess(
            transformed, metadata, classification, arrays, n_keypoints
        )

    def preprocess_batch(
        self, labels_batch: List[LoaderOutput]
    ) -> Tuple[List[Data], Dict[str, int]]:
        """Preprocess a batch of labels.

        @type labels_batch: List[Data]
        @param labels_batch: List of dictionaries mapping task names to
            the annotations as C{np.ndarray}
        @rtype: Tuple[List[Data], Dict[str, int]]
        @return: Tuple containing the preprocessed data and a dictionary
            mapping task names to the number of keypoints for that task.
        """
        data_batch = []
        bbox_counters = defaultdict(int)
        n_keypoints = {}

        for image, labels in labels_batch:
            data = {"image": image}
            height, width, _ = image.shape
            for target_name, target_type in self.targets.items():
                if target_type in self.special_tasks:
                    continue

                task = self.targets_to_tasks[target_name]

                if task not in labels:
                    data[target_name] = np.array([])
                    continue

                array = labels[task]

                if target_type in {"mask", "instance_mask"}:
                    data[target_name] = preprocess_mask(array)

                elif target_type == "bboxes":
                    data[target_name] = preprocess_bboxes(
                        array, bbox_counters[target_name]
                    )
                    bbox_counters[target_name] += data[target_name].shape[0]

                elif target_type == "keypoints":
                    n_keypoints[target_name] = array.shape[1] // 3
                    data[target_name] = preprocess_keypoints(
                        array, height, width
                    )

            data_batch.append(data)

        return data_batch, n_keypoints

    def postprocess(
        self,
        data: Data,
        metadata: Data,
        classification: Data,
        arrays: Data,
        n_keypoints: Dict[str, int],
    ) -> LoaderOutput:
        """Postprocess the augmented data back to LDF format.

        Discards labels associated with bboxes that are outside the
        image.

        @type data: Data
        @param data: Dictionary mapping task names to the annotations as
            C{np.ndarray}
        @type metadata: Data
        @param metadata: Dictionary mapping metadata task names to the
            annotations as C{np.ndarray}
        @type classification: Data
        @param classification: Dictionary mapping classification task
            names to the annotations as C{np.ndarray}
        @type arrays: Data
        @param arrays: Dictionary mapping array task names to the
            annotations as C{np.ndarray}
        @type n_keypoints: Dict[str, int]
        @param n_keypoints: Dictionary mapping task names to the number
            of keypoints for that task.
        @rtype: LoaderOutput
        @return: Tuple containing the augmented image and the labels.
        """
        out_labels = {}
        out_image = data.pop("image")
        image_height, image_width, _ = out_image.shape

        bboxes_orderings = {}

        for target_name, target_type in self.targets.items():
            if target_name not in data:
                continue

            array = data[target_name]
            if array.size == 0:
                continue

            task = self.targets_to_tasks[target_name]
            task_name = get_task_name(task)

            if target_type == "bboxes":
                out_labels[task], ordering = postprocess_bboxes(array)
                bboxes_orderings[task_name] = ordering

        for target_name, target_type in self.targets.items():
            if target_name not in data:
                continue

            array = data[target_name]
            if array.size == 0:
                continue

            task = self.targets_to_tasks[target_name]
            task_name = get_task_name(task)

            if target_type == "mask":
                out_labels[task] = postprocess_mask(array)

            elif (
                target_type == "instance_mask"
                and task_name in bboxes_orderings
            ):
                mask = postprocess_mask(array)
                out_labels[task] = mask[bboxes_orderings[task_name]]

            elif target_type == "keypoints" and task_name in bboxes_orderings:
                out_labels[task] = postprocess_keypoints(
                    array,
                    bboxes_orderings[task_name],
                    image_height,
                    image_width,
                    n_keypoints[target_name],
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
    def create_transformation(config: ConfigItem) -> A.BasicTransform:
        name = config["name"]
        params = config.get("params", {})
        if hasattr(A, name):
            return getattr(A, name)(**params)
        return TRANSFORMATIONS.get(name)(**params)  # type: ignore

    def extract_special_tasks(
        self, input_batch: List[LoaderOutput]
    ) -> Tuple[Data, Data, Data]:
        metadata = defaultdict(list)
        arrays = defaultdict(list)
        classification = defaultdict(list)
        for _, labels in input_batch:
            for target_name, target_type in self.targets.items():
                task = self.targets_to_tasks[target_name]
                if task not in labels:
                    continue

                if target_type == "array":
                    arrays[task].append(labels.pop(task))

                if target_type == "metadata":
                    metadata[task].append(labels.pop(task))

                if target_type == "classification":
                    classification[task].append(labels.pop(task))

        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        metadata = {k: np.concatenate(v) for k, v in metadata.items()}
        classification = {
            k: np.clip(sum(v), 0, 1) for k, v in classification.items()
        }
        return metadata, arrays, classification

    @staticmethod
    def task_to_target_name(task: str) -> str:
        """Returns the target name from a task. Replaces '/' and '-'
        with '_' and checks if the target name is a valid identifier.

        Example:

            >>> task_to_target_name("task-name/segmentation")
            'task_name_segmentation'
            >>> task_to_target_name("task/metadata/name")
            'task_metadata_name'

        @type task: str
        @param task: The task.
        @rtype: str
        @return: The target name.
        """
        target = task.replace("/", "_").replace("-", "_")
        assert target.isidentifier()
        return target


def wrap_transform(transform: A.BaseCompose, is_pixel: bool = False):
    def apply_transform(**data: np.ndarray) -> Data:
        if not transform.transforms:
            return data

        if is_pixel:
            data["image"] = transform(image=data["image"])["image"]
            return data

        return transform(**data)

    return apply_transform
