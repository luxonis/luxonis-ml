import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from math import prod
from typing import Any, Literal, TypeAlias, cast

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from loguru import logger
from typing_extensions import override

from luxonis_ml.data.utils.task_utils import get_task_name, task_is_metadata
from luxonis_ml.typing import ConfigItem, LoaderMultiOutput, Params

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

Data: TypeAlias = dict[str, np.ndarray]
TargetType: TypeAlias = Literal[
    "image",
    "array",
    "classification",
    "mask",
    "instance_mask",
    "bboxes",
    "keypoints",
    "metadata",
]


class AlbumentationConfigItem(ConfigItem):
    use_for_resizing: bool = False


class AlbumentationsEngine(AugmentationEngine, register_name="albumentations"):
    """Augmentation engine using the Albumentations library under the
    hood.

    Configuration Format
    ====================

    The configuration is a list of dictionaries, where the dictionaries
    contain the name of the transformation and optionally its parameters.
    It can also contain a boolean flag C{use_for_resizing} that indicates
    whether the transformation should be used for resizing. If no resizing
    augmentation is provided, the engine will use either L{A.Resize} or
    L{LetterboxResize} depending on the C{keep_aspect_ratio} parameter.

    The name must be either a valid name of an Albumentations
    transformation (accessible under the C{albumentations} namespace),
    or a name of a custom transformation registered in the
    L{TRANSFORMATIONS} registry.

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
                "params": {
                    "alpha": [0.3, 0.7],
                    "p": 0.5,
                },
            },
            {
                "name": "CustomResize",
                "use_for_resizing": True,
            },
        ]

    Transformation Order
    ====================

    The order of transformations provided in the configuration is not
    guaranteed to be preserved. The transformations are divided into
    the following groups and are applied in the same order:

        1. batch transformations: Subclasses of L{BatchTransform}.

        2. spatial transformations: Subclasses of `A.DualTransform`.

        3. custom transformations: Subclasses of `A.BasicTransform`,
            but not subclasses of more specific base classes above.

        4. pixel transformations: Subclasses of `A.ImageOnlyTransform`.
            These transformations act only on the image.


    Supported Augmentations
    =======================

    Official Augmentations
    ----------------------

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

    Custom Augmentations
    ====================

    Custom augmentations can be implemented by creating a subclass
    of L{A.BasicTransform} and registering it in the L{TRANSFORMATIONS}
    registry.

    Possible target types that the augmentation can receive are:
        - 'image': The image. All augmentations should usually
            support this target. For subclasses of L{A.ImageOnlyTransform}
            or L{A.DualTransform} this means overriding the C{apply} method.

        - 'bboxes': Bounding boxes. For subclasses of L{A.DualTransform},
            this means overriding the L{apply_to_bboxes} method.

        - 'keypoints': Keypoints. For subclasses of L{A.DualTransform},
            this means overriding the L{apply_to_keypoints} method.

        - 'mask': Segmentation masks. For subclasses of L{A.DualTransform},
            this means overriding the L{apply_to_mask} method.

        - 'instance_mask': Instance segmentation masks.
            For subclasses of L{BatchTransform}, this means overriding
            the C{apply_to_instance_mask} method.

            Subclasses of L{A.DualTransform} do not support this target,
            instance masks are treated as regular masks instead.

            Custom augmentations can support instance masks by
            implementing their own logic for handling them and overriding
            the C{targets} property to include the C{instance_mask} target.

        - 'array': Arbitrary arrays. Can only be supported by custom
            augmentations by implementing their own logic and adding
            the C{array} target to the C{targets} property.

        - 'metadata': Metadata labels.
            Same situation as with the 'array' type.

        - 'classification': One-hot encoded multi-task classification labels.
            Same situation as with the 'array' type.

    Example::

        class CustomArrayAugmentation(A.BasicTransform):

            @property
            @override
            def targets(self) -> Dict[str, Any]:
                return {
                    "image": self.apply,
                    "array": self.apply_to_array,
                }

            def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
                ...

            def apply_to_array(
                self, array: np.ndarray, **kwargs
            ) -> np.ndarray:
                ...
    """

    def _check_augmentation_warnings(
        self, config_item: dict[str, Any], available_target_types: set
    ) -> None:
        augmentation_name = config_item["name"]

        if "keypoints" in available_target_types and augmentation_name in [
            "HorizontalFlip",
            "VerticalFlip",
            "Transpose",
        ]:
            logger.warning(
                f"Using '{augmentation_name}' with keypoints."
                "If your dataset contains symmetric keypoints (e.g. left/right arms),"
                "you should use our custom HorizontalSymetricKeypointsFlip,"
                "VerticalSymetricKeypointsFlip, or TransposeSymmetricKeypoints"
                "to ensure keypoints are correctly reordered."
            )

    @override
    def __init__(
        self,
        height: int,
        width: int,
        targets: dict[str, str],
        n_classes: dict[str, int],
        source_names: list[str],
        config: Iterable[Params],
        keep_aspect_ratio: bool = True,
        is_validation_pipeline: bool = False,
        min_bbox_visibility: float = 0.0,
        seed: int | None = None,
        bbox_area_threshold: float = 0.0004,
    ):
        self.targets: dict[str, TargetType] = {}
        self.target_names_to_tasks = {}
        self.n_classes = n_classes
        self.image_size = (height, width)
        self.source_names = source_names
        self.bbox_area_threshold = bbox_area_threshold

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
            self.target_names_to_tasks[target_name] = task

        for source_name in source_names:
            self.targets[source_name] = "image"

        # Necessary for official Albumentations transforms.
        targets_without_instance_mask = {
            target_name: target_type
            if target_type != "instance_mask"
            else "mask"
            for target_name, target_type in self.targets.items()
        }

        pixel_transforms = []
        spatial_transforms = []
        batch_transforms = []
        custom_transforms = []
        resize_transform = None

        if is_validation_pipeline:
            config = (a for a in config if a["name"] == "Normalize")

        available_target_types = set(self.targets.values())

        for config_item in config:
            self._check_augmentation_warnings(
                config_item, available_target_types
            )
            cfg = AlbumentationConfigItem(**config_item)  # type: ignore
            transform = self.create_transformation(cfg)

            if cfg.use_for_resizing:
                logger.info(f"Using '{cfg.name}' for resizing.")
                if resize_transform is not None:
                    raise ValueError(
                        "Only one resizing augmentation can be provided."
                    )
                resize_transform = transform

            elif isinstance(transform, A.ImageOnlyTransform):
                pixel_transforms.append(transform)
            elif isinstance(transform, BatchTransform):
                batch_transforms.append(transform)
            elif isinstance(transform, (A.DualTransform, A.BaseCompose)):
                spatial_transforms.append(transform)
            elif isinstance(transform, A.BasicTransform):
                custom_transforms.append(transform)
            else:
                raise ValueError(
                    f"Unsupported transformation type: '{type(transform).__name__}'. "
                    f"Only subclasses of `A.BasicTransform` "
                    f"or `A.BaseCompose` are allowed. "
                )

        wrapped_spatial_ops: TransformsSeqType = []
        if "keypoints" in targets.values():
            for op in spatial_transforms:
                wrapped_spatial_ops.append(op)
                wrapped_spatial_ops.append(
                    A.Lambda(
                        image=lambda img, **kw: img,
                        keypoints=self._mark_invisible_keypoints,
                        p=1.0,
                    )
                )
        else:
            wrapped_spatial_ops = spatial_transforms

        if resize_transform is None:
            if keep_aspect_ratio:
                resize_transform = LetterboxResize(height=height, width=width)
            else:
                resize_transform = A.Resize(height=height, width=width)

        def get_params(is_custom: bool = False) -> dict[str, Any]:
            return {
                "bbox_params": A.BboxParams(
                    format="albumentations", min_visibility=min_bbox_visibility
                ),
                "keypoint_params": A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
                "additional_targets": (
                    self.targets
                    if is_custom
                    else targets_without_instance_mask
                ),
                "seed": seed,
            }

        # Warning issued when "bbox_params" or "keypoint_params"
        # are provided to a compose with transformations that
        # do not use them. We don't care about these warnings.
        with warnings.catch_warnings(record=True):
            self.batch_transform = BatchCompose(
                batch_transforms, **get_params(is_custom=True)
            )
            self.spatial_transform = wrap_transform(
                A.Compose(wrapped_spatial_ops, **get_params())
            )
            self.pixel_transform = wrap_transform(
                A.Compose(pixel_transforms),
                is_pixel=True,
                source_names=source_names,
            )
            self.resize_transform = wrap_transform(
                A.Compose([resize_transform], **get_params())
            )
            self.custom_transform = wrap_transform(
                A.Compose(custom_transforms, **get_params(is_custom=True))
            )

    @property
    @override
    def batch_size(self) -> int:
        return self.batch_transform.batch_size

    @override
    def apply(self, input_batch: list[LoaderMultiOutput]) -> LoaderMultiOutput:
        data_batch, n_keypoints = self.preprocess_batch(input_batch)

        data = self.batch_transform(data_batch)

        for target_name in list(data.keys()):
            value = data[target_name]
            if isinstance(value, np.ndarray) and value.size == 0:
                del data[target_name]

        data = self.spatial_transform(**data)
        data = self.custom_transform(**data)

        transformed_size = data["image"].shape[:2]

        if transformed_size != self.image_size:
            transformed_size = prod(transformed_size)
            target_size = prod(self.image_size)

            if transformed_size > target_size:
                data = self.resize_transform(**data)
                data = self.pixel_transform(**data)
            else:
                data = self.pixel_transform(**data)
                data = self.resize_transform(**data)
        else:
            data = self.pixel_transform(**data)

        return self.postprocess(data, n_keypoints)

    def preprocess_batch(
        self, labels_batch: list[LoaderMultiOutput]
    ) -> tuple[list[Data], dict[str, int]]:
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

        for image_dict, labels in labels_batch:
            data = {}

            key = next(iter(image_dict))
            data["_original_image_key"] = key
            for source_name, img in image_dict.items():
                if source_name == key:
                    data["image"] = img
                else:
                    data[source_name] = img

            sample_img = next(iter(image_dict.values()))
            height, width = sample_img.shape[:2]

            for target_name, target_type in self.targets.items():
                if target_name not in self.target_names_to_tasks:
                    continue

                task = self.target_names_to_tasks[target_name]

                if task not in labels:
                    if target_type == "mask":
                        data[target_name] = np.empty(
                            (
                                0,
                                0,
                                self.n_classes[
                                    self.target_names_to_tasks[target_name]
                                ],
                            )
                        )
                    elif target_type == "classification":
                        data[target_name] = np.zeros(
                            self.n_classes[
                                self.target_names_to_tasks[target_name]
                            ]
                        )
                    else:
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
                else:
                    data[target_name] = array

            data_batch.append(data)

        return data_batch, n_keypoints

    def postprocess(
        self, data: Data, n_keypoints: dict[str, int]
    ) -> LoaderMultiOutput:
        """Postprocess the augmented data back to LDF format.

        Discards labels associated with bboxes that are outside the
        image.

        @type data: Data
        @param data: Dictionary mapping task names to the annotations as
            C{np.ndarray}
        @type n_keypoints: Dict[str, int]
        @param n_keypoints: Dictionary mapping task names to the number
            of keypoints for that task.
        @rtype: LoaderMultiOutput
        @return: Tuple containing the augmented image dict and the
            labels.
        """
        out_labels = {}
        out_image_dict = {}

        image_keys = [k for k in data if k in ["image", *self.source_names]]

        for key in image_keys:
            img = data.pop(key)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            restored_key = (
                data["_original_image_key"] if key == "image" else key
            )
            out_image_dict[restored_key] = img

        sample_img = next(iter(out_image_dict.values()))
        image_height, image_width = sample_img.shape[:2]

        bboxes_indices = {}

        for target_name, target_type in self.targets.items():
            if target_name not in data:
                continue

            array = data[target_name]
            if array.size == 0:
                continue

            task = self.target_names_to_tasks[target_name]
            task_name = get_task_name(task)

            if target_type == "bboxes":
                out_labels[task], index = postprocess_bboxes(
                    array, self.bbox_area_threshold
                )
                bboxes_indices[task_name] = index

        for target_name, target_type in self.targets.items():
            if target_name not in data:
                continue

            array = data[target_name]
            if array.size == 0:
                continue

            task = self.target_names_to_tasks[target_name]
            task_name = get_task_name(task)

            if task_name not in bboxes_indices:
                if "bboxes" in self.targets.values():
                    bbox_ordering = np.array([], dtype=int)
                elif target_type == "keypoints":
                    bbox_ordering = np.arange(
                        array.shape[0] // n_keypoints[target_name]
                    )
                else:
                    bbox_ordering = np.arange(array.shape[0])
            else:
                bbox_ordering = bboxes_indices[task_name]

            if target_type == "mask":
                out_labels[task] = postprocess_mask(array)

            elif target_type == "instance_mask":
                masks = postprocess_mask(array)
                out_labels[task] = masks[bbox_ordering]

            elif target_type == "keypoints":
                out_labels[task] = postprocess_keypoints(
                    array,
                    bbox_ordering,
                    image_height,
                    image_width,
                    n_keypoints[target_name],
                )
            elif target_type in {"array", "metadata"}:
                out_labels[task] = array[bbox_ordering]

            elif target_type == "classification":
                out_labels[task] = array

        return out_image_dict, out_labels

    @staticmethod
    def _mark_invisible_keypoints(
        keypoints: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        keypoints: np.ndarray of shape (N,6) columns = [x, y, z, a, s, v]
        Zeroes out the visibility (last) column if (x,y) is out of image bounds.
        """
        shape = kwargs.get("shape")
        if shape is None:
            raise ValueError(
                "Shape must be provided in kwargs to mark invisible keypoints."
            )
        h, w = shape[:2]
        kps = keypoints.copy()
        xs, ys = kps[:, 0], kps[:, 1]
        oob = (xs < 0) | (ys < 0) | (xs >= w) | (ys >= h)
        kps[oob, -1] = 0.0
        return kps

    @staticmethod
    def create_transformation(
        config: AlbumentationConfigItem,
    ) -> A.BasicTransform:
        params = config.params.copy()

        # Recursively handle nested transform compositions
        # (for example: OneOf, SomeOf, Sequential)
        if "transforms" in params and isinstance(params["transforms"], list):
            nested_transforms = []
            for item in params["transforms"]:
                if isinstance(item, dict) and "name" in item:
                    nested_cfg = AlbumentationConfigItem(
                        name=str(item["name"]),
                        params=cast(Params, item.get("params", {})),
                        use_for_resizing=bool(
                            item.get("use_for_resizing", False)
                        ),
                    )
                    transform = AlbumentationsEngine.create_transformation(
                        nested_cfg
                    )
                    if isinstance(transform, BatchTransform):
                        raise ValueError(
                            f"Batch transform '{item['name']}' cannot be "
                            f"nested inside '{config.name}'. "
                            f"Batch transforms (e.g Mosaic4 and MixUp) "
                            f"require multiple images and must be used "
                            f"as top-level augmentations."
                        )
                    nested_transforms.append(transform)
                else:
                    raise ValueError(
                        f"Invalid nested transform configuration: {item}"
                    )
            params["transforms"] = nested_transforms

        if hasattr(A, config.name):
            return getattr(A, config.name)(**params)
        return TRANSFORMATIONS.get(config.name)(**params)  # type: ignore

    @staticmethod
    def task_to_target_name(task: str) -> str:
        target = task.replace("/", "_").replace("-", "_")
        assert target.isidentifier()
        return target


def wrap_transform(
    transform: A.BaseCompose,
    is_pixel: bool = False,
    source_names: list[str] | None = None,
) -> Callable[..., Data]:
    def apply_transform(**data: np.ndarray) -> Data:
        if not transform.transforms:
            return data

        if is_pixel:
            if source_names is None:
                raise ValueError(
                    "source_names must be provided for pixel transformations."
                )
            replay_transform = A.ReplayCompose(transform.transforms)

            result = replay_transform(image=data["image"])
            data["image"] = result["image"]

            replay = result["replay"]
            for source_name in source_names:
                if source_name == "image" or source_name not in data:
                    continue
                img = data[source_name]
                if img.ndim == 3:
                    data[source_name] = A.ReplayCompose.replay(
                        replay, image=img
                    )["image"]

            return data

        original_key = data.pop("_original_image_key", None)
        transformed = transform(**data)
        if original_key is not None:
            transformed["_original_image_key"] = original_key

        return transformed

    return apply_transform
