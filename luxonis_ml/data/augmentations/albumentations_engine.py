import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from math import prod
from typing import Any, Literal, TypeAlias, cast

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from loguru import logger
from pydantic import Field
from typing_extensions import override

from luxonis_ml.data.utils.task_utils import get_task_name, task_is_metadata
from luxonis_ml.typing import ConfigItem, LoaderMultiOutput, Params
from luxonis_ml.utils import deprecated

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
    """Configuration item for `AlbumentationsEngine`.

    Attributes:
        name: Name of the transformation. Must be either a valid name of an
            `Albumentations`_ transformation, or a name of a custom
            transformation registered in the `TRANSFORMATIONS` registry.
        params: Parameters for the transformation.
        use_for_resizing: Whether this transformation is eligible
            to be used for resizing.
        apply_on_stages: List of pipeline stages to apply
            this transformation on. Valid stages are
            ``"train"``, ``"val"``, and ``"test"``.
            By default, transformations are applied
            only during the ``"train"`` stage.

    .. _Albumentations: https://albumentations.ai/explore/

    """

    use_for_resizing: bool = False
    apply_on_stages: list[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["train"]
    )


class AlbumentationsEngine(AugmentationEngine, register_name="albumentations"):
    r"""Augmentation engine backed by Albumentations.

    .. contents:: Table of Contents

    Configuration Format
    ====================

    The configuration contains a list of transformations,
    each specified by its name and optional parameters as
    described in the `AlbumentationConfigItem` schema.

    The name must be either a valid name of an `Albumentations`_
    transformation, or a name of a custom transformation registered in the
    `TRANSFORMATIONS` registry.

    For example:

    .. python::

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
    the following groups and are applied in this order:

        1. Batch transformations: Subclasses of `BatchTransform`.

        2. Spatial transformations: Subclasses of `A.DualTransform`_.

        3. Custom transformations: Subclasses of `A.BasicTransform`_,
           but not subclasses of more specific base classes above.

        4. Pixel transformations: Subclasses of `A.ImageOnlyTransform`_.
           These transformations act only on the image.


    Supported Augmentations
    =======================

    Albumentations Augmentations
    ----------------------------

    All augmentations provided by the Albumentations library are supported.

    Batch Augmentations
    -----------------------------

    `MixUp`
    ~~~~~~~

    MixUp is a data augmentation technique that blends 2 source
    images into a single image using a weight coefficient alpha.

    `Mosaic4`
    ~~~~~~~~~

    Mosaic4 transformation combines 4 images into a single image
    by placing them in a :math:`2 \times 2` grid.

    Augmenting Unsupported Types
    ============================

    Albumentations does not natively support all label types supported by
    Luxonis Data Format. This section describes how unsupported types are
    handled.

    Note that the following applies only to officially supported
    augmentations. Custom augmentations can be implemented to handle
    arbitrary types.

    Classification
    --------------

    Classification can be properly augmented only for multi-label
    tasks, where each class is tied to a bounding box. In such cases,
    the classes belonging to bboxes falling outside the image are removed.
    In other cases, the classification annotation is kept as is.

    Metadata
    --------

    Metadata labels can contain arbitrary data and their semantics are
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

    Oriented bounding boxes are of shape :math:`\left(N, 5\right)` where
    the last element of each row contains the angle of the bbox.
    This format is not supported by Albumentations, however,
    Albumentations support angle to be a part of the keypoints.
    So, the oriented bounding boxes are split into regular
    bounding boxes and a set of keypoints that represent
    the center of the bbox and contain the angle as the third coordinate.

    Both the keypoints and the bboxes are augmented separately.
    At the end, the angle is extracted from the keypoints and added
    back to the bounding boxes. The keypoints are discarded.

    Custom Augmentations
    ====================

    Custom augmentations can be implemented by creating a subclass of
    `A.BasicTransform`_ and registering it in the `TRANSFORMATIONS`
    registry.

    Possible target types that the augmentation can receive are:

        - ``"image"``:

                The image. All augmentations should usually
                support this target. For subclasses of `A.ImageOnlyTransform`_
                or `A.DualTransform`_ this means overriding ``apply``.

        - ``"bboxes"``:

                Bounding boxes. For subclasses of `A.DualTransform`_,
                this means overriding ``apply_to_bboxes``.

        - ``"keypoints"``:

                Keypoints. For subclasses of `A.DualTransform`_,
                this means overriding ``apply_to_keypoints``.

        - ``"mask"``:

                Segmentation masks. For subclasses of
                `A.DualTransform`_, this means overriding ``apply_to_mask``.

        - ``"instance_mask"``:

                Instance segmentation masks.
                For subclasses of `BatchTransform`, this means overriding
                ``apply_to_instance_mask``.

                Subclasses of `A.DualTransform`_ do not support this target;
                instance masks are treated as regular masks instead.

                Custom augmentations can support instance masks by implementing
                their own logic for handling them and overriding the ``targets``
                property to include the ``"instance_mask"`` target.

        - ``"array"``:

                Arbitrary arrays. Can only be supported by custom
                augmentations by implementing their own logic and adding
                the ``"array"`` target to the ``targets`` property.

        - ``"metadata"``:

                Metadata labels.
                Same situation as with the ``"array"`` type.

        - ``"classification"``:

                One-hot encoded multi-task classification
                labels. Same situation as with the ``"array"`` type.

    For example:

    .. python::

        class CustomArrayAugmentation(A.BasicTransform):

            @property
            @override
            def targets(self):
                return {
                    "image": self.apply,
                    "array": self.apply_to_array,
                }

            @override
            def apply(self, image, **kwargs):
                ...

            def apply_to_array(
                self, array: np.ndarray, **kwargs
            ) -> np.ndarray:
                ...

    .. _Albumentations:
        https://albumentations.ai/explore/
    .. _A.DualTransform:
        https://github.com/albumentations-team/albumentations/blob/66212d77a44927a29d6a0e81621d3c27afbd929c/albumentations/core/transforms_interface.py#L545
    .. _A.ImageOnlyTransform:
        https://github.com/albumentations-team/albumentations/blob/66212d77a44927a29d6a0e81621d3c27afbd929c/albumentations/core/transforms_interface.py#L756
    .. _A.BasicTransform:
        https://github.com/albumentations-team/albumentations/blob/66212d77a44927a29d6a0e81621d3c27afbd929c/albumentations/core/transforms_interface.py#L49
    """

    def _check_augmentation_warnings(
        self,
        config_item: AlbumentationConfigItem,
        available_target_types: set,
    ) -> None:
        augmentation_name = config_item.name

        if "keypoints" in available_target_types and augmentation_name in [
            "HorizontalFlip",
            "VerticalFlip",
            "Transpose",
        ]:
            logger.warning(
                f"Using '{augmentation_name}' with keypoints."
                "If your dataset contains symmetric keypoints "
                "(e.g. left/right arms), you should use our custom "
                "`HorizontalSymmetricKeypointsFlip`, "
                "`VerticalSymmetricKeypointsFlip`, "
                "or `TransposeSymmetricKeypoints`"
                "to ensure keypoints are correctly reordered."
            )

    @deprecated(
        "is_validation_pipeline",
        suggest={"is_validation_pipeline": "pipeline_stage"},
        additional_message=(
            "Use `pipeline_stage='train'`, `'val'`, or `'test'` instead."
        ),
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
        is_validation_pipeline: bool | None = None,
        pipeline_stage: Literal["train", "val", "test"] | None = None,
        min_bbox_visibility: float = 0.0,
        seed: int | None = None,
        bbox_area_threshold: float = 0.0004,
    ):
        self._targets: dict[str, TargetType] = {}
        self._target_names_to_tasks = {}
        self._n_classes = n_classes
        self._image_size = (height, width)
        self._source_names = source_names
        self._bbox_area_threshold = bbox_area_threshold

        for task, task_type in targets.items():
            target_name = self._task_to_target_name(task)

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
                # Some Albumentations transforms read data["bboxes"] directly.
                if "bboxes" not in self._targets:
                    target_name = "bboxes"

            elif task_type == "keypoints":
                target_type = "keypoints"

            else:
                raise ValueError(
                    f"Unsupported task type: '{task_type}'. "
                    f"Only 'array', 'classification', 'segmentation', "
                    f"'instance_segmentation', 'boundingbox', "
                    f"'keypoints', and 'metadata' are supported."
                )

            self._targets[target_name] = target_type
            self._target_names_to_tasks[target_name] = task

        for source_name in source_names:
            self._targets[source_name] = "image"

        # Necessary for official Albumentations transforms.
        targets_without_instance_mask = {
            target_name: target_type
            if target_type != "instance_mask"
            else "mask"
            for target_name, target_type in self._targets.items()
        }

        pixel_transforms = []
        spatial_transforms = []
        batch_transforms = []
        custom_transforms = []
        resize_transform = None
        pipeline_stage = self._resolve_pipeline_stage(
            pipeline_stage=pipeline_stage,
            is_validation_pipeline=is_validation_pipeline,
        )
        validated_config = [
            cfg
            for cfg in (
                AlbumentationConfigItem.model_validate(config_item)
                for config_item in config
            )
            if cfg.name == "Normalize" or pipeline_stage in cfg.apply_on_stages
        ]

        available_target_types = set(self._targets.values())

        for cfg in validated_config:
            self._check_augmentation_warnings(cfg, available_target_types)

            if cfg.use_for_resizing:
                image_h, image_w = self._image_size
                cfg_h = cfg.params.get("height")
                cfg_w = cfg.params.get("width")
                cfg.params.setdefault("p", 1.0)
                if cfg_h != image_h or cfg_w != image_w:
                    logger.warning(
                        f"Resizing augmentation '{cfg.name}' has "
                        f"(height, width) that doesn't match "
                        f"image_size ({image_h}, {image_w}). Overriding."
                    )
                cfg.params["height"] = image_h
                cfg.params["width"] = image_w

            transform = self._create_transformation(cfg)

            if cfg.use_for_resizing:
                logger.info(f"Using '{cfg.name}' for resizing.")
                if resize_transform is not None:
                    raise ValueError(
                        "Only one resizing augmentation can be provided."
                    )
                resize_probability = cfg.params["p"]
                if isinstance(resize_probability, bool) or not isinstance(
                    resize_probability, (int, float)
                ):
                    raise TypeError(
                        f"Resizing augmentation '{cfg.name}' has invalid "
                        f"p={resize_probability!r}. Expected a float."
                    )
                resize_probability = float(resize_probability)
                if resize_probability < 1:
                    resize_transform = A.OneOf(
                        [
                            transform,
                            self._create_default_resize_transform(
                                keep_aspect_ratio=keep_aspect_ratio,
                                height=height,
                                width=width,
                                p=1 - resize_probability,
                            ),
                        ],
                        p=1.0,
                    )
                else:
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
                    f"Unsupported transformation type: "
                    f"'{type(transform).__name__}'. "
                    f"Only subclasses of `A.BasicTransform` "
                    f"or `A.BaseCompose` are allowed. "
                )

        wrapped_spatial_ops: TransformsSeqType = []
        if "keypoints" in targets.values():
            for op in spatial_transforms:
                wrapped_spatial_ops.append(op)
                wrapped_spatial_ops.append(
                    A.Lambda(
                        image=lambda img, **_: img,
                        keypoints=self._mark_invisible_keypoints,
                        p=1.0,
                    )
                )
        else:
            wrapped_spatial_ops = spatial_transforms

        if resize_transform is None:
            resize_transform = self._create_default_resize_transform(
                keep_aspect_ratio=keep_aspect_ratio,
                height=height,
                width=width,
            )

        def _get_params(is_custom: bool = False) -> dict[str, Any]:
            return {
                "bbox_params": A.BboxParams(
                    format="albumentations", min_visibility=min_bbox_visibility
                ),
                "keypoint_params": A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
                "additional_targets": (
                    self._targets
                    if is_custom
                    else targets_without_instance_mask
                ),
                "seed": seed,
            }

        # Warning issued when "bbox_params" or "keypoint_params"
        # are provided to a compose with transformations that
        # do not use them. We don't care about these warnings.
        with warnings.catch_warnings(record=True):
            self._batch_transform = BatchCompose(
                batch_transforms, **_get_params(is_custom=True)
            )
            self._spatial_transform = self._wrap_transform(
                A.Compose(wrapped_spatial_ops, **_get_params())
            )
            self._pixel_transform = self._wrap_transform(
                A.Compose(pixel_transforms),
                is_pixel=True,
                source_names=source_names,
            )
            self._resize_transform = self._wrap_transform(
                A.Compose([resize_transform], **_get_params())
            )
            self._custom_transform = self._wrap_transform(
                A.Compose(custom_transforms, **_get_params(is_custom=True))
            )

    @property
    @override
    def batch_size(self) -> int:
        return self._batch_transform.batch_size

    @override
    def apply(self, input_batch: list[LoaderMultiOutput]) -> LoaderMultiOutput:
        data_batch, n_keypoints = self._preprocess_batch(input_batch)

        data = self._batch_transform(data_batch)

        for target_name in list(data.keys()):
            value = data[target_name]
            if isinstance(value, np.ndarray) and value.size == 0:
                del data[target_name]

        data = self._spatial_transform(**data)
        data = self._custom_transform(**data)

        transformed_size = data["image"].shape[:2]

        if transformed_size != self._image_size:
            transformed_size = prod(transformed_size)
            target_size = prod(self._image_size)

            if transformed_size > target_size:
                data = self._resize_transform(**data)
                data = self._pixel_transform(**data)
            else:
                data = self._pixel_transform(**data)
                data = self._resize_transform(**data)
        else:
            data = self._pixel_transform(**data)

        return self._postprocess(data, n_keypoints)

    @staticmethod
    def _resolve_pipeline_stage(
        pipeline_stage: Literal["train", "val", "test"] | None,
        is_validation_pipeline: bool | None,
    ) -> Literal["train", "val", "test"]:
        if pipeline_stage is not None:
            return pipeline_stage
        return "val" if is_validation_pipeline else "train"

    def _preprocess_batch(
        self, labels_batch: list[LoaderMultiOutput]
    ) -> tuple[list[Data], dict[str, int]]:
        """Preprocess a batch of labels.

        Args:
            labels_batch: Loader outputs to preprocess.

        Returns:
            Preprocessed data and keypoint counts for each task.

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

            for target_name, target_type in self._targets.items():
                if target_name not in self._target_names_to_tasks:
                    continue

                task = self._target_names_to_tasks[target_name]

                if task not in labels:
                    if target_type == "mask":
                        data[target_name] = np.empty(
                            (
                                0,
                                0,
                                self._n_classes[
                                    self._target_names_to_tasks[target_name]
                                ],
                            )
                        )
                    elif target_type == "classification":
                        data[target_name] = np.zeros(
                            self._n_classes[
                                self._target_names_to_tasks[target_name]
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

    def _postprocess(
        self, data: Data, n_keypoints: dict[str, int]
    ) -> LoaderMultiOutput:
        """Postprocess the augmented data back to LDF format.

        Discards labels associated with bounding boxes that are outside the
        image.

        Args:
            data: Augmented data keyed by target name.
            n_keypoints: Mapping from task names to keypoint counts.

        Returns:
            Augmented images and labels.

        """
        out_labels = {}
        out_image_dict = {}

        image_keys = [k for k in data if k in ["image", *self._source_names]]

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

        for target_name, target_type in self._targets.items():
            if target_name not in data:
                continue

            array = data[target_name]
            if array.size == 0:
                continue

            task = self._target_names_to_tasks[target_name]
            task_name = get_task_name(task)

            if target_type == "bboxes":
                out_labels[task], index = postprocess_bboxes(
                    array, self._bbox_area_threshold
                )
                bboxes_indices[task_name] = index

        for target_name, target_type in self._targets.items():
            if target_name not in data:
                continue

            array = data[target_name]
            if array.size == 0:
                continue

            task = self._target_names_to_tasks[target_name]
            task_name = get_task_name(task)

            if task_name not in bboxes_indices:
                if "bboxes" in self._targets.values():
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
        keypoints: np.ndarray, shape: tuple[int, int], **_
    ) -> np.ndarray:
        h, w = shape[:2]
        kps = keypoints.copy()
        xs, ys = kps[:, 0], kps[:, 1]
        oob = (xs < 0) | (ys < 0) | (xs >= w) | (ys >= h)
        kps[oob, -1] = 0.0
        return kps

    def _create_default_resize_transform(
        self,
        keep_aspect_ratio: bool,
        height: int,
        width: int,
        p: float = 1.0,
    ) -> A.DualTransform:
        if keep_aspect_ratio:
            return LetterboxResize(
                height=height,
                width=width,
                p=p,
            )
        return A.Resize(height=height, width=width, p=p)

    @staticmethod
    def _create_transformation(
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
                    transform = AlbumentationsEngine._create_transformation(
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
    def _task_to_target_name(task: str) -> str:
        target = task.replace("/", "_").replace("-", "_")
        assert target.isidentifier()
        return target

    @staticmethod
    def _wrap_transform(
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
                        "`source_names` must be provided "
                        "for pixel transformations."
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
