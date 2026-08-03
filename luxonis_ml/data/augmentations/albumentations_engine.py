import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from math import prod
from typing import Any, Final, Literal, TypeAlias, cast

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from loguru import logger
from pydantic import Field
from typing_extensions import override

from luxonis_ml.data.utils.task_utils import get_task_type, task_is_metadata
from luxonis_ml.typing import ConfigItem, LoaderMultiOutput, Params
from luxonis_ml.utils import deprecated

from .base_engine import AugmentationEngine
from .batch_compose import BatchCompose
from .batch_transform import BatchTransform
from .custom import TRANSFORMATIONS
from .custom.letterbox_resize import create_letterbox_or_resize
from .utils import (
    instance_count,
    postprocess_bboxes,
    postprocess_keypoints,
    postprocess_mask,
    preprocess_bboxes,
    preprocess_keypoints,
    preprocess_mask,
)

Data: TypeAlias = dict[str, np.ndarray]
# LDF layout of a single annotation: its trailing shape and its dtype.
LabelSpec: TypeAlias = tuple[tuple[int, ...], np.dtype]

# Marks a parameter that cannot be reported as provenance.
_OMIT: Final = object()

# Parameters `Albumentations` can report back unchanged from how the
# transformation was configured. Such a value says nothing about what
# happened to a particular sample, but the same key carries a sampled value
# for other transformations, so the two are told apart per transformation.
_STATIC_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "fill",
        "fill_mask",
        "image_shapes",
        "interpolation",
        "mask_interpolation",
        "out_height",
        "out_width",
        "shape",
    }
)
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

# Target types describing individual instances, and so kept in step with the
# bounding boxes of their task group as those are filtered.
ASSOCIATED_TARGET_TYPES = frozenset(
    {"instance_mask", "keypoints", "array", "metadata"}
)


def _normalize_params(
    params: Mapping[Any, Any], transform: Any = None
) -> Params:
    """Keep the sampled parameters that can be reported as provenance.

    Args:
        params: Parameters `Albumentations` reported back.
        transform: Transformation the parameters belong to, needed to tell
            a configured value from a sampled one. Parameters nested inside
            another parameter have no transformation of their own.

    """
    normalized: Params = {}
    for key, value in params.items():
        key = str(key)
        value = _normalize_value(value)
        if value is _OMIT or _is_configured(transform, key, value):
            continue
        normalized[key] = value
    return normalized


def _is_configured(transform: Any, key: str, value: Any) -> bool:
    """Whether a reported parameter only echoes back the configuration.

    A transformation that samples one of these parameters reports a value
    that differs from the one it was configured with - `A.CropAndPad` draws
    its ``fill`` from the range it was given - and that value is part of
    what happened to the sample. Keys the transformation knows nothing
    about, such as the input ``shape``, are never sampled.
    """
    if key not in _STATIC_PARAMS:
        return False
    configured = getattr(transform, key, _OMIT)
    return configured is _OMIT or _normalize_value(configured) == value


def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _OMIT
    if isinstance(value, np.generic):
        return _normalize_value(value.item())
    if isinstance(value, Mapping):
        return _normalize_params(value)
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            item = _normalize_value(item)
            # Reporting the remaining items would silently reindex the
            # sequence and misrepresent its length.
            if item is _OMIT:
                return _OMIT
            normalized.append(item)
        return normalized
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _reset_params(transform: Any) -> None:
    if hasattr(transform, "params"):
        transform.params = {}
    if isinstance(transform, A.BaseCompose):
        for child in transform.transforms:
            _reset_params(child)


def _disambiguate_paths(paths: dict[int, str]) -> dict[int, str]:
    """Suffix repeated configured paths with ``"#<n>"``.

    The same transformation can be configured more than once, possibly with
    different parameters. Without the suffix all occurrences would collapse
    into a single entry reporting only the last one's runtime parameters.
    """
    duplicated = {
        path for path, count in Counter(paths.values()).items() if count > 1
    }
    seen: Counter[str] = Counter()
    disambiguated = {}
    for key, path in paths.items():
        if path in duplicated:
            seen[path] += 1
            path = f"{path}#{seen[path]}"
        disambiguated[key] = path
    return disambiguated


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
    -------------------

    These require several images at once and must be configured as
    top-level augmentations. Nesting one inside a composition such as
    ``OneOf`` or ``Sequential`` raises a ``ValueError``.

    `CutMix`
    ~~~~~~~~

    CutMix is a data augmentation technique that patches one source image
    into another using a rectangle sampled from a beta-distributed mixing
    coefficient.

    Bounding boxes of the first image survive as long as the part of them
    left outside the patch is at least ``bbox_min_visibility`` of their
    original area; the ones the patch covers past that are dropped
    together with their instance masks, keypoints and metadata. Bounding
    boxes of the second image are clipped to the patch.

    ``occluded_bbox_strategy`` picks what happens to a survivor the patch
    partly covers. The default ``"keep"`` leaves it at its original
    extent, since the object still fills the box and is only covered up;
    ``"clip"`` shrinks it to the largest part of itself the patch does not
    reach, so no box edge is left sitting on top of the patch.

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
                f"Using '{augmentation_name}' with keypoints. "
                "If your dataset contains symmetric keypoints "
                "(e.g. left/right arms), you should use our custom "
                "`HorizontalSymmetricKeypointsFlip`, "
                "`VerticalSymmetricKeypointsFlip`, "
                "or `TransposeSymmetricKeypoints`"
                " to ensure keypoints are correctly reordered."
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
        """Create an Albumentations-backed augmentation pipeline.

        Args:
            height: Target output image height.
            width: Target output image width.
            targets: Task names mapped to task types. Supported task
                types are ``"array"``, ``"classification"``,
                ``"segmentation"``, ``"instance_segmentation"``,
                ``"boundingbox"``, ``"keypoints"``, and metadata tasks.
            n_classes: Number of classes per task.
            source_names: Source names that should be treated as image
                targets.
            config: Iterable of augmentation configuration dictionaries.
            keep_aspect_ratio: Whether the default resize transform should
                preserve image aspect ratio.
            is_validation_pipeline: Optional legacy flag selecting
                evaluation behavior.

                .. deprecated:: 0.5.0
                    use ``pipeline_stage`` instead.

            pipeline_stage: Optional explicit pipeline stage. Valid values
                are ``"train"``, ``"val"``, and ``"test"``.
            min_bbox_visibility: Minimum visible fraction of a bounding
                box after augmentation.
            seed: Optional random seed.
            bbox_area_threshold: Minimum normalized bounding-box area kept
                after augmentation.

        Raises:
            ValueError: If a target task type is unsupported, more than
                one bounding-box target belongs to the same task group,
                more than one transform is marked for resizing, or a
                configured transform is not an Albumentations transform.
            TypeError: If a resizing transform has a non-numeric
                probability ``p``.

        """
        self._targets: dict[str, TargetType] = {}
        self._target_names_to_tasks = {}
        self._target_names_to_task_groups = {}
        self._n_classes = n_classes
        self._image_size = (height, width)
        self._source_names = source_names
        self._bbox_area_threshold = bbox_area_threshold
        self._applied_augmentations: dict[str, Params] = {}
        self._tracked_augmentation_paths: dict[int, str] = {}

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
            self._target_names_to_task_groups[target_name] = (
                self._get_task_group(task)
            )

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

            transform = self._create_transformation(
                cfg, self._tracked_augmentation_paths
            )

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
                    fallback_resize = create_letterbox_or_resize(
                        keep_aspect_ratio=keep_aspect_ratio,
                        height=height,
                        width=width,
                        p=1 - resize_probability,
                    )
                    # Either branch of the `OneOf` can be picked, so both
                    # are tracked. Reporting only the configured one would
                    # leave a resized sample looking un-augmented.
                    self._tracked_augmentation_paths[id(transform)] = (
                        f"OneOf/{cfg.name}"
                    )
                    self._tracked_augmentation_paths[id(fallback_resize)] = (
                        f"OneOf/{type(fallback_resize).__name__} (fallback)"
                    )
                    resize_transform = A.OneOf(
                        [transform, fallback_resize], p=1.0
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
            resize_transform = create_letterbox_or_resize(
                keep_aspect_ratio=keep_aspect_ratio,
                height=height,
                width=width,
            )

        self._tracked_augmentation_paths = _disambiguate_paths(
            self._tracked_augmentation_paths
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

        bbox_targets_by_group = {}
        for target_name, target_type in self._targets.items():
            if target_type != "bboxes":
                continue
            task_group = self._target_names_to_task_groups[target_name]
            if task_group in bbox_targets_by_group:
                raise ValueError(
                    "Multiple bounding-box targets belong to task group "
                    f"'{task_group}'."
                )
            bbox_targets_by_group[task_group] = target_name

        self._bbox_targets_by_group = bbox_targets_by_group
        bbox_associations = {
            bbox_target: {
                target_name: target_type
                for target_name, target_type in self._targets.items()
                if target_type in ASSOCIATED_TARGET_TYPES
                and self._target_names_to_task_groups[target_name]
                == task_group
            }
            for task_group, bbox_target in bbox_targets_by_group.items()
        }
        # Targets tied to a bbox target, the boxes themselves included. Only
        # these are reported as empty when a transform clears them out.
        self._grouped_targets = set(bbox_targets_by_group.values()) | {
            target_name
            for associations in bbox_associations.values()
            for target_name in associations
        }

        # Warning issued when "bbox_params" or "keypoint_params"
        # are provided to a compose with transformations that
        # do not use them. We don't care about these warnings.
        with warnings.catch_warnings(record=True):
            self._batch_transform = BatchCompose(
                batch_transforms,
                bbox_associations=bbox_associations,
                **_get_params(is_custom=True),
            )
            self._spatial_compose = A.Compose(
                wrapped_spatial_ops, **_get_params()
            )
            self._spatial_transform = self._wrap_transform(
                self._spatial_compose
            )
            # Composed once, and seeded like the rest: `A.ReplayCompose`
            # reseeds the transformations it is given, so building one per
            # call would throw the seeded random state away every time.
            self._pixel_compose = A.ReplayCompose(pixel_transforms)
            self._pixel_compose.set_random_seed(seed)
            self._pixel_transform = self._wrap_transform(
                self._pixel_compose,
                is_pixel=True,
                source_names=source_names,
            )
            self._resize_compose = A.Compose(
                [resize_transform], **_get_params()
            )
            self._resize_transform = self._wrap_transform(self._resize_compose)
            self._custom_compose = A.Compose(
                custom_transforms, **_get_params(is_custom=True)
            )
            self._custom_transform = self._wrap_transform(self._custom_compose)

    @property
    @override
    def batch_size(self) -> int:
        return self._batch_transform.batch_size

    @property
    @override
    def applied_augmentations(self) -> dict[str, Params]:
        """Configured paths and runtime parameters from the latest call."""
        return self._applied_augmentations

    @property
    @override
    def batch_augmentation_indices(self) -> list[int]:
        """Input positions contributing to the last augmented output."""
        return self._batch_transform.batch_augmentation_indices

    @override
    def apply(self, input_batch: list[LoaderMultiOutput]) -> LoaderMultiOutput:
        # Rebound, not cleared: the previous mapping is kept by its caller
        # alongside the sample it describes.
        self._applied_augmentations = {}
        for compose in self._compositions:
            _reset_params(compose)

        data_batch, n_keypoints, label_specs = self._preprocess_batch(
            input_batch
        )

        data = self._batch_transform(
            data_batch, keypoints_per_instance=n_keypoints
        )

        # A batch transform discards the members it did not merge, and a task
        # only those members carried is not part of this sample at all.
        contributed = {}
        for index in self.batch_augmentation_indices:
            for target_name, spec in label_specs[index].items():
                contributed.setdefault(target_name, spec)

        # Albumentations chokes on zero-size targets, so they are dropped
        # here. Ones a batch transform emptied are remembered so that
        # postprocessing can still report them as empty rather than omit
        # the task entirely.
        emptied_targets = {}
        for target_name in list(data.keys()):
            value = data[target_name]
            if isinstance(value, np.ndarray) and value.size == 0:
                del data[target_name]
                if (
                    target_name in contributed
                    and target_name in self._grouped_targets
                ):
                    emptied_targets[target_name] = contributed[target_name]

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

        self._record_applied_augmentations()

        return self._postprocess(data, n_keypoints, emptied_targets)

    @property
    def _compositions(self) -> tuple[A.BaseCompose, ...]:
        return (
            self._batch_transform,
            self._spatial_compose,
            self._custom_compose,
            self._pixel_compose,
            self._resize_compose,
        )

    def _record_applied_augmentations(self) -> None:
        """Record what applied, once every composition has run.

        Albumentations leaves the sampled ``params`` on each transformation
        after the call, so the whole pipeline can be read at the end rather
        than after every stage. The report is then keyed in pipeline-group
        order, which does not shift with the resize and pixel stages
        swapping places for an oversized sample.
        """
        self._record_applied_batch_augmentations()
        for compose in self._compositions:
            self._record_composed_augmentations(compose)

    def _record_composed_augmentations(self, transform: Any) -> None:
        """Record the tracked transformations of a composition that applied.

        Transformations are matched by identity rather than by class name,
        so registry aliases, repeated entries, and the `A.Lambda` instances
        the engine injects itself are all handled correctly.

        Only `A.BaseCompose` instances are descended into. A ``transforms``
        attribute is not enough: leaf transformations such as
        `A.ColorJitter` expose one for their internal adjustment functions,
        and treating those as children would drop the transformation.
        """
        if isinstance(transform, A.BaseCompose):
            for child in transform.transforms:
                self._record_composed_augmentations(child)
            return

        # Batch transforms are read from `BatchCompose.applied_params`
        # instead; their own `params` only describe the last sub-batch.
        if isinstance(transform, BatchTransform):
            return

        # Unlike a batch transform, whose `update_transform_params` may
        # sample nothing at all, a transformation applied here always
        # reports at least the input ``shape`` that
        # `A.BasicTransform.update_transform_params` adds, so empty
        # parameters mean the transformation did not apply.
        path = self._tracked_augmentation_paths.get(id(transform))
        if path is not None and getattr(transform, "params", None):
            self._applied_augmentations[path] = _normalize_params(
                transform.params, transform
            )

    def _record_applied_batch_augmentations(self) -> None:
        """Record batch transforms from the parameters `BatchCompose` kept.

        A batch transform is invoked once per sub-batch, so its ``params``
        only describe the last invocation, which may well be one whose
        output was discarded. `BatchCompose` keeps the parameters of the
        invocations that shaped the returned sample, which is what
        provenance has to report.
        """
        batch_transforms = {
            id(transform): transform
            for transform in self._batch_transform.transforms
        }
        for key, params in self._batch_transform.applied_params.items():
            path = self._tracked_augmentation_paths.get(key)
            if path is not None:
                self._applied_augmentations[path] = _normalize_params(
                    params, batch_transforms.get(key)
                )

    def _preprocess_batch(
        self, labels_batch: list[LoaderMultiOutput]
    ) -> tuple[list[Data], dict[str, int], list[dict[str, LabelSpec]]]:
        """Preprocess a batch of labels.

        Args:
            labels_batch: Loader outputs to preprocess.

        Returns:
            Preprocessed data, keypoint counts for each task, and, for every
            member of the batch, the LDF layout of each target it provided
            labels for.

        """
        data_batch = []
        n_keypoints = {}
        label_specs: list[dict[str, LabelSpec]] = []

        for image_dict, labels in labels_batch:
            data = {}
            specs: dict[str, LabelSpec] = {}

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
                specs[target_name] = (array.shape[1:], array.dtype)

                if target_type in {"mask", "instance_mask"}:
                    data[target_name] = preprocess_mask(array)

                elif target_type == "bboxes":
                    data[target_name] = preprocess_bboxes(array)

                elif target_type == "keypoints":
                    n_keypoints[target_name] = array.shape[1] // 3
                    data[target_name] = preprocess_keypoints(
                        array, height, width
                    )
                else:
                    data[target_name] = array

            data_batch.append(data)
            label_specs.append(specs)

        return data_batch, n_keypoints, label_specs

    def _postprocess(
        self,
        data: Data,
        n_keypoints: dict[str, int],
        emptied_targets: dict[str, LabelSpec],
    ) -> LoaderMultiOutput:
        """Postprocess the augmented data back to LDF format.

        Discards labels associated with bounding boxes that are outside the
        image.

        Args:
            data: Augmented data keyed by target name.
            n_keypoints: Mapping from task names to keypoint counts.
            emptied_targets: LDF layout of the targets a batch transform
                emptied because all the bounding boxes they belong to were
                filtered out. They are reported as empty labels rather than
                omitted.

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
            if target_type != "bboxes" or target_name not in data:
                continue

            # An emptied bbox target is kept rather than skipped: its
            # associated labels are still reported as empty, so dropping the
            # task would leave them without the boxes they belong to.
            task = self._target_names_to_tasks[target_name]
            out_labels[task], index = postprocess_bboxes(
                data[target_name], self._bbox_area_threshold
            )
            bboxes_indices[self._target_names_to_task_groups[target_name]] = (
                index
            )

        for target_name, target_type in self._targets.items():
            if target_type == "bboxes" or target_name not in data:
                continue

            array = data[target_name]
            if array.size == 0:
                continue

            task = self._target_names_to_tasks[target_name]

            if target_type == "mask":
                out_labels[task] = postprocess_mask(array)
                continue

            if target_type == "classification":
                out_labels[task] = array
                continue

            available = instance_count(
                array, target_type, n_keypoints.get(target_name, 0)
            )
            bbox_ordering = self._resolve_ordering(
                self._target_names_to_task_groups[target_name],
                bboxes_indices,
                available,
                task,
            )

            if target_type == "instance_mask":
                out_labels[task] = postprocess_mask(array)[bbox_ordering]

            elif target_type == "keypoints":
                out_labels[task] = postprocess_keypoints(
                    array,
                    bbox_ordering,
                    image_height,
                    image_width,
                    n_keypoints[target_name],
                )
            else:
                out_labels[task] = array[bbox_ordering]

        for target_name, (trailing, dtype) in emptied_targets.items():
            task = self._target_names_to_tasks[target_name]
            if self._targets[target_name] == "instance_mask":
                # The recorded shape is the pre-augmentation one.
                trailing = (image_height, image_width)
            out_labels[task] = np.zeros((0, *trailing), dtype=dtype)

        return out_image_dict, out_labels

    def _resolve_ordering(
        self,
        task_group: str,
        bboxes_indices: dict[str, np.ndarray],
        available: int,
        task: str,
    ) -> np.ndarray:
        """Instances of a task to keep, in the order its boxes survived.

        Instances a bounding box no longer accounts for are dropped. An
        ordering reaching past the labels that are actually there means the
        annotation carried a different number of instances than boxes, which
        `BatchCompose` has already refused to guess at; the excess is dropped
        so that the sample still loads.
        """
        if task_group not in bboxes_indices:
            if task_group in self._bbox_targets_by_group:
                return np.array([], dtype=int)
            return np.arange(available)

        ordering = bboxes_indices[task_group]
        if ordering.size and ordering.max() >= available:
            logger.warning(
                f"Task '{task}' has {available} instances for "
                f"{ordering.max() + 1} bounding boxes; the instances without "
                f"a box are dropped and the rest may be misaligned."
            )
            ordering = ordering[ordering < available]
        return ordering

    @staticmethod
    def _resolve_pipeline_stage(
        pipeline_stage: Literal["train", "val", "test"] | None,
        is_validation_pipeline: bool | None,
    ) -> Literal["train", "val", "test"]:
        if pipeline_stage is not None:
            return pipeline_stage
        return "val" if is_validation_pipeline else "train"

    @staticmethod
    def _mark_invisible_keypoints(
        keypoints: np.ndarray, shape: tuple[int, int], **_
    ) -> np.ndarray:
        """Mark keypoints outside the image bounds as invisible.

        Args:
            keypoints: Keypoints with visibility stored in the last column.
            shape: Image shape used for bounds checking.

        Returns:
            Copy of ``keypoints`` with out-of-bounds visibility set to zero.

        """
        h, w = shape[:2]
        kps = keypoints.copy()
        xs, ys = kps[:, 0], kps[:, 1]
        oob = (xs < 0) | (ys < 0) | (xs >= w) | (ys >= h)
        kps[oob, -1] = 0.0
        return kps

    @staticmethod
    def _create_transformation(
        config: AlbumentationConfigItem,
        tracked_paths: dict[int, str],
        parent_path: tuple[str, ...] = (),
    ) -> A.BasicTransform | A.BaseCompose:
        """Instantiate a configured transformation.

        Args:
            config: Validated configuration item to instantiate.
            tracked_paths: Collects the configured path of every created
                leaf transformation, keyed by its identity. Provenance is
                matched by identity so that registry aliases and repeated
                entries are handled correctly.
            parent_path: Configured names of the enclosing compositions.

        Returns:
            The instantiated transformation. Configurations naming a
            composition (``OneOf``, ``SomeOf``, ``Sequential``) yield an
            `A.BaseCompose`.

        Raises:
            ValueError: If a batch transform is nested inside another
                transformation, or if a nested item is not a valid
                transformation configuration.

        """
        params = config.params.copy()
        current_path = (*parent_path, config.name)

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
                        nested_cfg, tracked_paths, current_path
                    )
                    if isinstance(transform, BatchTransform):
                        raise ValueError(
                            f"Batch transform '{item['name']}' cannot be "
                            f"nested inside '{config.name}'. "
                            "Batch transforms (e.g. Mosaic4, MixUp, "
                            "and CutMix) require multiple images and "
                            "must be used as top-level augmentations."
                        )
                    nested_transforms.append(transform)
                else:
                    raise ValueError(
                        f"Invalid nested transform configuration: {item}"
                    )
            params["transforms"] = nested_transforms

        if hasattr(A, config.name):
            transform = getattr(A, config.name)(**params)
        else:
            constructor = TRANSFORMATIONS.get(config.name)
            transform = constructor(**params)  # type: ignore

        if not isinstance(transform, A.BaseCompose):
            tracked_paths[id(transform)] = "/".join(current_path)
        return transform

    @staticmethod
    def _get_task_group(task: str) -> str:
        """Return the complete task path without its task-type suffix.

        Unlike `get_task_name`, this keeps every level of a nested task
        name, so ``"a/b/keypoints"`` groups under ``"a/b"``. The leading
        ``"/"`` of LDF's default task marks a name that is empty on purpose,
        while a task with no separator at all has no name to group by and is
        only ever grouped with itself.
        """
        group = task.removesuffix(get_task_type(task)).removesuffix("/")
        if group:
            return group
        return "" if task.startswith("/") else task

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
        """Wrap an Albumentations composition for loader data dictionaries.

        Args:
            transform: Albumentations composition to wrap. Must be an
                `A.ReplayCompose` when ``is_pixel`` is set.
            is_pixel: Whether the composition contains pixel-only
                transforms that should be replayed across image sources.
            source_names: Image source names replayed for pixel-only
                transforms.

        Returns:
            Callable that applies the composition to a data dictionary.

        Raises:
            ValueError: If ``is_pixel`` is true and ``source_names`` is not
                provided.

        """

        def apply_transform(**data: np.ndarray) -> Data:
            if not transform.transforms:
                return data

            if is_pixel:
                if source_names is None:
                    raise ValueError(
                        "`source_names` must be provided "
                        "for pixel transformations."
                    )
                result = transform(image=data["image"])
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
            else:
                original_key = data.pop("_original_image_key", None)
                data = transform(**data)
                if original_key is not None:
                    data["_original_image_key"] = original_key

            return data

        return apply_transform
