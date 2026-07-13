from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Final, Literal

from luxonis_ml.typing import LoaderMultiOutput, Params
from luxonis_ml.utils import AutoRegisterMeta, Registry

AUGMENTATION_ENGINES: Final[Registry[type["AugmentationEngine"]]] = Registry(
    name="augmentation_engines"
)
"""Registry for augmentation engines."""


# TODO: The engine should probably also handle normalization
# so it doesn't have to be done by injecting a normalization
# transformation to the config in LuxonisTrain.
class AugmentationEngine(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=AUGMENTATION_ENGINES,
    register=False,
):
    @abstractmethod
    def __init__(
        self,
        height: int,
        width: int,
        targets: Mapping[str, str],
        n_classes: Mapping[str, int],
        source_names: list[str],
        config: Iterable[Params],
        keep_aspect_ratio: bool,
        is_validation_pipeline: bool | None = None,
        pipeline_stage: Literal["train", "val", "test"] | None = None,
        min_bbox_visibility: float = 0.0,
        seed: int | None = None,
        bbox_area_threshold: float = 0.0004,
    ):
        """Initialize augmentation pipeline from configuration.

        Args:
            height: Target image height.

            width: Target image width.

            targets: Task names mapped to task types, such as
                ``{"detection/boundingbox": "bbox"}``.

            n_classes: Number of associated classes for each task, such as
                ``{"cars/boundingbox": 2}``.

            source_names: Source names expected in loader images.

            config: Augmentation configuration. Each item describes one
                augmentation; interpretation is engine-specific.

            keep_aspect_ratio: Whether to preserve image aspect ratio while
                resizing.

            is_validation_pipeline: Optional backward-compatible
                train-versus-eval hint.

                .. deprecated:: 0.5.0
                    use ``pipeline_stage`` instead.

            pipeline_stage: Optional explicit pipeline stage. When provided,
                it takes precedence over ``is_validation_pipeline``.

            min_bbox_visibility: Minimum fraction of the original bounding
                box that must remain visible after augmentation.

            seed: Optional random seed for reproducible augmentation.

            bbox_area_threshold: Minimum normalized area for bounding boxes
                to remain valid. The default removes very small boxes and
                their associated keypoints.

        """
        ...

    @abstractmethod
    def apply(self, input_batch: list[LoaderMultiOutput]) -> LoaderMultiOutput:
        """Apply the augmentation pipeline to the data.

        Args:
            input_batch: Loader outputs to augment.
                The number of items must match the engine's batch size.

        Returns:
            Augmented loader output.

        """
        ...

    @property
    @abstractmethod
    def batch_size(self) -> int:
        r"""The batch size required by the augmentation pipeline.

        The batch size is the number of images requested by the
        augmentation pipeline in case of batch-based augmentations.

        For example, if the augmentation pipeline contains the MixUp
        augmentation, the batch size should be :math:`2`.

        If the pipeline requires MixUp and also Mosaic4 augmentations,
        the batch size should be :math:`8 = \left(2 \cdot 4\right)`.
        """
        ...
