from abc import ABC, abstractmethod
from typing import Any, Dict, List

from luxonis_ml.data.utils import LuxonisLoaderOutput


class BaseAugmentationPipeline(ABC):
    @classmethod
    @abstractmethod
    def from_config(
        cls,
        height: int,
        width: int,
        config: List[Dict[str, Any]],
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
