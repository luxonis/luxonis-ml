from abc import ABC, abstractmethod
from typing import Dict, List, Type

from luxonis_ml.typing import ConfigItem, LoaderOutput
from luxonis_ml.utils import AutoRegisterMeta, Registry

AUGMENTATION_ENGINES: Registry[Type["AugmentationEngine"]] = Registry(
    name="augmentation_engines"
)


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
        targets: Dict[str, str],
        config: List[ConfigItem],
        keep_aspect_ratio: bool,
        is_validation_pipeline: bool,
        min_bbox_visibility: float = 0,
    ):
        """Initialize augmentation pipeline from configuration.

        @type height: int
        @param height: Target image height
        @type width: int
        @param width: Target image width
        @type targets: Dict[str, str]
        @param targets: Dictionary mapping task names to task types.
            Example::
                {
                    "detection/boundingbox": "bbox",
                    "detection/segmentation": "mask",
                }
        @type config: List[Dict[str, Any]]
        @param config: List of dictionaries with augmentation
            configurations. The configuration is a list of dictionaries
            where each dictionary contains the name of the augmentation
            and optionally its parameters. The name must be a valid
            augmentation class name. The parameters are passed to the
            constructor of the augmentation class.
            The order of the augmentations is not guaranteed to be
            preserved.

            Example::

                [
                    {"name": "HorizontalFlip", "params": {"p": 0.5}},
                    {"name": "RandomBrightnessContrast", "params": {"p": 0.1}},
                    {"name": "Defocus"}
                ]

        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep aspect ratio
        @type is_validation_pipeline: bool
        @param is_validation_pipeline: Whether this is a validation
            pipeline (in which case some augmentations are skipped)
        @type min_bbox_visibility: float
        @param min_bbox_visibility: Minimum area of a bounding box to be
            considered visible.
        """
        ...

    @abstractmethod
    def apply(self, data: List[LoaderOutput]) -> LoaderOutput:
        """Apply the augmentation pipeline to the data.

        @type data: List[LuxonisLoaderOutput]
        @param data: List of data to augment. The length of the list
            must be equal to the batch size.
        @rtype: LuxonisLoaderOutput
        @return: Augmented data
        """
        ...

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
        the batch size should be 8 (2 * 4).
        """
        ...

    @property
    def is_batched(self) -> bool:
        """Whether the augmentation pipeline is batched."""
        return self.batch_size > 1
