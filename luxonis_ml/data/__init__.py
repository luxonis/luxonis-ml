import pkg_resources

from ..guard_extras import guard_missing_extra

with guard_missing_extra("data"):
    from .augmentations import Augmentations, TrainAugmentations, ValAugmentations
    from .datasets import (
        DATASETS_REGISTRY,
        BaseDataset,
        DatasetGenerator,
        DatasetGeneratorFunction,
        LuxonisComponent,
        LuxonisDataset,
        LuxonisSource,
    )
    from .loader import BaseLoader, LabelType, LuxonisLoader, LuxonisLoaderOutput
    from .parsers import LuxonisParser
    from .utils.enums import (
        BucketStorage,
        BucketType,
        ImageType,
        MediaType,
    )


def load_dataset_plugins() -> None:
    """Registers any external dataset BaseDataset class plugins."""
    for entry_point in pkg_resources.iter_entry_points("dataset_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register_module(module=plugin_class)


load_dataset_plugins()

__all__ = [
    "Augmentations",
    "BaseDataset",
    "BaseLoader",
    "BucketStorage",
    "BucketType",
    "DatasetGenerator",
    "DatasetGeneratorFunction",
    "DATASETS_REGISTRY",
    "ImageType",
    "LabelType",
    "LuxonisComponent",
    "LuxonisDataset",
    "LuxonisLoader",
    "LuxonisLoaderOutput",
    "LuxonisParser",
    "LuxonisSource",
    "MediaType",
    "TrainAugmentations",
    "ValAugmentations",
]
