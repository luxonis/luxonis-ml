import pkg_resources

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("data"):
    from .augmentations import Augmentations
    from .datasets import (
        DATASETS_REGISTRY,
        BaseDataset,
        DatasetIterator,
        LuxonisComponent,
        LuxonisDataset,
        LuxonisSource,
    )
    from .loaders import (
        LOADERS_REGISTRY,
        BaseLoader,
        Labels,
        LuxonisLoader,
        LuxonisLoaderOutput,
    )
    from .parsers import LuxonisParser
    from .utils.enums import (
        BucketStorage,
        BucketType,
        ImageType,
        LabelType,
        MediaType,
    )


def load_dataset_plugins() -> None:
    """Registers any external dataset BaseDataset class plugins."""
    for entry_point in pkg_resources.iter_entry_points("dataset_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register_module(module=plugin_class)


def load_loader_plugins() -> None:
    """Registers any external dataset BaseLoader class plugins."""
    for entry_point in pkg_resources.iter_entry_points("loader_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register_module(module=plugin_class)


load_dataset_plugins()
load_loader_plugins()

__all__ = [
    "Augmentations",
    "BaseDataset",
    "BaseLoader",
    "BucketStorage",
    "BucketType",
    "DatasetIterator",
    "DATASETS_REGISTRY",
    "LOADERS_REGISTRY",
    "ImageType",
    "LabelType",
    "Labels",
    "LuxonisComponent",
    "LuxonisDataset",
    "LuxonisLoader",
    "LuxonisLoaderOutput",
    "LuxonisParser",
    "LuxonisSource",
    "MediaType",
]
