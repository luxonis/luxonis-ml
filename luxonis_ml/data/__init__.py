import pkg_resources

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("data"):
    from .augmentations import AlbumentationsEngine
    from .datasets import (
        DATASETS_REGISTRY,
        BaseDataset,
        DatasetIterator,
        LuxonisComponent,
        LuxonisDataset,
        LuxonisSource,
        UpdateMode,
    )
    from .loaders import LOADERS_REGISTRY, BaseLoader, LuxonisLoader
    from .parsers import LuxonisParser
    from .utils.enums import BucketStorage, BucketType, ImageType, MediaType


def load_dataset_plugins() -> None:  # pragma: no cover
    """Registers any external dataset BaseDataset class plugins."""
    for entry_point in pkg_resources.iter_entry_points("dataset_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register_module(module=plugin_class)


def load_loader_plugins() -> None:  # pragma: no cover
    """Registers any external dataset BaseLoader class plugins."""
    for entry_point in pkg_resources.iter_entry_points("loader_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register_module(module=plugin_class)


load_dataset_plugins()
load_loader_plugins()

__all__ = [
    "AlbumentationsEngine",
    "BaseDataset",
    "BaseLoader",
    "BucketStorage",
    "BucketType",
    "DatasetIterator",
    "DATASETS_REGISTRY",
    "LOADERS_REGISTRY",
    "ImageType",
    "LuxonisComponent",
    "LuxonisDataset",
    "UpdateMode",
    "LuxonisLoader",
    "LuxonisParser",
    "LuxonisSource",
    "MediaType",
]
