from importlib.metadata import entry_points

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("data"):
    from .augmentations import AlbumentationsEngine
    from .datasets import (
        DATASETS_REGISTRY,
        BaseDataset,
        Category,
        DatasetIterator,
        LuxonisComponent,
        LuxonisDataset,
        LuxonisSource,
        Metadata,
        UpdateMode,
    )
    from .loaders import LOADERS_REGISTRY, BaseLoader, LuxonisLoader
    from .parsers import LuxonisParser
    from .utils.enums import BucketStorage, BucketType, ImageType, MediaType


def load_dataset_plugins() -> None:  # pragma: no cover
    """Registers any external dataset BaseDataset class plugins."""
    for entry_point in _get_entry_points_subset("dataset_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register(module=plugin_class)


def load_loader_plugins() -> None:  # pragma: no cover
    """Registers any external dataset BaseLoader class plugins."""
    for entry_point in _get_entry_points_subset("loader_plugins"):
        plugin_class = entry_point.load()
        DATASETS_REGISTRY.register(module=plugin_class)


def _get_entry_points_subset(key: str) -> list:
    """Returns subset of selected entry points.

    Safe function for older (py3.8) and newer py versions
    """
    entry_points_obj = entry_points()
    if isinstance(entry_points_obj, dict):
        # py3.8 specific
        selected_entry_points = entry_points_obj.get(key, [])
    else:
        selected_entry_points = entry_points_obj.select(group=key)
    return selected_entry_points


load_dataset_plugins()
load_loader_plugins()

__all__ = [
    "DATASETS_REGISTRY",
    "LOADERS_REGISTRY",
    "AlbumentationsEngine",
    "BaseDataset",
    "BaseLoader",
    "BucketStorage",
    "BucketType",
    "Category",
    "DatasetIterator",
    "ImageType",
    "LuxonisComponent",
    "LuxonisDataset",
    "LuxonisLoader",
    "LuxonisParser",
    "LuxonisSource",
    "MediaType",
    "Metadata",
    "UpdateMode",
]
