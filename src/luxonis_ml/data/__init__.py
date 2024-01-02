from ..guard_extras import guard_missing_extra
import pkg_resources
from luxonis_ml.data.utils.registry import DATASETS

with guard_missing_extra("data"):
    from .utils.enums import (
        MediaType,
        ImageType,
        LDFTransactionType,
        BucketType,
        BucketStorage,
    )
    from .dataset import BaseDataset, LuxonisDataset, LuxonisSource, LuxonisComponent
    from .parsers import LuxonisParser, DatasetType
    from .loader import BaseLoader, LuxonisLoader, LabelType, LuxonisLoaderOutput
    from .augmentations import Augmentations, TrainAugmentations, ValAugmentations


def load_dataset_plugins() -> None:
    """Registers any external dataset BaseDataset class plugins."""
    for entry_point in pkg_resources.iter_entry_points("dataset_plugins"):
        plugin_class = entry_point.load()
        DATASETS.register_module(module=plugin_class)


load_dataset_plugins()

__all__ = [
    "BaseDataset",
    "LuxonisDataset",
    "LuxonisSource",
    "LuxonisComponent",
    "LuxonisParser",
    "DatasetType",
    "BaseLoader",
    "LuxonisLoader",
    "LabelType",
    "LuxonisLoaderOutput",
    "Augmentations",
    "TrainAugmentations",
    "ValAugmentations",
    "MediaType",
    "ImageType",
    "LDFTransactionType",
    "BucketType",
    "BucketStorage",
]
