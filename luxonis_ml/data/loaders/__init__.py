from .base_loader import LOADERS_REGISTRY, BaseLoader, Labels, LuxonisLoaderOutput
from .luxonis_loader import LuxonisLoader

__all__ = [
    "BaseLoader",
    "Labels",
    "LuxonisLoader",
    "LuxonisLoaderOutput",
    "LOADERS_REGISTRY",
]
