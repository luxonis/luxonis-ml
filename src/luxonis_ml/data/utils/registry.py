"""This module implements a metaclass for automatic registration of classes."""

from luxonis_ml.utils.registry import Registry

DATASETS = Registry(name="datasets")
