"""Placeholder for the removed ``AugmentationsCollector``.

The module is kept so that code importing the collector by its module path
fails with the migration hint below rather than with a bare
``ModuleNotFoundError`` that says nothing about the replacement.
"""

raise ImportError(
    "'AugmentationsCollector' was removed. Augmentation provenance "
    "is now tracked by the engine itself: read it from "
    "'LoaderOutput.metadata[\"augmentations\"]', or directly from "
    "'AugmentationEngine.applied_augmentations'."
)
