"""Placeholder for the removed ``AugmentationsCollector``.

The module is kept so that code importing the collector by its module path
fails with the migration hint below rather than with a bare
``ModuleNotFoundError`` that says nothing about the replacement.

Importing the module itself succeeds. Raising on import would take down
every tool that walks the package module by module, `pytest` collection
included.
"""

_REMOVED_MESSAGE = (
    "'AugmentationsCollector' was removed. Augmentation provenance "
    "is now tracked by the engine itself: read it from "
    "'LoaderOutput.metadata[\"augmentations\"]', or directly from "
    "'AugmentationEngine.applied_augmentations'."
)


def __getattr__(name: str) -> object:
    if name == "AugmentationsCollector":
        # `ImportError` rather than `AttributeError`, because
        # `from ... import AugmentationsCollector` replaces an
        # `AttributeError` with its own message and the hint would be lost.
        raise ImportError(_REMOVED_MESSAGE)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
