import importlib

import pytest

import luxonis_ml.data.utils as data_utils


def test_removed_collector_points_at_its_replacement():
    """``AugmentationsCollector`` was replaced by engine-side tracking.

    Both ways it used to be imported have to carry the migration hint. The
    module path is the one downstream code uses, and it would otherwise
    fail with a bare `ModuleNotFoundError`.
    """
    with pytest.raises(ImportError, match="applied_augmentations"):
        from luxonis_ml.data.utils import (
            AugmentationsCollector,
        )

    with pytest.raises(ImportError, match="applied_augmentations"):
        from luxonis_ml.data.utils.augmentations_collector import (  # noqa: F401
            AugmentationsCollector,
        )


def test_placeholder_module_is_importable():
    """The placeholder must not raise when the module itself is imported.

    Test collection imports every module of the package, so raising on
    import failed the whole test run rather than the one import the hint is
    meant for.
    """
    importlib.import_module("luxonis_ml.data.utils.augmentations_collector")


def test_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="NotAnApi"):
        _ = data_utils.NotAnApi
