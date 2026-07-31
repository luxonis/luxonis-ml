import pytest

import luxonis_ml.data.utils as data_utils


def test_removed_collector_points_at_its_replacement():
    """``AugmentationsCollector`` was replaced by engine-side tracking.

    Both ways it used to be imported have to carry the migration hint. The
    module path is the one downstream code uses, and it would otherwise
    fail with a bare `ModuleNotFoundError`.
    """
    with pytest.raises(ImportError, match="applied_augmentations"):
        from luxonis_ml.data.utils import (  # noqa: F401
            AugmentationsCollector,
        )

    with pytest.raises(ImportError, match="applied_augmentations"):
        import luxonis_ml.data.utils.augmentations_collector  # noqa: F401


def test_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="NotAnApi"):
        _ = data_utils.NotAnApi
