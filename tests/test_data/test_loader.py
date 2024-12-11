import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader


def test_invalid():
    with pytest.raises(FileNotFoundError):
        LuxonisLoader(LuxonisDataset("non-existent"))
