import pytest

from luxonis_ml.nn_archive.utils import infer_layout


def test_infer_layout():
    assert infer_layout([1, 3, 256, 256]) == "NCHW"
    assert infer_layout([1, 1, 256, 256]) == "NCHW"
    assert infer_layout([1, 4, 256, 256]) == "NCHW"
    assert infer_layout([1, 19, 7, 8]) == "NCDE"
    assert infer_layout([256, 256, 3]) == "HWC"
    assert infer_layout([256, 256, 1]) == "HWC"
    assert infer_layout([256, 256, 12]) == "HWC"

    with pytest.raises(ValueError, match="Too many dimensions"):
        infer_layout(list(range(30)))
