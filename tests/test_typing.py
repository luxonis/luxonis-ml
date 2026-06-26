import numpy as np

from luxonis_ml.typing import (
    LoaderOutput,
    Params,
    all_not_none,
    any_not_none,
    check_type,
)


def test_all_not_none():
    assert all_not_none([1, 2, 3])
    assert not all_not_none([1, 2, None])
    assert not all_not_none([None, None, None])
    assert all_not_none([])


def test_any_not_none():
    assert any_not_none([1, 2, 3])
    assert any_not_none([1, 2, None])
    assert not any_not_none([None, None, None])
    assert not any_not_none([])


def test_check_type():
    assert check_type(1, int)
    assert not check_type(1, str)
    assert check_type([1, 2, 3], list[int])
    assert not check_type([1, 2, 3], list[str])


def test_loader_output_tuple_compatibility():
    image = np.empty((100, 100, 3), dtype=np.uint8)
    labels = {"task/boundingbox": np.empty((5, 4), dtype=np.float32)}
    metadata: Params = {"source": "camera-a"}

    output = LoaderOutput({"image": image}, labels, metadata)

    assert len(output) == 2
    unpacked_image, unpacked_labels = output
    assert unpacked_image is image
    assert unpacked_labels is labels
    assert output[0] is image
    assert output[1] is labels
    assert output.images == {"image": image}
    assert output.labels is labels
    assert output.metadata is metadata
    assert output.image is image

    right = np.empty((10, 10, 3), dtype=np.uint8)
    multi_output = LoaderOutput({"left": image, "right": right}, labels, {})
    assert multi_output[0] == multi_output.images
    assert multi_output.image is image
