import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from luxonis_ml.nn_archive.utils import infer_layout

# Layouts are lettered from "C" through "Z".
MAX_DIMS = 24


@st.composite
def valid_shapes(draw: st.DrawFn) -> list[int]:
    first = draw(st.integers(1, 512))
    max_dims = MAX_DIMS if first == 1 else MAX_DIMS - 1
    rest = draw(st.lists(st.integers(1, 512), max_size=max_dims - 1))
    return [first, *rest]


shapes = valid_shapes()


@given(shape=shapes)
def test_one_unique_letter_per_dimension(shape: list[int]):
    layout = infer_layout(shape)

    assert len(layout) == len(shape)
    assert len(set(layout)) == len(layout)
    assert set(layout) <= set(string.ascii_uppercase)


@given(shape=shapes)
def test_batch_dimension(shape: list[int]):
    layout = infer_layout(shape)

    assert layout.startswith("N") == (shape[0] == 1)
    assert "N" not in layout[1:]


@given(channels=st.integers(2, 64), size=st.integers(65, 512))
def test_image_layouts(channels: int, size: int):
    """The smallest of three spatial dimensions is the channel one."""
    assert infer_layout([channels, size, size]) == "CHW"
    assert infer_layout([size, size, channels]) == "HWC"
    assert infer_layout([1, channels, size, size]) == "NCHW"
    assert infer_layout([1, size, size, channels]) == "NHWC"


@given(size=st.integers(2, 512))
def test_leading_one_is_batch(size: int):
    """Treat a leading ``1`` as a batch dimension.

    A single-channel channel-first image is not recognized as ``CHW``.
    """
    assert infer_layout([1, size, size]) == "NCD"
    assert infer_layout([size, size, 1]) == "HWC"


@given(
    shape=st.lists(
        st.integers(1, 512), min_size=MAX_DIMS + 1, max_size=MAX_DIMS + 8
    )
)
def test_too_many_dimensions(shape: list[int]):
    with pytest.raises(ValueError, match="Too many dimensions"):
        infer_layout(shape)


@given(
    shape=st.lists(st.integers(2, 512), min_size=MAX_DIMS, max_size=MAX_DIMS)
)
def test_non_batch_shape_has_one_fewer_available_letter(shape: list[int]):
    with pytest.raises(ValueError, match="Too many dimensions"):
        infer_layout(shape)


def test_empty_shape():
    with pytest.raises(ValueError, match="empty shape"):
        infer_layout([])


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
