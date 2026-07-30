from typing import TypeAlias

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from luxonis_ml.data.utils import rgb_to_bool_masks
from luxonis_ml.typing import RGB

colors = st.tuples(*3 * [st.integers(0, 255)])

RGBMask: TypeAlias = tuple[np.ndarray, dict[str, RGB]]


@st.composite
def rgb_masks(draw: st.DrawFn) -> RGBMask:
    """Draw an RGB segmentation mask together with its class palette.

    The class colors are unique, so every pixel belongs to at most one
    class. Pixels are painted from the palette plus a few arbitrary
    extra colors.
    """
    names = draw(
        st.lists(
            st.text(min_size=1, max_size=6),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    palette = draw(
        st.lists(colors, min_size=len(names), max_size=len(names), unique=True)
    )
    extra = draw(st.lists(colors, max_size=2))

    height = draw(st.integers(1, 6))
    width = draw(st.integers(1, 6))
    pixels = draw(
        st.lists(
            st.sampled_from([*palette, *extra]),
            min_size=height * width,
            max_size=height * width,
        )
    )
    mask = np.array(pixels, dtype=np.uint8).reshape(height, width, 3)

    return mask, dict(zip(names, palette, strict=True))


@given(rgb_mask=rgb_masks())
def test_masks_match_direct_comparison(rgb_mask: RGBMask):
    segmentation_mask, class_colors = rgb_mask

    masks = dict(rgb_to_bool_masks(segmentation_mask, class_colors))

    assert masks.keys() == class_colors.keys()
    for name, mask in masks.items():
        assert mask.dtype == np.bool_
        assert np.array_equal(
            mask, np.all(segmentation_mask == class_colors[name], axis=-1)
        )


@given(rgb_mask=rgb_masks())
def test_background_completes_the_partition(rgb_mask: RGBMask):
    segmentation_mask, class_colors = rgb_mask

    masks = dict(
        rgb_to_bool_masks(
            segmentation_mask, class_colors, add_background_class=True
        )
    )

    assert list(masks) == ["background", *class_colors]
    stacked = np.stack(list(masks.values()))
    assert stacked.shape == (len(masks), *segmentation_mask.shape[:2])
    assert np.array_equal(
        stacked.sum(axis=0), np.ones(segmentation_mask.shape[:2])
    )


def test_rgb_to_bool_masks():
    segmentation_mask = np.array(
        [
            [[0, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[0, 0, 0], [0, 255, 0], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )

    class_colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }

    expected_results = {
        "background": np.array([[True, False, False], [True, False, False]]),
        "red": np.array([[False, True, False], [False, False, False]]),
        "green": np.array([[False, False, True], [False, True, False]]),
        "blue": np.array([[False, False, False], [False, False, True]]),
    }

    for class_name, mask in rgb_to_bool_masks(
        segmentation_mask, class_colors, add_background_class=True
    ):
        assert class_name in expected_results
        assert np.array_equal(mask, expected_results[class_name])
