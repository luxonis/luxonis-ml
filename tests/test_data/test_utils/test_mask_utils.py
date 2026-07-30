import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from luxonis_ml.data.utils import rgb_to_bool_masks
from luxonis_ml.typing import RGB

colors = st.tuples(*3 * [st.integers(0, 255)])


@st.composite
def masks_and_class_colors(
    draw: st.DrawFn,
) -> tuple[np.ndarray, dict[str, RGB]]:
    """Draw an RGB segmentation mask together with its class palette.

    The colors are unique so that every pixel belongs to at most one
    class. The mask is painted from the class colors plus a few extra
    colors, which end up as background pixels unless they collide with a
    class color.
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
    class_colors = dict(zip(names, palette, strict=True))

    height = draw(st.integers(1, 6))
    width = draw(st.integers(1, 6))
    pixel_colors = [*palette, *draw(st.lists(colors, max_size=2))]
    pixels = draw(
        st.lists(
            st.sampled_from(pixel_colors),
            min_size=height * width,
            max_size=height * width,
        )
    )

    mask = np.array(pixels, dtype=np.uint8).reshape(height, width, 3)
    return mask, class_colors


@given(masks_and_class_colors())
def test_masks_match_a_direct_color_comparison(
    mask_and_colors: tuple[np.ndarray, dict[str, RGB]],
):
    segmentation_mask, class_colors = mask_and_colors

    masks = dict(rgb_to_bool_masks(segmentation_mask, class_colors))

    assert masks.keys() == class_colors.keys()
    for name, mask in masks.items():
        expected = np.all(segmentation_mask == class_colors[name], axis=-1)
        assert np.array_equal(mask, expected)


@given(masks_and_class_colors())
def test_masks_partition_every_pixel_exactly_once(
    mask_and_colors: tuple[np.ndarray, dict[str, RGB]],
):
    segmentation_mask, class_colors = mask_and_colors

    masks = dict(
        rgb_to_bool_masks(
            segmentation_mask, class_colors, add_background_class=True
        )
    )

    assert list(masks) == ["background", *class_colors]
    stacked = np.stack(list(masks.values()))
    assert np.array_equal(
        stacked.sum(axis=0), np.ones(segmentation_mask.shape[:2])
    )


@given(masks_and_class_colors())
def test_masks_are_boolean_and_keep_the_spatial_shape(
    mask_and_colors: tuple[np.ndarray, dict[str, RGB]],
):
    segmentation_mask, class_colors = mask_and_colors

    for _, mask in rgb_to_bool_masks(
        segmentation_mask, class_colors, add_background_class=True
    ):
        assert mask.dtype == np.bool_
        assert mask.shape == segmentation_mask.shape[:2]


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
