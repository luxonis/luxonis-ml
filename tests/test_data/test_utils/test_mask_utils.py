import numpy as np

from luxonis_ml.data.utils import rgb_to_bool_masks


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
