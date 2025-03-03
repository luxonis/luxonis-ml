import cv2
import numpy as np
import pytest

from luxonis_ml.data.utils.visualizations import (
    ColorMap,
    concat_images,
    create_text_image,
    distinct_color_generator,
    draw_cross,
    draw_dashed_rectangle,
    get_contrast_color,
    hsv_to_rgb,
    resolve_color,
    rgb_to_hsv,
    str_to_rgb,
    visualize,
)


def test_distinct_color_generator():
    assert list(distinct_color_generator(10)) == [
        (48, 105, 242),
        (161, 242, 48),
        (242, 48, 218),
        (48, 242, 209),
        (242, 153, 48),
        (96, 48, 242),
        (56, 242, 48),
        (242, 48, 113),
        (48, 169, 242),
        (226, 242, 48),
    ]


def test_color_map():
    colors = ColorMap()
    assert colors["red"] == (48, 105, 242)
    assert colors[23] == (161, 242, 48)
    assert colors[(12,)] == (242, 48, 218)
    assert colors[23] == (161, 242, 48)
    assert len(colors) == 3
    assert set(colors) == {"red", 23, (12,)}


def test_resolve_color_string():
    assert resolve_color("red") == (1.0, 0.0, 0.0)
    assert resolve_color("#00FF00") == (0.0, 1.0, 0.0)


def test_resolve_color_int():
    assert resolve_color(128) == (128, 128, 128)
    with pytest.raises(ValueError, match="out of range"):
        resolve_color(300)


def test_resolve_color_tuple():
    assert resolve_color((100, 150, 200)) == (100, 150, 200)
    with pytest.raises(ValueError, match="out of range"):
        resolve_color((100, 150, 300))


def test_rgb_to_hsv():
    assert np.allclose(rgb_to_hsv((255, 0, 0)), (0.0, 1.0, 1.0))
    assert np.allclose(rgb_to_hsv((0, 255, 0)), (120.0, 1.0, 1.0))
    assert np.allclose(rgb_to_hsv((0, 0, 255)), (240.0, 1.0, 1.0))


def test_hsv_to_rgb():
    assert hsv_to_rgb((0.0, 1.0, 1.0)) == (255, 0, 0)
    assert hsv_to_rgb((120.0, 1.0, 1.0)) == (0, 255, 0)
    assert hsv_to_rgb((240.0, 1.0, 1.0)) == (0, 0, 255)


def test_get_contrast_color():
    assert get_contrast_color((255, 0, 0)) == (0, 255, 255)
    assert get_contrast_color((0, 255, 0)) == (255, 0, 255)
    assert get_contrast_color((0, 0, 255)) == (255, 255, 0)


def test_str_to_rgb():
    assert str_to_rgb("test") == (39, 180, 246)
    assert str_to_rgb("example") == (138, 229, 51)


def test_concat_images():
    image_dict = {
        "image1": np.full((50, 50, 3), 100, dtype=np.uint8),
        "image2": np.full((30, 70, 3), 150, dtype=np.uint8),
        "image3": np.full((60, 40, 3), 200, dtype=np.uint8),
    }

    result = concat_images(image_dict, padding=5, label_height=20)

    n_cols = 2  # Based on 3 images, should form a 2x2 grid
    n_rows = 2
    max_h = 60  # Tallest image
    max_w = 70  # Widest image
    cell_height = max_h + 2 * 5 + 20
    cell_width = max_w + 2 * 5
    expected_height = cell_height * n_rows
    expected_width = cell_width * n_cols

    assert result.shape == (expected_height, expected_width, 3)

    y_start = 0
    x_start = 0
    label_region = result[
        y_start : y_start + 20, x_start : x_start + cell_width
    ]
    assert np.array_equal(
        label_region, create_text_image("image1", cell_width, 20)
    )

    img_region = result[
        y_start + 20 + 5 : y_start + 20 + 5 + 50,
        x_start + 5 : x_start + 5 + 50,
    ]
    assert np.array_equal(img_region, image_dict["image1"])


def test_draw_cross():
    image = np.zeros((10, 10), dtype=np.uint8)
    draw_cross(image, (5, 5), 2, color=1, thickness=1)
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(image, expected)


def test_draw_dashed_rectangle():
    image = np.zeros((10, 10), dtype=np.uint8)
    draw_dashed_rectangle(
        image, (1, 1), (8, 8), color=1, thickness=1, dash_length=2
    )
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(image, expected)


def test_visualize():
    width, height = 20, 20
    image = np.zeros((height, width, 3), dtype=np.uint8)
    instance_mask = np.zeros((1, height, width), dtype=np.uint8)
    instance_mask[:, 0 : height // 5, 0 : height // 5] = 1
    semantic_mask = np.zeros((4, height, width), dtype=np.uint8)
    semantic_mask[
        1, height // 5 : 2 * height // 5, height // 5 : 2 * height // 5
    ] = 1
    semantic_mask[
        2, 2 * height // 5 : 3 * height // 5, 2 * height // 5 : 3 * height // 5
    ] = 1
    semantic_mask[
        3, 3 * height // 5 : 4 * height // 5, 3 * height // 5 : 4 * height // 5
    ] = 1
    semantic_mask[0, ...] = 1 - np.sum(semantic_mask[1:], axis=0)
    labels = {
        "task/boundingbox": np.array([[0.0, 0.2, 0.2, 0.65, 0.65]]),
        "task/keypoints": np.array([[0, 0, 0], [0.9, 0.1, 1], [0.5, 0.5, 2]]),
        "task/instance_segmentation": instance_mask,
        "task2/boundingbox": np.array([[0.0, 0.4, 0.4, 0.2, 0.1]]),
        "semantic/segmentation": semantic_mask,
    }

    expected_labels = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected_labels = np.stack([expected_labels] * 3, axis=-1)
    expected_semantic = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected_semantic = np.stack([expected_semantic] * 3, axis=-1)
    expected_images = {
        "image": image.copy(),
        "labels": expected_semantic + expected_labels,
    }
    classes = {
        "task": {"class_name": 0},
        "semantic": {"background": 0, "red": 1, "green": 2, "blue": 3},
        "task2": {"class_name2": 0},
    }
    image = visualize(image, labels, classes, blend_all=True)
    image = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(bool).astype(np.uint8)
    )
    expected_image = (
        cv2.cvtColor(concat_images(expected_images), cv2.COLOR_RGB2GRAY)
        .astype(bool)
        .astype(np.uint8)
    )
    assert np.array_equal(expected_image, image)
