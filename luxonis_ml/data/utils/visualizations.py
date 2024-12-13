import colorsys
import hashlib
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import matplotlib.colors
import numpy as np

from luxonis_ml.data.utils import get_task_name, task_type_iterator
from luxonis_ml.typing import Color, Labels

font = cv2.FONT_HERSHEY_SIMPLEX


def resolve_color(color: Color) -> Tuple[int, int, int]:
    """Resolves a color to an RGB tuple.

    @type color: Color
    @param color: The color to resolve. Can be a string, an integer or a
        tuple.
    @rtype: Tuple[int, int, int]
    @return: The RGB tuple.
    """

    def _check_range(val: int) -> None:
        if val < 0 or val > 255:
            raise ValueError(f"Color value {val} is out of range [0, 255]")

    if isinstance(color, str):
        return matplotlib.colors.to_rgb(color)
    elif isinstance(color, int):
        _check_range(color)
        return color, color, color
    else:
        for c in color:
            _check_range(c)
        return color


def rgb_to_hsb(color: Color) -> Tuple[float, float, float]:
    """Converts an RGB color to HSB.

    @type color: Color
    @param color: The color to convert.
    @rtype: Tuple[float, float, float]
    @return: The HSB tuple.
    """
    r, g, b = resolve_color(color)
    h, s, br = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360, s, br


def hsb_to_rgb(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Converts an HSB color to RGB.

    @type color: Tuple[int, int, int]
    @param color: The color to convert as an HSB tuple.
    @rtype: Tuple[int, int, int]
    @return: The RGB tuple.
    """
    h, s, b = color
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, b)
    return int(r * 255), int(g * 255), int(b * 255)


def get_contrast_color(color: Color) -> Tuple[int, int, int]:
    """Returns a contrasting color for the given RGB color.

    @type color: Color
    @param color: The color to contrast.
    @rtype: Tuple[int, int, int]
    @return: The contrasting color.
    """

    h, s, v = rgb_to_hsb(resolve_color(color))
    h = (h + 180) % 360
    return hsb_to_rgb((h, s, v))


def str_to_rgb(string: str) -> Tuple[int, int, int]:
    """Converts a string to its unique RGB color.

    @type string: str
    @param string: The string to convert.
    @rtype: Tuple[int, int, int]
    @return: The RGB tuple.
    """
    h = int(hashlib.md5(string.encode()).hexdigest(), 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF

    return r, g, b


def draw_dashed_rectangle(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Color,
    thickness: int = 1,
    dash_length: int = 10,
) -> None:
    """Draws a dashed rectangle on the image.

    Adheres to OpenCV's rectangle drawing convention.

    @type image: np.ndarray
    @param image: The image to draw on.
    @type pt1: Tuple[int, int]
    @param pt1: The top-left corner of the rectangle.
    @type pt2: Tuple[int, int]
    @param pt2: The bottom-right corner of the rectangle.
    @type color: Color
    @param color: The color of the rectangle.
    @type thickness: int
    @param thickness: The thickness of the rectangle. Default is 1.
    @type dash_length: int
    @param dash_length: The length of the dashes. Default is 10.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    def draw_dashed_line(p1: Tuple[int, int], p2: Tuple[int, int]) -> None:
        line_length = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        dashes = [
            (i, i + dash_length)
            for i in range(0, line_length, 2 * dash_length)
        ]
        for start, end in dashes:
            if end > line_length:
                end = line_length
            start_point = (
                int(p1[0] + (p2[0] - p1[0]) * start / line_length),
                int(p1[1] + (p2[1] - p1[1]) * start / line_length),
            )
            end_point = (
                int(p1[0] + (p2[0] - p1[0]) * end / line_length),
                int(p1[1] + (p2[1] - p1[1]) * end / line_length),
            )
            cv2.line(
                image, start_point, end_point, resolve_color(color), thickness
            )

    draw_dashed_line((x1, y1), (x2, y1))
    draw_dashed_line((x2, y1), (x2, y2))
    draw_dashed_line((x2, y2), (x1, y2))
    draw_dashed_line((x1, y2), (x1, y1))


def draw_cross(
    img: np.ndarray,
    center: Tuple[int, int],
    size: int = 5,
    color: Color = 0,
    thickness: int = 1,
) -> None:
    """Draws a cross on the image.

    @type img: np.ndarray
    @param img: The image to draw on.
    @type center: Tuple[int, int]
    @param center: The center of the cross.
    @type size: int
    @param size: The size of the cross. Default is 5.
    @type color: Color
    @param color: The color of the cross. Default is black.
    @type thickness: int
    @param thickness: The thickness of the cross. Default is 1.
    """
    x, y = center
    color = resolve_color(color)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def create_text_image(
    text: str,
    width: int,
    height: int,
    font_size: float = 0.7,
    bg_color: Color = 255,
    text_color: Color = 0,
) -> np.ndarray:
    """Creates an image with the given text centered in the image.

    @type text: str
    @param text: The text to display.
    @type width: int
    @param width: The width of the image.
    @type height: int
    @param height: The height of the image.
    @type font_size: float
    @param font_size: The font size of the text. Default is 0.7.
    @type bg_color: Tuple[int, int, int]
    @param bg_color: The background color of the image. Default is
        white.
    @type text_color: Tuple[int, int, int]
    @param text_color: The color of the text. Default is black.
    """
    img = np.full((height, width, 3), resolve_color(bg_color), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size = cv2.getTextSize(text, font, font_size, 1)[0]

    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(
        img,
        text,
        (text_x, text_y),
        fontFace=font,
        fontScale=font_size,
        color=resolve_color(text_color),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return img


def concat_images(
    image_dict: Dict[str, np.ndarray],
    padding: int = 10,
    label_height: int = 30,
) -> np.ndarray:
    """Concatenates images into a single image with labels.

    It will attempt to create a square grid of images.

    @type image_dict: Dict[str, np.ndarray]
    @param image_dict: A dictionary mapping image names to images.
    @type padding: int
    @param padding: The padding between images. Default is 10.
    @type label_height: int
    @param label_height: The height of the label. Default
    @rtype: np.ndarray
    @return: The concatenated image.
    """
    n_images = len(image_dict)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)

    max_h = max(img.shape[0] for img in image_dict.values())
    max_w = max(img.shape[1] for img in image_dict.values())

    cell_height = max_h + 2 * padding + label_height
    cell_width = max_w + 2 * padding

    output = np.full(
        (cell_height * n_rows, cell_width * n_cols, 3), 255, dtype=np.uint8
    )

    for idx, (name, img) in enumerate(image_dict.items()):
        i = idx // n_cols
        j = idx % n_cols

        y_start = i * cell_height
        x_start = j * cell_width

        label = create_text_image(name, cell_width, label_height)
        output[
            y_start : y_start + label_height, x_start : x_start + cell_width
        ] = label

        h, w = img.shape[:2]
        y_img = y_start + label_height + padding
        x_img = x_start + padding
        output[y_img : y_img + h, x_img : x_img + w] = img

    return output


def visualize(
    image: np.ndarray,
    labels: Labels,
    class_names: Dict[str, List[str]],
    blend_all: bool = False,
) -> np.ndarray:
    """Visualizes the labels on the image.

    @type image: np.ndarray
    @param image: The image to visualize.
    @type labels: Labels
    @param labels: The labels to visualize.
    @type class_names: Dict[str, List[str]]
    @param class_names: A dictionary mapping task names to a list of
        class names.
    @type blend_all: bool
    @param blend_all: Whether to blend all labels (apart from semantic
        segmentations) into a single image. This means mixing labels
        belonging to different tasks. Default is False.
    @rtype: np.ndarray
    @return: The visualized image.
    """
    h, w, _ = image.shape
    images = {"image": image}

    def create_mask(
        image: np.ndarray, arr: np.ndarray, task_name: str, is_instance: bool
    ) -> np.ndarray:
        mask_viz = np.zeros((h, w, 3)).astype(np.uint8)
        for i, mask in enumerate(arr):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if is_instance:
                mask_viz[mask > 0] = str_to_rgb(
                    class_names[task_name][int(mask.max() - 1)]
                )
            else:
                mask_viz[mask == 1] = (
                    str_to_rgb(class_names[task_name][i])
                    if (i != 0 or len(arr) == 1)
                    else (0, 0, 0)
                )

        binary_mask = (mask_viz > 0).astype(np.uint8)

        return np.where(
            binary_mask > 0,
            cv2.addWeighted(image, 0.4, mask_viz, 0.6, 0),
            image,
        )

    bbox_classes = defaultdict(list)

    for task, arr in task_type_iterator(labels, "segmentation"):
        task_name = get_task_name(task)
        images[task_name] = create_mask(
            image, arr, task_name, is_instance=False
        )

    for task, arr in task_type_iterator(labels, "instance_segmentation"):
        task_name = get_task_name(task)
        image_name = task_name if not blend_all else "labels"
        curr_image = images.get(image_name, image.copy())
        images[image_name] = create_mask(
            curr_image, arr, task_name, is_instance=True
        )

    for task, arr in task_type_iterator(labels, "boundingbox"):
        task_name = get_task_name(task)
        image_name = task_name if not blend_all else "labels"
        curr_image = images.get(image_name, image.copy())

        draw_function = cv2.rectangle

        is_sublabel = len(task.split("/")) > 2

        if is_sublabel:
            draw_function = draw_dashed_rectangle

        arr[:, [1, 3]] *= w
        arr[:, [2, 4]] *= h
        arr[:, 3] += arr[:, 1]
        arr[:, 4] += arr[:, 2]
        arr = arr.astype(int)

        for box in arr:
            class_id = int(box[0])
            bbox_classes[task_name].append(class_id)
            color = str_to_rgb(class_names[task_name][class_id])
            draw_function(
                curr_image,
                (box[1], box[2]),
                (box[3], box[4]),
                color,
                thickness=2,
            )
        images[image_name] = curr_image

    for task, arr in task_type_iterator(labels, "keypoints"):
        task_name = get_task_name(task)
        image_name = task_name if not blend_all else "labels"
        curr_image = images.get(image_name, image.copy())

        task_classes = class_names[task_name]

        for i, kp in enumerate(arr):
            kp = kp.reshape(-1, 3)
            if len(bbox_classes[task_name]) > i:
                class_id = bbox_classes[task_name][i]
                color = get_contrast_color(str_to_rgb(task_classes[class_id]))
            else:
                color = (255, 0, 0)
            for k in kp:
                visibility = int(k[-1])
                if visibility == 0:
                    continue

                if visibility == 2:
                    draw_function = cv2.circle
                    size = 2
                else:
                    draw_function = draw_cross
                    size = 5

                draw_function(
                    curr_image,
                    (int(k[0] * w), int(k[1] * h)),
                    size,
                    color=color,
                    thickness=2,
                )
        images[image_name] = curr_image

    return concat_images(images)
