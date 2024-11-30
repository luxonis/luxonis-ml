import hashlib
import math
from typing import Dict, Iterator, List, Tuple

import cv2
import numpy as np

from luxonis_ml.data.loaders import Labels


def _task_to_rgb(string: str) -> tuple:
    h = int(hashlib.md5(string.encode()).hexdigest(), 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = h & 0x0000FF

    return (r, g, b)


def create_text_image(
    text: str,
    width: int,
    height: int,
    font_size: float = 0.7,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
):
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
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size = cv2.getTextSize(text, font, font_size, 1)[0]

    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_size,
        text_color,
        1,
        cv2.LINE_AA,
    )

    return img


def concat_images(
    image_dict: Dict[str, np.ndarray],
    padding: int = 10,
    label_height: int = 30,
):
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


def _label_type_iterator(
    labels: Labels, label_type: str
) -> Iterator[Tuple[str, np.ndarray]]:
    for task, arr in labels.items():
        lt = task.split("/")[-1]
        if lt == label_type:
            yield task, arr


def visualize(
    image: np.ndarray, labels: Labels, class_names: Dict[str, List[str]]
) -> np.ndarray:
    """Visualizes the labels on the image.

    @type image: np.ndarray
    @param image: The image to visualize.
    @type labels: Labels
    @param labels: The labels to visualize.
    @type class_names: Dict[str, List[str]]
    @param class_names: A dictionary mapping task names to a list of
        class names.
    @rtype: np.ndarray
    @return: The visualized image.
    """
    h, w, _ = image.shape
    images = {"image": image}

    for task, arr in _label_type_iterator(labels, "boundingbox"):
        curr_image = image.copy()
        for box in arr:
            cv2.rectangle(
                curr_image,
                (int(box[1] * w), int(box[2] * h)),
                (int(box[1] * w + box[3] * w), int(box[2] * h + box[4] * h)),
                _task_to_rgb(class_names[task.split("/")[0]][int(box[0])]),
                2,
            )
        images[task] = curr_image

    for task, arr in _label_type_iterator(labels, "keypoints"):
        curr_image = image.copy()
        task_classes = class_names[task.split("/")[0]]

        for kp in arr:
            cls_ = int(kp[0])
            kp = kp[1:].reshape(-1, 3)
            for k in kp:
                cv2.circle(
                    curr_image,
                    (int(k[0] * w), int(k[1] * h)),
                    2,
                    _task_to_rgb(task_classes[cls_]),
                    2,
                )
        images[task] = curr_image

    for task, arr in _label_type_iterator(labels, "segmentation"):
        mask_viz = np.zeros((h, w, 3)).astype(np.uint8)
        for i, mask in enumerate(arr):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_viz[mask == 1] = (
                _task_to_rgb(class_names[task.split("/")[0]][i])
                if (i != 0 or len(arr) == 1)
                else (0, 0, 0)
            )
        images[task] = mask_viz

    return concat_images(images)
