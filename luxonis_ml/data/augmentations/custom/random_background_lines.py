import random
from typing import Any

import albumentations as A
import cv2
import numpy as np
from typing_extensions import override


class RandomBackgroundLines(A.DualTransform):
    """Randomly draws lines on the background of an image, avoiding foreground objects.

    @type num_lines: tuple
    @param num_lines: Range of number of lines to draw. Defaults to (3, 10).
    @type line_thickness: tuple
    @param line_thickness: Range of line thickness. Defaults to (10, 50).
    @type line_length: tuple
    @param line_length: Range of line lengths as a fraction of the diagonal of the image. Defaults to (0.1, 0.5).
    @type gray_range: tuple
    @param gray_range: Range of grayscale values for the line color (0=black, 255=white). Defaults to (0, 80).
    @type p: float
    @param p: Probability of applying the transform. Defaults to 0.5.
    """

    def __init__(
        self,
        num_lines: tuple = (3, 10),
        line_thickness: tuple = (10, 50),
        line_length: tuple = (0.2, 0.6),
        gray_range: tuple = (0, 96),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_lines = num_lines
        self.line_thickness = line_thickness
        self.line_length = line_length
        self.gray_range = gray_range

    @override
    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Updates augmentation parameters with the necessary metadata.

        @param params: The existing augmentation parameters dictionary.
        @type params: Dict[str, Any]
        @param data: The data dictionary.
        @type data: Dict[str, Any]
        @return: Additional parameters for the augmentation.
        @rtype: Dict[str, Any]
        """

        seg_mask = data.get("_segmentation")
        if seg_mask.shape[-1] != 1:
            seg_mask = seg_mask[:, :, 0]
        return {
            "seg_mask": seg_mask,
        }

    def apply(
        self, image: np.ndarray, seg_mask: np.ndarray, **params
    ) -> np.ndarray:
        """Applies the random background lines augmentation to the image.

        @type image: np.ndarray
        @param image: The input image.
        @type seg_mask: np.ndarray
        @param seg_mask: The segmentation mask.
        @return: The augmented image with lines drawn on the background.
        @rtype: np.ndarray
        """
        if seg_mask is None:
            raise ValueError("Mask is None. Please provide a valid mask.")

        h, w = image.shape[:2]
        diagonal = np.sqrt(h**2 + w**2)

        lines_canvas = np.zeros_like(image)

        num_lines = random.randint(self.num_lines[0], self.num_lines[1])

        thicknesses = [
            random.randint(self.line_thickness[0], self.line_thickness[1])
            for _ in range(num_lines)
        ]

        lengths = [
            random.uniform(self.line_length[0], self.line_length[1]) * diagonal
            for _ in range(num_lines)
        ]

        start_points = [
            (random.randint(0, w - 1), random.randint(0, h - 1))
            for _ in range(num_lines)
        ]

        angles = [
            random.choice([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
            for _ in range(num_lines)
        ]

        gray_values = [
            random.randint(self.gray_range[0], self.gray_range[1])
            for _ in range(num_lines)
        ]

        for i in range(num_lines):
            x1, y1 = start_points[i]
            angle = angles[i]
            length = lengths[i]
            thickness = thicknesses[i]

            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))

            if x2 < 0:
                y2 = int(
                    y1 + (0 - x1) * np.tan(angle) if angle != np.pi / 2 else y1
                )
                x2 = 0
            elif x2 >= w:
                y2 = int(
                    y1 + (w - 1 - x1) * np.tan(angle)
                    if angle != np.pi / 2
                    else y1
                )
                x2 = w - 1

            if y2 < 0:
                x2 = int(
                    x1 + (0 - y1) / np.tan(angle) if np.tan(angle) != 0 else x1
                )
                y2 = 0
            elif y2 >= h:
                x2 = int(
                    x1 + (h - 1 - y1) / np.tan(angle)
                    if np.tan(angle) != 0
                    else x1
                )
                y2 = h - 1

            color = (gray_values[i], gray_values[i], gray_values[i])
            cv2.line(lines_canvas, (x1, y1), (x2, y2), color, thickness)

        foreground_mask = seg_mask < 0.5

        background_mask = ~foreground_mask

        result = image.copy()

        lines_exist = np.any(lines_canvas > 0, axis=2)
        update_mask = background_mask & lines_exist

        if np.any(update_mask):
            result[update_mask] = lines_canvas[update_mask]

        return result

    @override
    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """Keep the mask unchanged during augmentation.

        @type mask: np.ndarray
        @param mask: The input segmentation mask.
        @return: The unmodified mask.
        @rtype: np.ndarray
        """
        return mask
