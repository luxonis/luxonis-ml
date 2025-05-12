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
    @type p: float
    @param p: Probability of applying the transform. Defaults to 0.5.
    """

    def __init__(
        self,
        num_lines: tuple = (3, 10),
        line_thickness: tuple = (10, 50),
        line_length: tuple = (0.1, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_lines = num_lines
        self.line_thickness = line_thickness
        self.line_length = line_length

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

        result = image.copy()
        h, w = image.shape[:2]
        diagonal = np.sqrt(h**2 + w**2)

        if seg_mask is None:
            raise ValueError("Mask is None. Please provide a valid mask.")

        background_mask = seg_mask >= 0.5
        num_lines = random.randint(self.num_lines[0], self.num_lines[1])

        for _ in range(num_lines):
            thickness = random.randint(
                self.line_thickness[0], self.line_thickness[1]
            )
            length = (
                random.uniform(self.line_length[0], self.line_length[1])
                * diagonal
            )

            for _ in range(20):
                background_points = np.where(background_mask)
                if len(background_points[0]) == 0:
                    continue

                idx = random.randint(0, len(background_points[0]) - 1)
                y1 = background_points[0][idx]
                x1 = background_points[1][idx]

                angle = random.choice(
                    [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
                )

                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))

                if x2 < 0:
                    y2 = int(
                        y1 + (0 - x1) * np.tan(angle)
                        if angle != np.pi / 2
                        else y1
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
                        x1 + (0 - y1) / np.tan(angle)
                        if np.tan(angle) != 0
                        else x1
                    )
                    y2 = 0
                elif y2 >= h:
                    x2 = int(
                        x1 + (h - 1 - y1) / np.tan(angle)
                        if np.tan(angle) != 0
                        else x1
                    )
                    y2 = h - 1

                line_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(line_mask, (x1, y1), (x2, y2), 1, thickness)

                foreground_mask = seg_mask < 0.5
                if np.any(np.logical_and(line_mask > 0, foreground_mask)):
                    continue

                color = (0, 0, 0)
                cv2.line(result, (x1, y1), (x2, y2), color, thickness)
                break

        return result
