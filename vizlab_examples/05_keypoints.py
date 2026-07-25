"""Keypoints + COCO-17 skeleton, composed with a bounding box.

Shows a person pose: joints sized by confidence, limbs from the COCO-17 skeleton,
and a box around the instance — all added to the same image to demonstrate
composition. The third column of each keypoint is a confidence in ``[0, 1]``.
"""

import numpy as np
from _common import gradient, save

from luxonis_ml.vizlab import COCO_17, BBox, Image, Keypoints

# COCO-17 order: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
POSE = np.array(
    [
        [250, 92, 0.99],  # nose
        [262, 82, 0.98],  # left_eye
        [238, 82, 0.97],  # right_eye
        [276, 90, 0.90],  # left_ear
        [224, 90, 0.72],  # right_ear
        [296, 152, 0.99],  # left_shoulder
        [204, 152, 0.98],  # right_shoulder
        [316, 222, 0.95],  # left_elbow
        [184, 222, 0.93],  # right_elbow
        [326, 288, 0.88],  # left_wrist
        [174, 288, 0.60],  # right_wrist
        [282, 292, 0.97],  # left_hip
        [218, 292, 0.97],  # right_hip
        [290, 402, 0.94],  # left_knee
        [210, 402, 0.94],  # right_knee
        [294, 506, 0.85],  # left_ankle
        [206, 506, 0.83],  # right_ankle
    ],
    dtype=float,
)


def main() -> None:
    """Render a keypoint skeleton and a matching box on one image."""
    img = Image(gradient(500, 600, hue=0.66))
    img.add(BBox((150, 60, 200, 485), label="person", score=0.98))
    img.add(Keypoints(POSE, skeleton=COCO_17, label="person"))
    save(img, "05_keypoints.png")


if __name__ == "__main__":
    main()
