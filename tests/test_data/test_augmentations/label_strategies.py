"""Hypothesis strategies for the label types the engine advertises.

Samples are built *self-consistent*: within one task group every per-instance
label describes the same instances, so a generated sample is something
`LuxonisLoader` could plausibly hand over.
"""

from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st

from luxonis_ml.typing import Labels

# "metadata" is spelled as a bare task type here and expanded to
# "metadata/<field>" when the task string is built.
PER_INSTANCE_TASK_TYPES = [
    "boundingbox",
    "keypoints",
    "instance_segmentation",
    "array",
    "metadata",
]
WHOLE_IMAGE_TASK_TYPES = ["segmentation", "classification"]
ALL_TASK_TYPES = PER_INSTANCE_TASK_TYPES + WHOLE_IMAGE_TASK_TYPES


@dataclass
class TaskGroupSpec:
    """One task group: a set of task types over a shared instance count."""

    name: str
    task_types: list[str]
    n_instances: int
    n_classes: int
    n_keypoints: int

    @staticmethod
    def _suffix(task_type: str) -> str:
        return "metadata/id" if task_type == "metadata" else task_type

    def task(self, task_type: str) -> str:
        return f"{self.name}/{self._suffix(task_type)}"

    @property
    def tasks(self) -> dict[str, str]:
        return {
            self.task(task_type): self._suffix(task_type)
            for task_type in self.task_types
        }


@dataclass
class SampleSpec:
    """A full engine configuration plus the batch to feed it."""

    groups: list[TaskGroupSpec]
    source_channels: dict[str, int]
    height: int
    width: int
    image_height: int
    image_width: int
    config: list[dict]

    @property
    def source_names(self) -> list[str]:
        return list(self.source_channels)

    @property
    def targets(self) -> dict[str, str]:
        targets = {}
        for group in self.groups:
            targets.update(group.tasks)
        return targets

    @property
    def n_classes(self) -> dict[str, int]:
        return {
            task: group.n_classes
            for group in self.groups
            for task in group.tasks
        }

    def images(self) -> dict[str, np.ndarray]:
        return {
            name: np.full(
                (self.image_height, self.image_width, channels),
                fill_value=(i + 1) * 20,
                dtype=np.uint8,
            )
            for i, (name, channels) in enumerate(self.source_channels.items())
        }


def _instance_boxes(n_instances: int) -> np.ndarray:
    """Non-overlapping boxes laid out on a diagonal, well inside the image."""
    boxes = []
    for i in range(n_instances):
        offset = 0.05 + i * 0.18
        boxes.append([0.0, offset, offset, 0.12, 0.12])
    return np.array(boxes, dtype=np.float64).reshape(n_instances, 5)


def build_labels(group: TaskGroupSpec, height: int, width: int) -> Labels:
    """Build one group's labels, consistent across every task type in it."""
    labels: Labels = {}
    n = group.n_instances
    boxes = _instance_boxes(n)

    for task_type in group.task_types:
        task = group.task(task_type)

        if task_type == "boundingbox":
            labels[task] = boxes.copy()

        elif task_type == "keypoints":
            keypoints = np.zeros((n, group.n_keypoints * 3))
            for i in range(n):
                for k in range(group.n_keypoints):
                    keypoints[i, k * 3] = boxes[i, 1] + 0.06
                    keypoints[i, k * 3 + 1] = boxes[i, 2] + 0.06
                    keypoints[i, k * 3 + 2] = 2.0
            labels[task] = keypoints

        elif task_type == "instance_segmentation":
            masks = np.zeros((n, height, width), dtype=np.uint8)
            for i in range(n):
                r0 = int(boxes[i, 2] * height)
                c0 = int(boxes[i, 1] * width)
                r1 = max(r0 + 1, int((boxes[i, 2] + boxes[i, 4]) * height))
                c1 = max(c0 + 1, int((boxes[i, 1] + boxes[i, 3]) * width))
                masks[i, r0:r1, c0:c1] = i + 1
            labels[task] = masks

        elif task_type == "array":
            labels[task] = np.arange(n * 2, dtype=np.float64).reshape(n, 2)

        elif task_type == "metadata":
            labels[task] = np.arange(n, dtype=np.float64)

        elif task_type == "segmentation":
            masks = np.zeros((group.n_classes, height, width), dtype=np.uint8)
            masks[0, : height // 2, : width // 2] = 1
            labels[task] = masks

        elif task_type == "classification":
            classes = np.zeros(group.n_classes)
            classes[0] = 1.0
            labels[task] = classes

    return labels


task_types = st.lists(
    st.sampled_from(ALL_TASK_TYPES),
    min_size=1,
    max_size=len(ALL_TASK_TYPES),
    unique=True,
).map(sorted)


@st.composite
def task_groups(draw: st.DrawFn, name: str) -> TaskGroupSpec:
    return TaskGroupSpec(
        name=name,
        task_types=draw(task_types),
        n_instances=draw(st.integers(min_value=0, max_value=3)),
        n_classes=draw(st.integers(min_value=1, max_value=3)),
        n_keypoints=draw(st.integers(min_value=1, max_value=3)),
    )


BATCH_CONFIGS = [
    [],
    [{"name": "HorizontalFlip", "params": {"p": 1.0}}],
    [{"name": "RandomBrightnessContrast", "params": {"p": 1.0}}],
    [{"name": "Affine", "params": {"p": 1.0, "rotate": 15}}],
    [{"name": "MixUp", "params": {"p": 1.0}}],
    [{"name": "Mosaic4", "params": {"p": 1.0, "height": 64, "width": 64}}],
    [
        {"name": "Mosaic4", "params": {"p": 1.0, "height": 64, "width": 64}},
        {"name": "MixUp", "params": {"p": 1.0}},
    ],
    [
        {"name": "MixUp", "params": {"p": 1.0}},
        {"name": "HorizontalFlip", "params": {"p": 1.0}},
        {"name": "RandomBrightnessContrast", "params": {"p": 1.0}},
    ],
]


@st.composite
def sample_specs(
    draw: st.DrawFn, min_groups: int = 1, max_groups: int = 2
) -> SampleSpec:
    n_groups = draw(st.integers(min_value=min_groups, max_value=max_groups))
    groups = [draw(task_groups(f"group{i}")) for i in range(n_groups)]

    n_sources = draw(st.integers(min_value=1, max_value=3))
    channels = draw(
        st.lists(
            st.sampled_from([1, 3]), min_size=n_sources, max_size=n_sources
        )
    )
    source_channels = {f"source{i}": channels[i] for i in range(n_sources)}

    size = draw(st.sampled_from([32, 64]))
    return SampleSpec(
        groups=groups,
        source_channels=source_channels,
        height=size,
        width=size,
        image_height=draw(st.sampled_from([48, 64])),
        image_width=draw(st.sampled_from([48, 64])),
        config=draw(st.sampled_from(BATCH_CONFIGS)),
    )
