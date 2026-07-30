from copy import deepcopy

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.typing import Labels

from .label_strategies import (
    ALL_TASK_TYPES,
    BATCH_CONFIGS,
    PER_INSTANCE_TASK_TYPES,
    SampleSpec,
    TaskGroupSpec,
    build_labels,
    sample_specs,
)

# Derandomized so that a failure is reproducible from the test id alone.
combination_settings = settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def run(spec: SampleSpec) -> tuple[dict[str, np.ndarray], Labels]:
    """Build an engine from ``spec`` and push a full batch through it."""
    engine = AlbumentationsEngine(
        spec.height,
        spec.width,
        spec.targets,
        spec.n_classes,
        spec.source_names,
        spec.config,
        seed=42,
    )
    labels: Labels = {}
    for group in spec.groups:
        labels.update(build_labels(group, spec.image_height, spec.image_width))
    images = spec.images()
    batch = [(images, deepcopy(labels)) for _ in range(engine.batch_size)]
    return engine.apply(batch)


def assert_group_is_consistent(
    group: TaskGroupSpec, out_labels: Labels
) -> None:
    """Every per-instance label in a group must describe the same instances."""
    counts = {}
    for task_type in group.task_types:
        if task_type not in PER_INSTANCE_TASK_TYPES:
            continue
        task = group.task(task_type)
        if task not in out_labels:
            continue
        array = out_labels[task]
        if task_type == "keypoints":
            assert array.ndim == 2, f"{task} should be (N, 3K)"
            assert array.shape[1] == group.n_keypoints * 3, (
                f"{task} lost keypoints: {array.shape}"
            )
        counts[task_type] = array.shape[0]

    if len(set(counts.values())) > 1:
        pytest.fail(
            f"task group '{group.name}' returned mismatched instance "
            f"counts: {counts}"
        )


@given(spec=sample_specs())
@combination_settings
def test_any_label_combination_round_trips(spec: SampleSpec) -> None:
    out_images, out_labels = run(spec)

    assert set(out_images) == set(spec.source_names)
    for name, image in out_images.items():
        assert image.shape[:2] == (spec.height, spec.width), (
            f"source '{name}' was not resized to the requested output size"
        )
        assert image.shape[2] == spec.source_channels[name], (
            f"source '{name}' changed channel count"
        )

    for group in spec.groups:
        assert_group_is_consistent(group, out_labels)

        if "segmentation" in group.task_types:
            mask = out_labels[group.task("segmentation")]
            assert mask.shape == (group.n_classes, spec.height, spec.width)

        if "classification" in group.task_types:
            classes = out_labels[group.task("classification")]
            assert classes.shape == (group.n_classes,)


@given(spec=sample_specs())
@combination_settings
def test_instances_never_outnumber_their_bboxes(spec: SampleSpec) -> None:
    """Per-instance labels are capped by the boxes that survived."""
    _, out_labels = run(spec)

    for group in spec.groups:
        if "boundingbox" not in group.task_types:
            continue
        boxes = out_labels.get(group.task("boundingbox"))
        if boxes is None:
            continue
        for task_type in PER_INSTANCE_TASK_TYPES:
            if task_type not in group.task_types:
                continue
            array = out_labels.get(group.task(task_type))
            if array is None:
                continue
            assert array.shape[0] == boxes.shape[0], (
                f"'{group.task(task_type)}' has {array.shape[0]} rows for "
                f"{boxes.shape[0]} boxes"
            )


@given(
    task_type=st.sampled_from(ALL_TASK_TYPES),
    config=st.sampled_from(BATCH_CONFIGS),
)
@combination_settings
def test_every_label_type_works_on_its_own(
    task_type: str, config: list[dict]
) -> None:
    """Each advertised label type is usable as the only label in a dataset."""
    spec = SampleSpec(
        groups=[
            TaskGroupSpec(
                name="solo",
                task_types=[task_type],
                n_instances=2,
                n_classes=2,
                n_keypoints=2,
            )
        ],
        source_channels={"image": 3},
        height=64,
        width=64,
        image_height=64,
        image_width=64,
        config=config,
    )

    _, out_labels = run(spec)

    task = spec.groups[0].task(task_type)
    assert task in out_labels, (
        f"'{task}' disappeared from the output; a task present in the input "
        f"must always be reported, even when empty"
    )


@given(spec=sample_specs(min_groups=2, max_groups=3))
@combination_settings
def test_task_groups_do_not_interfere(spec: SampleSpec) -> None:
    """Adding a task group must not change what the other groups return."""
    _, combined = run(spec)

    for group in spec.groups:
        isolated_spec = SampleSpec(
            groups=[group],
            source_channels=spec.source_channels,
            height=spec.height,
            width=spec.width,
            image_height=spec.image_height,
            image_width=spec.image_width,
            config=spec.config,
        )
        _, isolated = run(isolated_spec)

        for task in group.tasks:
            assert (task in combined) == (task in isolated), (
                f"'{task}' is reported differently depending on whether "
                f"other task groups are present"
            )
            if task in combined:
                assert combined[task].shape == isolated[task].shape, (
                    f"'{task}' has shape {combined[task].shape} alongside "
                    f"other groups but {isolated[task].shape} on its own"
                )
