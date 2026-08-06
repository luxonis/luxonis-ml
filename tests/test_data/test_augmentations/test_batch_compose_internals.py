from collections.abc import Iterator

import albumentations as A
import numpy as np
import pytest
from loguru import logger

from luxonis_ml.data.augmentations import BatchCompose, BatchTransform, MixUp

from .helpers import KeepFirstSample

IMAGE = np.zeros((16, 16, 3), dtype=np.uint8)


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    """Collect loguru warnings, which pytest's caplog does not see."""
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


class PushFirstBoxOutOfFrame(KeepFirstSample):
    """Moves the first box outside the image so filtering drops it.

    Indices are stamped before filtering, so the surviving boxes then carry
    the non-contiguous indices that compaction exists to handle.
    """

    def apply_to_bboxes(
        self, bboxes_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        boxes = bboxes_batch[0].copy()
        boxes[0, :4] = [1.2, 1.2, 1.5, 1.5]
        return boxes


class ReturnsFlatBoxes(KeepFirstSample):
    def apply_to_bboxes(self, _batch: list[np.ndarray], **_) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0])


class DropsFirstBox(KeepFirstSample):
    """Filters a box inside the transform, leaving the labels untouched.

    The surviving boxes look untouched from the outside, so nothing but the
    instance count gives the mismatch away.
    """

    def apply_to_bboxes(
        self, bboxes_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return bboxes_batch[0][1:]


def compose(
    associations: dict[str, dict[str, str]],
    additional_targets: dict[str, str] | None = None,
    transform: BatchTransform | None = None,
) -> BatchCompose:
    return BatchCompose(
        [transform or PushFirstBoxOutOfFrame()],
        bbox_params=A.BboxParams(format="albumentations", min_visibility=0.5),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        additional_targets=additional_targets or {},
        bbox_associations=associations,
    )


def boxes(n: int) -> np.ndarray:
    return np.array(
        [
            [0.1 * i, 0.1 * i, 0.1 * i + 0.2, 0.1 * i + 0.2, 0.0, float(i)]
            for i in range(n)
        ]
    )


def batch(**fields: np.ndarray) -> list[dict[str, np.ndarray]]:
    return [
        {"image": IMAGE, **{k: v.copy() for k, v in fields.items()}}
        for _ in range(2)
    ]


def test_batch_size_must_match_the_composition() -> None:
    composition = BatchCompose([KeepFirstSample()])

    with pytest.raises(ValueError, match="Batch size must be equal to 2"):
        composition([{"image": IMAGE}])


def test_bbox_fields_absent_from_the_data_are_skipped() -> None:
    composition = compose({"bboxes": {"metadata": "metadata"}})

    output = composition(batch(metadata=np.array([1.0])))

    assert output["metadata"].tolist() == [1.0]


def test_one_dimensional_bboxes_are_rejected() -> None:
    composition = compose({"bboxes": {}}, transform=ReturnsFlatBoxes())

    with pytest.raises(ValueError, match="must be a 2D array"):
        composition(batch(bboxes=boxes(2)))


def test_associated_fields_absent_from_the_data_are_skipped() -> None:
    composition = compose({"bboxes": {"metadata": "metadata"}})

    output = composition(batch(bboxes=boxes(2)))

    assert "metadata" not in output


def test_already_empty_associated_fields_are_left_alone() -> None:
    composition = compose({"bboxes": {"metadata": "metadata"}})

    output = composition(batch(bboxes=boxes(2), metadata=np.array([])))

    assert output["metadata"].size == 0


@pytest.mark.parametrize(
    ("target_type", "value"),
    [
        ("instance_mask", np.ones((16, 16, 5), dtype=np.uint8)),
        ("metadata", np.ones(5)),
    ],
)
def test_unmatched_fields_warn_once_and_are_left_alone(
    target_type: str, value: np.ndarray, warnings_log: list[str]
) -> None:
    composition = compose(
        {"bboxes": {"label": target_type}}, {"label": target_type}
    )
    original = value.copy()

    output = composition(batch(bboxes=boxes(2), label=value))

    assert np.array_equal(output["label"], original)
    warned = "".join(warnings_log)
    assert f"shape {value.shape}" in warned
    assert "cannot be matched" in warned


def test_unmatched_keypoints_warn_and_are_left_alone(
    warnings_log: list[str],
) -> None:
    """Keypoint rows must number exactly ``bbox_count * n_keypoints``."""
    composition = compose(
        {"bboxes": {"label": "keypoints"}}, {"label": "keypoints"}
    )
    keypoints = np.array([[float(i), float(i), 2.0] for i in range(5)])

    output = composition(
        batch(bboxes=boxes(2), label=keypoints),
        keypoints_per_instance={"label": 1},
    )

    assert output["label"].shape[0] == 5
    assert "cannot be matched" in "".join(warnings_log)


def test_keypoints_without_a_known_count_are_left_alone(
    warnings_log: list[str],
) -> None:
    composition = compose(
        {"bboxes": {"label": "keypoints"}}, {"label": "keypoints"}
    )
    keypoints = np.array([[float(i), float(i), 2.0] for i in range(2)])

    composition(batch(bboxes=boxes(2), label=keypoints))

    assert "cannot be matched" in "".join(warnings_log)


def test_every_affected_sample_warns(warnings_log: list[str]) -> None:
    """A warning per sample, so the scale of the problem is visible.

    Suppressing repeats for the lifetime of the composition would report a
    dataset-wide misalignment exactly once per training run.
    """
    composition = compose(
        {"bboxes": {"label": "metadata"}}, {"label": "metadata"}
    )

    composition(batch(bboxes=boxes(2), label=np.ones(5)))
    composition(batch(bboxes=boxes(2), label=np.ones(5)))

    assert "".join(warnings_log).count("cannot be matched") == 2


def test_boxes_dropped_inside_a_transform_are_reported(
    warnings_log: list[str],
) -> None:
    """A transform that filters boxes itself must not pass unnoticed.

    Its output indices come back contiguous, so only the instance count
    reveals that the labels no longer describe the surviving boxes.
    """
    composition = compose(
        {"bboxes": {"label": "metadata"}},
        {"label": "metadata"},
        transform=DropsFirstBox(),
    )

    composition(batch(bboxes=boxes(3), label=np.arange(3.0)))

    assert "cannot be matched" in "".join(warnings_log)


def test_label_fields_are_not_overwritten_by_the_index_column() -> None:
    """Albumentations appends label columns to the right of the index one.

    Writing the instance index to the last column would replace every box's
    class label with its row number.
    """
    composition = BatchCompose(
        [KeepFirstSample()],
        bbox_params=A.BboxParams(
            format="albumentations", label_fields=["class_labels"]
        ),
        bbox_associations={"bboxes": {"metadata": "metadata"}},
    )
    indexed = np.array(
        [
            [0.1 * i, 0.1 * i, 0.1 * i + 0.2, 0.1 * i + 0.2, float(i)]
            for i in range(3)
        ]
    )

    output = composition(
        [
            {
                "image": IMAGE,
                "bboxes": indexed.copy(),
                "class_labels": np.array([7, 8, 9]),
                "metadata": np.arange(3.0),
            }
            for _ in range(2)
        ]
    )

    assert output["class_labels"].tolist() == [7, 8, 9]


def test_samples_without_a_label_get_one_empty_instance_per_box() -> None:
    """A batch transform concatenates only the samples that carry a field.

    Without a placeholder for the samples that do not, the merged labels
    come back shorter than the merged boxes and nothing can say which boxes
    they belong to.
    """
    composition = BatchCompose(
        [MixUp(p=1.0)],
        bbox_params=A.BboxParams(format="albumentations"),
        additional_targets={"label": "metadata"},
        bbox_associations={"bboxes": {"label": "metadata"}},
    )

    output = composition(
        [
            {"image": IMAGE, "bboxes": boxes(2), "label": np.arange(2.0)},
            {"image": IMAGE, "bboxes": boxes(3), "label": np.array([])},
        ]
    )

    assert len(output["label"]) == len(output["bboxes"]) == 5
    assert output["label"][:2].tolist() == [0.0, 1.0]


def test_missing_object_labels_are_padded_with_none() -> None:
    values = [
        np.array(["kept"], dtype=object),
        np.array([], dtype=object),
    ]

    BatchCompose._fill_empty_entries(values, "metadata", [1, 2], [], 0)

    assert values[1].tolist() == [None, None]


def test_indices_are_restamped_without_a_bbox_processor() -> None:
    """Nothing filters the boxes, but they still need distinct indices.

    Each sample is numbered from zero on the way in, so leaving the merged
    boxes as they are points several of them at the same label.
    """
    composition = BatchCompose(
        [MixUp(p=1.0)],
        bbox_associations={"bboxes": {"metadata": "metadata"}},
    )
    box = np.array([[0.1, 0.1, 0.3, 0.3, 0.0, 0.0]])

    output = composition(
        [
            {"image": IMAGE, "bboxes": box.copy(), "metadata": np.array([7])}
            for _ in range(2)
        ]
    )

    assert output["bboxes"][:, -1].tolist() == [0.0, 1.0]


def test_make_contiguous_leaves_non_arrays_alone() -> None:
    data = {"image": np.zeros((4, 4))[::2], "note": "kept as is"}

    out = BatchCompose._make_contiguous(data)  # type: ignore[arg-type]

    assert out["image"].flags["C_CONTIGUOUS"]
    assert out["note"] == "kept as is"
