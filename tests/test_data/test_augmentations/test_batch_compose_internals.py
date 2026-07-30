"""Behaviour of `BatchCompose` on inputs the engine does not normally send.

`BatchCompose` is exported, so it can be driven directly with bbox
associations that do not line up with the data. These tests pin down what it
does then: warn once and leave the data alone, rather than raise or silently
regroup it.
"""

from collections.abc import Iterator

import albumentations as A
import numpy as np
import pytest
from loguru import logger

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.augmentations import BatchCompose, BatchTransform
from luxonis_ml.typing import Labels

IMAGE = np.zeros((16, 16, 3), dtype=np.uint8)


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    """Collect loguru warnings, which pytest's caplog does not see."""
    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handler)


class KeepFirstSample(BatchTransform):
    """Keeps the first sample's targets exactly as they arrived.

    Every target is passed through untouched so a test can see precisely what
    `BatchCompose` itself did to the data.
    """

    def __init__(self):
        super().__init__(batch_size=2, p=1.0)

    def apply(self, image_batch: list[np.ndarray], **_) -> np.ndarray:
        return image_batch[0]

    def apply_to_mask(self, masks_batch: list[np.ndarray], **_) -> np.ndarray:
        return masks_batch[0]

    def apply_to_instance_mask(
        self, masks_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return masks_batch[0]

    def apply_to_bboxes(
        self, bboxes_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return bboxes_batch[0]

    def apply_to_keypoints(
        self, keypoints_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return keypoints_batch[0]

    def apply_to_metadata(
        self, metadata_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return metadata_batch[0]

    def apply_to_array(self, array_batch: list[np.ndarray], **_) -> np.ndarray:
        return array_batch[0]


class PushFirstBoxOutOfFrame(KeepFirstSample):
    """Moves the first box outside the image so filtering drops it.

    Indices are stamped before filtering, so the surviving boxes then carry
    non-contiguous indices - the state compaction exists to handle.
    """

    def apply_to_bboxes(
        self, bboxes_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        boxes = bboxes_batch[0].copy()
        boxes[0, :4] = [1.2, 1.2, 1.5, 1.5]
        return boxes


class ReturnsFlatBoxes(KeepFirstSample):
    """Hands back a one-dimensional bbox array, which is never valid."""

    def apply_to_bboxes(self, _batch: list[np.ndarray], **_) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0])


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
    """A declared association need not appear in every sample."""
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
    ("target_type", "value", "expected_message"),
    [
        ("instance_mask", np.ones((16, 16, 5), dtype=np.uint8), "5 instances"),
        ("metadata", np.ones(5), "5 values"),
    ],
)
def test_unmatched_fields_warn_once_and_are_left_alone(
    target_type: str,
    value: np.ndarray,
    expected_message: str,
    warnings_log: list[str],
) -> None:
    """Counts that cannot be matched to boxes must not be regrouped."""
    composition = compose(
        {"bboxes": {"label": target_type}}, {"label": target_type}
    )
    original = value.copy()

    output = composition(batch(bboxes=boxes(2), label=value))

    assert np.array_equal(output["label"], original), (
        "an unmatched field must be passed through untouched"
    )
    warned = "".join(warnings_log)
    assert expected_message in warned
    assert "cannot be matched up" in warned


def test_unmatched_keypoints_warn_and_are_left_alone(
    warnings_log: list[str],
) -> None:
    """Keypoint rows must be exactly ``bbox_count * n_keypoints``."""
    composition = compose(
        {"bboxes": {"label": "keypoints"}}, {"label": "keypoints"}
    )
    keypoints = np.array([[float(i), float(i), 2.0] for i in range(5)])

    output = composition(
        batch(bboxes=boxes(2), label=keypoints),
        keypoints_per_instance={"label": 1},
    )

    assert output["label"].shape[0] == 5
    assert "5 keypoint rows" in "".join(warnings_log)


def test_keypoints_without_a_known_count_are_left_alone(
    warnings_log: list[str],
) -> None:
    """Without the per-instance count there is no safe way to regroup."""
    composition = compose(
        {"bboxes": {"label": "keypoints"}}, {"label": "keypoints"}
    )
    keypoints = np.array([[float(i), float(i), 2.0] for i in range(2)])

    composition(batch(bboxes=boxes(2), label=keypoints))

    assert "cannot be matched up" in "".join(warnings_log)


def test_the_same_field_only_warns_once(warnings_log: list[str]) -> None:
    composition = compose(
        {"bboxes": {"label": "metadata"}}, {"label": "metadata"}
    )

    composition(batch(bboxes=boxes(2), label=np.ones(5)))
    composition(batch(bboxes=boxes(2), label=np.ones(5)))

    assert "".join(warnings_log).count("cannot be matched up") == 1


def test_make_contiguous_leaves_non_arrays_alone() -> None:
    """Not every value in the data dict is a numpy array."""
    data = {"image": np.zeros((4, 4))[::2], "note": "kept as is"}

    out = BatchCompose._make_contiguous(data)  # type: ignore[arg-type]

    assert out["image"].flags["C_CONTIGUOUS"]
    assert out["note"] == "kept as is"


def test_labels_emptied_by_a_spatial_transform_are_reported_empty() -> None:
    """A crop can remove every box after the batch stage has finished."""
    engine = AlbumentationsEngine(
        16,
        16,
        {"task/boundingbox": "boundingbox", "task/metadata/id": "metadata"},
        {"task/boundingbox": 1, "task/metadata/id": 1},
        ["image"],
        [
            {
                "name": "Crop",
                "params": {
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 4,
                    "y_max": 4,
                    "p": 1.0,
                },
            }
        ],
    )
    labels: Labels = {
        "task/boundingbox": np.array([[0.0, 0.7, 0.7, 0.2, 0.2]]),
        "task/metadata/id": np.array([9.0]),
    }

    _, out_labels = engine.apply(
        [({"image": np.zeros((16, 16, 3), dtype=np.uint8)}, labels)]
    )

    assert out_labels.get("task/boundingbox", np.zeros((0, 5))).shape[0] == 0
    assert out_labels.get("task/metadata/id", np.zeros(0)).shape[0] == 0
