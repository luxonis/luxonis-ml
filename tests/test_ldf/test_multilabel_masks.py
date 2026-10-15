"""Semantic masks with the ``multilabel`` flag."""

import numpy as np
import pytest
from pydantic import ValidationError

from luxonis_ml.ldf import (
    DatasetRecord,
    DatasetSchema,
    InstanceSegmentationAnnotation,
    SegmentationAnnotation,
)

IMAGE = np.zeros((1, 4, 3), dtype=np.uint8)

# One row of four pixels. `road` and `sidewalk` share pixel 1, and `wet`
# covers pixels 1 to 3.
ROAD = np.array([[1, 1, 0, 0]])
SIDEWALK = np.array([[0, 1, 1, 0]])
WET = np.array([[0, 1, 1, 1]])


def scene(*, wet_is_flagged: bool) -> tuple[DatasetRecord, DatasetSchema]:
    """Return a record of the three masks and the schema of its task."""
    record = DatasetRecord.model_validate(
        {
            "media": IMAGE,
            "annotation": {
                "scene": [
                    {"class": "road", "segmentation": {"mask": ROAD}},
                    {"class": "sidewalk", "segmentation": {"mask": SIDEWALK}},
                    {
                        "class": "wet",
                        "segmentation": {
                            "mask": WET,
                            "multilabel": wet_is_flagged,
                        },
                    },
                ]
            },
        }
    )
    schema = DatasetSchema(
        tasks={"scene": ["segmentation", "classification"]},
        classes={"scene": {"road": 0, "sidewalk": 1, "wet": 2}},
        multilabel={"scene"} if wet_is_flagged else set(),
    )
    return record, schema


def mask(array: np.ndarray, *, multilabel: bool = False) -> dict:
    return {"mask": array, "multilabel": multilabel}


def test_a_flagged_mask_takes_no_pixels_from_a_later_mask():
    wet = SegmentationAnnotation.model_validate(mask(WET, multilabel=True))
    road = SegmentationAnnotation.model_validate(mask(ROAD))

    combined = SegmentationAnnotation.combine_to_numpy([wet, road], [2, 0], 3)

    assert combined[0].tolist() == ROAD.tolist()


def test_only_a_flagged_mask_stores_the_flag():
    """The rows of a dataset without multi-label masks do not change."""
    plain = SegmentationAnnotation.model_validate(mask(ROAD))
    flagged = SegmentationAnnotation.model_validate(
        mask(ROAD, multilabel=True)
    )

    assert "multilabel" not in plain.to_parquet_json()
    assert '"multilabel":true' in flagged.to_parquet_json()


def test_an_instance_mask_refuses_the_flag():
    with pytest.raises(ValidationError, match="semantic masks only"):
        InstanceSegmentationAnnotation.model_validate(
            mask(ROAD, multilabel=True)
        )


def test_a_rebuilt_multilabel_record_gives_the_same_labels():
    record, schema = scene(wet_is_flagged=True)
    labels = record.to_loader_output(schema).labels["scene/segmentation"]

    rebuilt = record.to_loader_output(schema).to_ldf()
    rebuilt_labels = rebuilt.to_loader_output(schema).labels

    assert labels.tolist() == [[[1, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 1, 1, 1]]]
    assert all(
        detection.segmentation is not None
        and detection.segmentation.multilabel
        for detection in rebuilt.annotation["scene"]
    )
    assert np.array_equal(rebuilt_labels["scene/segmentation"], labels)


def test_a_rebuilt_single_label_record_has_no_flag():
    """A rebuilt record must not turn a single-label task multi-label."""
    record, schema = scene(wet_is_flagged=False)

    rebuilt = record.to_loader_output(schema).to_ldf()

    assert not any(
        detection.segmentation is not None
        and detection.segmentation.multilabel
        for detection in rebuilt.annotation["scene"]
    )
