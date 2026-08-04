"""Regression tests for the annotation validators."""

import json
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pydantic
import pytest
from PIL import Image

from luxonis_ml.ldf import (
    ArrayAnnotation,
    BBoxAnnotation,
    DatasetRecord,
    Detection,
    KeypointAnnotation,
    SegmentationAnnotation,
    load_annotation,
)


def test_bbox_missing_field_is_a_validation_error():
    """Indexing the raw input used to raise a bare ``KeyError: 'h'``."""
    with pytest.raises(pydantic.ValidationError) as info:
        BBoxAnnotation(x=0.1, y=0.1, w=0.1)  # type: ignore[call-arg]
    assert [(e["loc"], e["type"]) for e in info.value.errors()] == [
        (("h",), "missing")
    ]

    with pytest.raises(pydantic.ValidationError) as info:
        Detection(class_name="car", boundingbox={"x": 0.1, "y": 0.1, "w": 0.1})  # type: ignore[arg-type]
    assert [(e["loc"], e["type"]) for e in info.value.errors()] == [
        (("boundingbox", "h"), "missing")
    ]


def test_bbox_non_numeric_field_is_a_validation_error():
    """Comparing the raw input used to raise a bare ``TypeError``."""
    with pytest.raises(pydantic.ValidationError):
        BBoxAnnotation(x="a", y=0.1, w=0.1, h=0.1)  # type: ignore[arg-type]


def test_bbox_non_mapping_input_is_a_validation_error():
    with pytest.raises(pydantic.ValidationError):
        BBoxAnnotation.model_validate([0.1, 0.1, 0.1, 0.1])


@pytest.mark.parametrize(
    "annotation",
    [
        {"segmentation": [1, 2]},
        {"keypoints": 5},
        {"keypoints": [5]},
        {"instance_segmentation": "mask"},
    ],
)
def test_malformed_payload_is_a_validation_error(annotation: dict[str, Any]):
    """The validators used to index the raw input before checking it."""
    with pytest.raises(pydantic.ValidationError):
        Detection(**annotation)


def test_non_mapping_record_is_a_validation_error():
    with pytest.raises(pydantic.ValidationError):
        DatasetRecord.model_validate(5)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # Out-of-range coordinates are clipped to [0, 1].
        ({"x": -0.5, "y": 0.1, "w": 0.2, "h": 0.2}, (0, 0.1, 0.2, 0.2)),
        # `x + w > 1` clips the width so the sum is 1.
        ({"x": 0.5, "y": 0.1, "w": 0.9, "h": 0.2}, (0.5, 0.1, 0.5, 0.2)),
    ],
)
def test_bbox_clipping_does_not_mutate_the_input(
    values: dict[str, float], expected: tuple[float, float, float, float]
):
    original = dict(values)
    box = BBoxAnnotation.model_validate(values)

    assert values == original
    assert (box.x, box.y, box.w, box.h) == expected


# All of these are coerced to `float` by pydantic, so all of them have to be
# clipped. Only `np.float32` is a `numbers.Real`.
@pytest.mark.parametrize(
    "coordinates",
    [
        [np.float32(-0.5), np.float32(0.1), np.float32(0.2), np.float32(0.2)],
        [np.array(-0.5), np.array(0.1), np.array(0.2), np.array(0.2)],
        [Decimal("-0.5"), Decimal("0.1"), Decimal("0.2"), Decimal("0.2")],
        ["-0.5", "0.1", "0.2", "0.2"],
    ],
    ids=["numpy-scalar", "numpy-0d-array", "decimal", "string"],
)
def test_bbox_clipping_applies_to_every_coercible_type(coordinates: list[Any]):
    box = BBoxAnnotation.model_validate(
        dict(zip("xywh", coordinates, strict=True))
    )

    assert box.x == 0


def test_bbox_sum_clipping_applies_to_decimals():
    """`x + w > 1` was left unclipped for anything but a `float`."""
    coordinates = map(Decimal, ["0.9", "0.1", "0.5", "0.2"])
    box = BBoxAnnotation.model_validate(
        dict(zip("xywh", coordinates, strict=True))
    )

    assert box.x + box.w == pytest.approx(1)


def test_keypoint_clipping_does_not_mutate_the_input():
    keypoints = [(0.5, 0.5, 2), (1.5, 0.2, 1)]
    values = {"keypoints": list(keypoints)}

    annotation = KeypointAnnotation.model_validate(values)

    assert values == {"keypoints": keypoints}
    assert annotation.keypoints == [(0.5, 0.5, 2), (1.0, 0.2, 1)]


@pytest.mark.parametrize(
    ("task_type", "data"),
    [
        ("boundingbox", {"x": 0.5, "y": 0.5, "w": 0.9, "h": 0.2}),
        ("keypoints", {"keypoints": [(1.5, 0.2, 1)]}),
        # The RLE validator rewrites `counts` into the encoded bytes.
        ("segmentation", {"height": 4, "width": 4, "counts": "11213ON0"}),
        ("segmentation", {"height": 4, "width": 4, "counts": [1, 1, 2, 12]}),
    ],
)
def test_load_annotation_does_not_mutate_the_input(
    task_type: str, data: dict[str, Any]
):
    """The mapping is validated directly, so nothing must leak back."""
    original = json.dumps(data)

    load_annotation(task_type, data)  # type: ignore[arg-type]

    assert json.dumps(data) == original


@pytest.mark.parametrize(
    ("task_type", "data"),
    [
        ("boundingbox", {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}),
        ("keypoints", {"keypoints": [(0.5, 0.2, 1)]}),
        ("segmentation", {"height": 4, "width": 4, "counts": "11213ON0"}),
    ],
)
def test_load_annotation_accepts_a_read_only_mapping(
    task_type: str, data: dict[str, Any]
):
    """`load_annotation` is typed to take any `Mapping`."""
    load_annotation(task_type, MappingProxyType(data))  # type: ignore[arg-type]


def test_mask_size_wins_over_supplied_height_and_width():
    """A stale `height`/`width` used to override the encoded mask size."""
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 1

    annotation = SegmentationAnnotation(
        mask=mask,  # type: ignore[call-arg]
        height=999,
        width=999,
    )

    assert (annotation.height, annotation.width) == (4, 6)
    np.testing.assert_array_equal(annotation.to_numpy(), mask)


def test_record_rejects_both_file_and_files(tempdir: Path):
    """`files` used to be silently replaced by the single `file`."""
    with pytest.raises(pydantic.ValidationError, match="not both"):
        DatasetRecord(
            file=tempdir / "image.png",  # type: ignore[call-arg]
            files={
                "left": tempdir / "left.png",
                "right": tempdir / "right.png",
            },
        )


@pytest.mark.parametrize(
    ("record", "location"),
    [
        ({"files": [1, 2]}, ("files",)),
        ({"files": "image.png"}, ("files",)),
        ({"files": {"image": 5}}, ("files", "image")),
        ({"file": 5}, ("files", "image")),
    ],
)
def test_malformed_files_are_a_validation_error(
    record: dict[str, Any], location: tuple[str, ...]
):
    """Normalizing the paths used to raise a bare `AttributeError`."""
    with pytest.raises(pydantic.ValidationError) as info:
        DatasetRecord(**record)
    assert info.value.errors()[0]["loc"] == location


def test_record_rejects_conflicting_task_and_task_name(tempdir: Path):
    """The deprecated `task` used to silently win over `task_name`."""
    (tempdir / "image.png").touch()
    with pytest.raises(pydantic.ValidationError, match="Conflicting values"):
        DatasetRecord(
            file=tempdir / "image.png",  # type: ignore[call-arg]
            task="detection",  # type: ignore[call-arg]
            task_name="segmentation",
        )


def test_record_accepts_matching_task_and_task_name(tempdir: Path):
    (tempdir / "image.png").touch()
    record = DatasetRecord(
        file=tempdir / "image.png",  # type: ignore[call-arg]
        task="detection",  # type: ignore[call-arg]
        task_name="detection",
    )

    assert record.task_name == "detection"


def test_scale_to_boxes_error_names_the_actual_field():
    with pytest.raises(pydantic.ValidationError, match=r"`scale_to_boxes`"):
        Detection(
            class_name="car",
            scale_to_boxes=True,
            keypoints={"keypoints": [(0.5, 0.5, 2)]},  # type: ignore[arg-type]
        )


def test_png_mask_is_read_without_opencv(tempdir: Path):
    """`opencv` is not part of the `ldf` extra."""
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 1
    Image.fromarray(mask * 255).save(tempdir / "mask.png")

    annotation = SegmentationAnnotation(mask=tempdir / "mask.png")  # type: ignore[call-arg]

    np.testing.assert_array_equal(annotation.to_numpy(), mask)


def test_array_path_falls_back_when_mmap_is_unavailable(
    tempdir: Path, monkeypatch: pytest.MonkeyPatch
):
    """Memory mapping is not supported on every filesystem."""
    np.save(tempdir / "array.npy", np.zeros(4))
    load = np.load

    def no_mmap(*args: Any, **kwargs: Any) -> Any:
        if kwargs.pop("mmap_mode", None) is not None:
            raise OSError("mmap is not supported here")
        return load(*args, **kwargs)

    monkeypatch.setattr(np, "load", no_mmap)

    assert ArrayAnnotation(path=tempdir / "array.npy").path.name == "array.npy"


def test_sub_detections_inherit_the_record_metadata(tempdir: Path):
    """The serialized metadata is now threaded through the recursion."""
    (tempdir / "image.png").touch()
    record = DatasetRecord(
        file=tempdir / "image.png",  # type: ignore[call-arg]
        sample_metadata={"camera": "left"},
        annotation=Detection(
            class_name="person",
            sub_detections={"head": Detection(class_name="head")},
        ),
    )

    rows = list(record.to_parquet_rows())

    assert [row["task_type"] for row in rows] == [
        "classification",
        "classification",
    ]
    assert {row["sample_metadata"] for row in rows} == {'{"camera": "left"}'}
