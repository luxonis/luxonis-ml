from pathlib import Path

import cv2
import numpy as np
import pytest

from luxonis_ml.data.parsers.solo_parser import _parse_capture

_SOLO = "type.unity.com/unity.solo."


def _capture(sequence_path: Path) -> dict:
    """Write a capture with one box, one mask and annotations to skip."""
    cv2.imwrite(
        str(sequence_path / "step0.camera.png"),
        np.zeros((4, 6, 3), dtype=np.uint8),
    )
    mask = np.zeros((4, 6, 3), dtype=np.uint8)
    mask[1:3, 2:5] = (0, 0, 255)
    cv2.imwrite(str(sequence_path / "step0.instance.png"), mask)

    box = {
        "@type": f"{_SOLO}BoundingBox2DAnnotation",
        "values": [
            {
                "labelName": "car",
                "instanceId": 1,
                "origin": [0, 0],
                "dimension": [3, 2],
            }
        ],
    }
    return {
        "filename": "step0.camera.png",
        "dimension": [6, 4],
        "annotations": [
            {"@type": f"{_SOLO}DepthAnnotation"},
            box,
            {**box, "values": [{**box["values"][0], "labelName": "truck"}]},
            {
                "@type": f"{_SOLO}InstanceSegmentationAnnotation",
                "filename": "step0.instance.png",
                "instances": [{"instanceId": 1, "color": [255, 0, 0, 255]}],
            },
        ],
    }


def test_capture_joins_box_and_instance_mask(tempdir: Path):
    (record,) = _parse_capture(tempdir, _capture(tempdir), set())

    assert isinstance(record, dict)
    annotation = record["annotation"]
    assert annotation["class"] == "car"
    assert annotation["boundingbox"] == {"x": 0, "y": 0, "w": 0.5, "h": 0.5}
    assert annotation["instance_segmentation"]["mask"].sum() == 6


def test_capture_reads_each_kind_once_per_step(tempdir: Path):
    processed = {
        "BoundingBox2DAnnotation",
        "InstanceSegmentationAnnotation",
    }

    assert list(_parse_capture(tempdir, _capture(tempdir), processed)) == []


def test_capture_needs_its_image(tempdir: Path):
    capture = _capture(tempdir)
    capture["filename"] = "missing.png"

    with pytest.raises(FileNotFoundError, match=r"missing\.png"):
        list(_parse_capture(tempdir, capture, set()))


@pytest.mark.parametrize(
    ("contents", "error"),
    [(None, FileNotFoundError), (b"not an image", ValueError)],
)
def test_capture_needs_a_readable_mask(
    tempdir: Path, contents: bytes | None, error: type[Exception]
):
    capture = _capture(tempdir)
    mask_path = tempdir / "step0.instance.png"
    mask_path.unlink()
    if contents is not None:
        mask_path.write_bytes(contents)

    with pytest.raises(error, match=r"step0\.instance\.png"):
        list(_parse_capture(tempdir, capture, set()))
