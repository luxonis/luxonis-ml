"""Round-trip: LuxonisLoader output back into canonical LDF records."""

from pathlib import Path

from luxonis_ml.data import BucketStorage, LuxonisLoader
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.loaders.label_converter import loader_output_to_records

from .utils import create_dataset, create_image


def test_loader_output_to_records_roundtrip(dataset_name: str, tempdir: Path):
    def generator() -> DatasetIterator:
        for i in range(3):
            path = str(create_image(i, tempdir))
            yield {
                "file": path,
                "task_name": "detection",
                "annotation": {
                    "class": "car",
                    "instance_id": 0,
                    "boundingbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                    "keypoints": {
                        "keypoints": [(0.15, 0.25, 2), (0.2, 0.3, 1)]
                    },
                },
            }
            yield {
                "file": path,
                "task_name": "detection",
                "annotation": {
                    "class": "person",
                    "instance_id": 1,
                    "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
                    "keypoints": {"keypoints": [(0.5, 0.5, 2), (0.6, 0.6, 0)]},
                },
            }

    dataset = create_dataset(dataset_name, generator(), BucketStorage.LOCAL)
    classes = dataset.get_classes()

    loader = LuxonisLoader(dataset, view="train")
    sample = next(iter(loader))

    records = loader_output_to_records(sample.labels, classes=classes)
    assert "detection" in records

    detections = records["detection"]._annotations()
    assert len(detections) == 2

    names = {d.class_name for d in detections}
    assert names == {"car", "person"}

    for det in detections:
        assert det.boundingbox is not None
        assert det.keypoints is not None
        # 2 keypoints per instance were added.
        assert len(det.keypoints.keypoints) == 2

    # Boxes are paired with their class by row index; coords survive the trip.
    by_name = {d.class_name: d for d in detections}
    car = by_name["car"].boundingbox
    assert car.x == 0.1  # type: ignore
    assert car.w == 0.3  # type: ignore


def test_metadata_only_task_yields_detections() -> None:
    """A metadata-only task (e.g. OCR) still produces one detection per entry.

    The instance count is derived from the metadata array length, not just
    spatial annotations, so box-less metadata is not dropped.
    """
    import numpy as np

    labels = {"text/metadata/text": np.array(["HELLO", "WORLD"], dtype=object)}
    records = loader_output_to_records(labels, classes={"text": {}})

    detections = records["text"]._annotations()
    assert len(detections) == 2
    assert [d.boundingbox for d in detections] == [None, None]
    assert [d.metadata["text"] for d in detections] == ["HELLO", "WORLD"]


def test_background_class_is_not_labeled() -> None:
    """The 'background' class never becomes a visible label."""
    import numpy as np

    # A background box is unlabeled (drawn, but no class chip); a real one keeps
    # its name. Names are stripped before rendering and the background check, so
    # stray whitespace does not defeat either.
    boxes = {
        "det/boundingbox": np.array(
            [[0, 0.1, 0.1, 0.2, 0.2], [1, 0.5, 0.5, 0.2, 0.2]]
        )
    }
    dets = loader_output_to_records(
        boxes, classes={"det": {" background ": 0, "  car ": 1}}
    )["det"]._annotations()
    assert [d.class_name for d in dets] == [None, "car"]

    # A set background classification bit produces no chip.
    cls = {"cls/classification": np.array([1, 1, 0])}
    dets2 = loader_output_to_records(
        cls, classes={"cls": {"background": 0, " cat": 1, "dog": 2}}
    )["cls"]._annotations()
    assert [d.class_name for d in dets2] == ["cat"]
