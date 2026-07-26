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


def test_absent_metadata_does_not_create_phantom_boxless(
    dataset_name: str, tempdir: Path
) -> None:
    """Regression: a per-instance metadata task that is absent for a sample must
    not decode into box-less 'detections' whose every value reads as ``0.0``.

    Previously the empty task was filled with a class-length zero vector (as if it
    were classification), which decoded into phantom box-less instances. Those
    surfaced in the metadata card as ``key: 0.0`` while never appearing on hover.
    """

    def generator() -> DatasetIterator:
        img_a = str(create_image(0, tempdir))
        img_b = str(create_image(1, tempdir))
        # img_a registers a 2-class detection task carrying per-instance metadata.
        yield {
            "file": img_a,
            "task_name": "detection",
            "annotation": {
                "class": "car",
                "instance_id": 0,
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                "metadata": {"track_id": 7},
            },
        }
        yield {
            "file": img_a,
            "task_name": "detection",
            "annotation": {
                "class": "person",
                "instance_id": 1,
                "boundingbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
                "metadata": {"track_id": 8},
            },
        }
        # img_b has a single detection and no metadata for it.
        yield {
            "file": img_b,
            "task_name": "detection",
            "annotation": {
                "class": "car",
                "instance_id": 0,
                "boundingbox": {"x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2},
            },
        }

    dataset = create_dataset(
        dataset_name,
        generator(),
        BucketStorage.LOCAL,
        splits={"train": 1.0},
    )
    classes = dataset.get_classes()
    loader = LuxonisLoader(dataset, view="train")

    boxless_with_metadata = []
    for sample in loader:
        records = loader_output_to_records(sample.labels, classes=classes)
        for record in records.values():
            for det in record._annotations():
                if det.boundingbox is None and det.metadata:
                    boxless_with_metadata.append(det.metadata)
    assert boxless_with_metadata == []


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


def test_render_background_keeps_segmentation_background() -> None:
    """``render_background`` surfaces the background segmentation channel."""
    import numpy as np

    # A (C, H, W) semantic map: channel 0 is background, channel 1 is road.
    seg = np.zeros((2, 2, 2), dtype=np.uint8)
    seg[0, 0, :] = 1  # background occupies the top row
    seg[1, 1, :] = 1  # road occupies the bottom row
    labels = {"seg/segmentation": seg}
    classes = {"seg": {" background": 0, " road": 1}}

    # By default the background channel is dropped, mirroring detection.
    default = loader_output_to_records(labels, classes=classes)[
        "seg"
    ]._annotations()
    assert [d.class_name for d in default] == ["road"]

    # With render_background it becomes a drawable, stripped-name mask.
    shown = loader_output_to_records(
        labels, classes=classes, render_background=True
    )["seg"]._annotations()
    assert [d.class_name for d in shown] == ["background", "road"]
    assert all(d.segmentation is not None for d in shown)
