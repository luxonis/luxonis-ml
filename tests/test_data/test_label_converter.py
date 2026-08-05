"""Round-trip: LuxonisLoader output back into canonical LDF records."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.data import BucketStorage, LuxonisLoader
from luxonis_ml.data.datasets import Category
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.ldf import DatasetRecord, Detection
from luxonis_ml.typing import Labels, LoaderOutput

from .utils import create_dataset, create_image


def _detections(record: DatasetRecord) -> list[Detection]:
    assert record.annotation is not None
    return record.annotation


def _records(labels: Labels, **kwargs) -> dict[str, DatasetRecord]:
    """Convert raw label arrays through the same path a loader sample takes."""
    return LoaderOutput({}, labels, {}).to_ldf(**kwargs)


def test_to_ldf_roundtrip(dataset_name: str, tempdir: Path):
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

    loader = LuxonisLoader(dataset, view="train")
    sample = next(iter(loader))

    # The loader attaches the dataset's classes, so no argument is needed.
    assert sample.classes == dataset.get_classes()

    records = sample.to_ldf()
    assert "detection" in records

    detections = _detections(records["detection"])
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
    loader = LuxonisLoader(dataset, view="train")

    boxless_with_metadata = []
    for sample in loader:
        records = sample.to_ldf()
        for record in records.values():
            for det in _detections(record):
                if det.boundingbox is None and det.metadata:
                    boxless_with_metadata.append(det.metadata)
    assert boxless_with_metadata == []


def test_metadata_only_task_yields_detections() -> None:
    """A metadata-only task (e.g. OCR) still produces one detection per entry.

    The instance count is derived from the metadata array length, not just
    spatial annotations, so box-less metadata is not dropped.
    """
    labels = {"text/metadata/text": np.array(["HELLO", "WORLD"], dtype=object)}
    records = _records(labels, classes={"text": {}})

    detections = _detections(records["text"])
    assert len(detections) == 2
    assert [d.boundingbox for d in detections] == [None, None]
    assert [d.metadata["text"] for d in detections] == ["HELLO", "WORLD"]


def test_conversion_preserves_prediction_scores() -> None:
    labels = {
        "det/boundingbox": np.array(
            [[0, 0.1, 0.1, 0.2, 0.2], [0, 0.5, 0.5, 0.2, 0.2]]
        ),
        "det/metadata/score": np.array([0.15, 0.9]),
    }

    detections = _detections(
        _records(labels, classes={"det": {"car": 0}})["det"]
    )

    assert [detection.metadata["score"] for detection in detections] == [
        0.15,
        0.9,
    ]


def test_background_class_is_not_labeled() -> None:
    """The 'background' class never becomes a visible label."""
    # A background box is unlabeled (drawn, but no class chip); a real one keeps
    # its name. Names are stripped before rendering and the background check, so
    # stray whitespace does not defeat either.
    boxes = {
        "det/boundingbox": np.array(
            [[0, 0.1, 0.1, 0.2, 0.2], [1, 0.5, 0.5, 0.2, 0.2]]
        )
    }
    dets = _detections(
        _records(boxes, classes={"det": {" background ": 0, "  car ": 1}})[
            "det"
        ]
    )
    assert [d.class_name for d in dets] == [None, "car"]

    # A set background classification bit produces no chip.
    cls = {"cls/classification": np.array([1, 1, 0])}
    dets2 = _detections(
        _records(cls, classes={"cls": {"background": 0, " cat": 1, "dog": 2}})[
            "cls"
        ]
    )
    assert [d.class_name for d in dets2] == ["cat"]


def test_render_background_keeps_segmentation_background() -> None:
    """``render_background`` surfaces the background segmentation channel."""
    # A (C, H, W) semantic map: channel 0 is background, channel 1 is road.
    seg = np.zeros((2, 2, 2), dtype=np.uint8)
    seg[0, 0, :] = 1  # background occupies the top row
    seg[1, 1, :] = 1  # road occupies the bottom row
    labels = {"seg/segmentation": seg}
    classes = {"seg": {" background": 0, " road": 1}}

    # By default the background channel is dropped, mirroring detection.
    default = _detections(_records(labels, classes=classes)["seg"])
    assert [d.class_name for d in default] == ["road"]

    # With render_background it becomes a drawable, stripped-name mask.
    shown = _detections(
        _records(labels, classes=classes, render_background=True)["seg"]
    )
    assert [d.class_name for d in shown] == ["background", "road"]
    assert all(d.segmentation is not None for d in shown)


def test_nested_loader_task_paths_remain_distinct() -> None:
    """Nested tasks keep their full paths instead of overwriting one another."""
    labels = {
        "/driver/boundingbox": np.array([[0, 0.1, 0.1, 0.2, 0.2]]),
        "/passenger/boundingbox": np.array([[0, 0.3, 0.3, 0.2, 0.2]]),
        "det/face/boundingbox": np.array([[0, 0.5, 0.5, 0.2, 0.2]]),
    }
    classes = {
        "/driver": {"driver": 0},
        "/passenger": {"passenger": 0},
        "det/face": {"face": 0},
    }

    records = _records(labels, classes=classes)

    assert set(records) == {"/driver", "/passenger", "det/face"}
    assert {
        task: _detections(record)[0].class_name
        for task, record in records.items()
    } == {
        "/driver": "driver",
        "/passenger": "passenger",
        "det/face": "face",
    }
    driver_box = _detections(records["/driver"])[0].boundingbox
    passenger_box = _detections(records["/passenger"])[0].boundingbox
    assert driver_box is not None
    assert passenger_box is not None
    assert driver_box.x == 0.1
    assert passenger_box.x == 0.3


def test_boxless_spatial_tasks_keep_classification_tags() -> None:
    """Keypoints and instance masks retain classes that they cannot encode."""
    labels = {
        "pose/keypoints": np.array([[0.25, 0.5, 2]]),
        "pose/classification": np.array([0, 1]),
        "objects/instance_segmentation": np.ones((1, 4, 4), np.uint8),
        "objects/classification": np.array([1]),
    }
    classes = {
        "pose": {"background": 0, "person": 1},
        "objects": {"vehicle": 0},
    }

    records = _records(labels, classes=classes)

    pose = _detections(records["pose"])
    assert any(detection.keypoints is not None for detection in pose)
    assert any(detection.class_name == "person" for detection in pose)

    objects = _detections(records["objects"])
    assert any(
        detection.instance_segmentation is not None for detection in objects
    )
    assert any(detection.class_name == "vehicle" for detection in objects)


def test_padded_metadata_rows_are_not_labels() -> None:
    labels = {
        "det/boundingbox": np.array(
            [[0, 0.1, 0.1, 0.2, 0.2], [0, 0.5, 0.5, 0.2, 0.2]]
        ),
        "det/metadata/tag": np.array(["kept", None], dtype=object),
    }
    detections = _detections(
        _records(labels, classes={"det": {"car": 0}})["det"]
    )
    assert [d.metadata for d in detections] == [{"tag": "kept"}, {}]


@pytest.mark.parametrize("keep_strings", [False, True])
def test_partial_categorical_metadata_round_trips(
    dataset_name: str, tempdir: Path, keep_strings: bool
) -> None:
    def generator() -> DatasetIterator:
        path = str(create_image(0, tempdir))
        for instance_id, x, weather in (
            (0, 0.1, Category("sunny")),
            (1, 0.4, None),
            (2, 0.7, Category("rainy")),
        ):
            annotation: dict = {
                "class": "car",
                "instance_id": instance_id,
                "boundingbox": {"x": x, "y": 0.1, "w": 0.2, "h": 0.2},
            }
            if weather is not None:
                annotation["metadata"] = {"weather": weather}
            yield {
                "file": path,
                "task_name": "detection",
                "annotation": annotation,
            }

    dataset = create_dataset(
        dataset_name, generator(), BucketStorage.LOCAL, splits={"train": 1.0}
    )
    encodings = dataset.get_categorical_encodings()
    assert encodings == {
        "detection/metadata/weather": {"sunny": 0, "rainy": 1}
    }

    loader = LuxonisLoader(
        dataset, view="train", keep_categorical_as_strings=keep_strings
    )
    sample = next(iter(loader))
    assert sample.categorical_encodings == encodings

    records = sample.to_ldf()
    assert [d.metadata for d in _detections(records["detection"])] == [
        {"weather": "sunny"},
        {},
        {"weather": "rainy"},
    ]


def test_to_ldf_without_classes_is_an_error() -> None:
    """A `LoaderOutput` built without classes cannot name its detections.

    Class ids are meaningless on their own, so a loader that does not attach a
    mapping has to be told about one instead of silently producing unnamed
    detections.
    """
    labels = {"det/boundingbox": np.array([[0, 0.1, 0.1, 0.2, 0.2]])}
    sample = LoaderOutput({}, labels, {})

    with pytest.raises(ValueError, match="without a class mapping"):
        sample.to_ldf()

    detections = _detections(sample.to_ldf(classes={"det": {"car": 0}})["det"])
    assert [d.class_name for d in detections] == ["car"]


def test_explicit_classes_override_the_attached_ones() -> None:
    """Passing ``classes`` wins over whatever the loader attached."""
    labels = {"det/boundingbox": np.array([[0, 0.1, 0.1, 0.2, 0.2]])}
    sample = LoaderOutput({}, labels, {}, classes={"det": {"car": 0}})

    renamed = sample.to_ldf(classes={"det": {"truck": 0}})
    assert [d.class_name for d in _detections(renamed["det"])] == ["truck"]
    assert [d.class_name for d in _detections(sample.to_ldf()["det"])] == [
        "car"
    ]


def test_images_are_attached_to_every_record(
    dataset_name: str, tempdir: Path
) -> None:
    """The sample's images end up in the records, not just its labels."""

    def generator() -> DatasetIterator:
        path = str(create_image(0, tempdir))
        yield {
            "file": path,
            "task_name": "detection",
            "annotation": {
                "class": "car",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            },
        }
        yield {
            "file": path,
            "task_name": "segmentation",
            "annotation": {
                "class": "road",
                "segmentation": {"mask": np.ones((512, 512), np.uint8)},
            },
        }

    dataset = create_dataset(
        dataset_name, generator(), BucketStorage.LOCAL, splits={"train": 1.0}
    )
    sample = next(iter(LuxonisLoader(dataset, view="train")))
    records = sample.to_ldf()

    assert set(records) == {"detection", "segmentation"}
    for record in records.values():
        assert np.array_equal(record.files["image"], sample.image)


def test_array_annotations_round_trip(
    dataset_name: str, tempdir: Path
) -> None:
    """An array label decodes back into the array that was stored.

    The loader pads each instance's array into the slot of its class, so the
    conversion has to undo that to recover the original.
    """
    arrays = {
        "car": np.arange(12, dtype=np.float64).reshape(3, 4),
        "person": np.ones((3, 4)) * 7,
    }

    def generator() -> DatasetIterator:
        path = str(create_image(0, tempdir))
        for i, (class_name, array) in enumerate(arrays.items()):
            array_path = tempdir / f"{class_name}.npy"
            np.save(array_path, array)
            yield {
                "file": path,
                "task_name": "embeddings",
                "annotation": {
                    "class": class_name,
                    "instance_id": i,
                    "array": {"path": str(array_path)},
                },
            }

    dataset = create_dataset(
        dataset_name, generator(), BucketStorage.LOCAL, splits={"train": 1.0}
    )
    sample = next(iter(LuxonisLoader(dataset, view="train")))
    detections = _detections(sample.to_ldf()["embeddings"])

    recovered = {
        detection.class_name: detection.array.to_numpy()
        for detection in detections
        if detection.array is not None
    }
    assert set(recovered) == set(arrays)
    for class_name, array in arrays.items():
        np.testing.assert_array_equal(recovered[class_name], array)


def test_reconstructed_records_are_rejected_by_add(
    dataset_name: str, tempdir: Path
) -> None:
    """Records holding in-memory data cannot be added back to a dataset.

    Storing them would mean writing new media and ``.npy`` files, which is not
    supported yet -- the point is that it fails loudly rather than writing a
    record that points at ``str(array)``.
    """

    def generator() -> DatasetIterator:
        yield {
            "file": str(create_image(0, tempdir)),
            "task_name": "detection",
            "annotation": {
                "class": "car",
                "boundingbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            },
        }

    dataset = create_dataset(
        dataset_name, generator(), BucketStorage.LOCAL, splits={"train": 1.0}
    )
    record = next(iter(LuxonisLoader(dataset, view="train"))).to_ldf()[
        "detection"
    ]

    with pytest.raises(NotImplementedError, match="in-memory image"):
        dataset.add([record])
    with pytest.raises(NotImplementedError, match="in-memory image"):
        list(record.to_parquet_rows())
