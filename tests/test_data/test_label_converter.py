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


def _detections(record: DatasetRecord, task_name: str) -> list[Detection]:
    return record.annotation[task_name]


def _record(labels: Labels, **kwargs) -> DatasetRecord:
    """Convert raw label arrays through the same path a loader sample takes."""
    return LoaderOutput({}, labels, {}).to_ldf(**kwargs)


def test_record_to_loader_output_round_trip() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    record = DatasetRecord(
        files={"image": image},
        annotation={
            "detection": [
                Detection(
                    class_name="car",
                    instance_id=7,
                    boundingbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},  # type: ignore[arg-type]
                    metadata={"score": 0.9},
                )
            ]
        },
        sample_metadata={"frame": 12},
    )

    output = record.to_loader_output()
    restored = output.to_ldf()
    round_trip = restored.to_loader_output(
        tasks={
            "detection": [
                "boundingbox",
                "classification",
                "metadata/score",
            ]
        },
        classes=output.classes,
        images=output.images,
    )

    assert output.metadata == round_trip.metadata == {"frame": 12}
    assert output.labels.keys() == round_trip.labels.keys()
    for task, labels in output.labels.items():
        np.testing.assert_array_equal(labels, round_trip.labels[task])


def test_empty_record_tasks_have_empty_loader_labels() -> None:
    record = DatasetRecord(
        files={"image": np.zeros((6, 10, 3), dtype=np.uint8)},
        annotation={"detection": [], "embeddings": []},
    )

    output = record.to_loader_output(
        tasks={
            "detection": [
                "boundingbox",
                "classification",
                "instance_segmentation",
                "keypoints",
                "metadata/id",
                "segmentation",
            ],
            "embeddings": ["array"],
        },
        classes={"detection": {"car": 0}, "embeddings": {"feature": 0}},
        n_keypoints={"detection": 2},
    )

    assert output.labels["detection/boundingbox"].shape == (0, 5)
    assert output.labels["detection/classification"].shape == (1,)
    assert output.labels["detection/instance_segmentation"].shape == (0, 6, 10)
    assert output.labels["detection/keypoints"].shape == (0, 6)
    assert output.labels["detection/metadata/id"].shape == (0,)
    assert output.labels["detection/segmentation"].shape == (1, 6, 10)
    assert output.labels["embeddings/array"].shape == (0,)
    assert output.to_ldf().annotation == {
        "detection": [],
        "embeddings": [],
    }


def test_empty_standalone_task_requires_its_task_schema() -> None:
    record = DatasetRecord(
        files={"image": np.zeros((4, 4, 3), dtype=np.uint8)},
        annotation={"detection": []},
    )

    with pytest.raises(ValueError, match="Cannot infer task types"):
        record.to_loader_output()

    with pytest.raises(ValueError, match="without its class mapping"):
        record.to_loader_output(tasks={"detection": ["classification"]})


def test_loader_preserves_true_negative_tasks(
    dataset_name: str, tempdir: Path
) -> None:
    array_path = tempdir / "embedding.npy"
    np.save(array_path, np.arange(4))
    records = [
        DatasetRecord(
            file=create_image(0, tempdir),  # type: ignore[call-arg]
            annotation={
                "detection": [
                    Detection(
                        class_name="car",
                        boundingbox={
                            "x": 0.1,
                            "y": 0.2,
                            "w": 0.3,
                            "h": 0.4,
                        },  # type: ignore[arg-type]
                    )
                ],
                "embeddings": [],
            },
            sample_metadata={"sample": "detection"},
        ),
        DatasetRecord(
            file=create_image(1, tempdir),  # type: ignore[call-arg]
            annotation={
                "detection": [],
                "embeddings": [
                    Detection(
                        class_name="feature",
                        array={"path": array_path},  # type: ignore[arg-type]
                    )
                ],
            },
            sample_metadata={"sample": "embedding"},
        ),
    ]
    dataset = create_dataset(
        dataset_name, iter(records), BucketStorage.LOCAL, splits={"train": 1.0}
    )

    samples = {
        sample.metadata["sample"]: sample
        for sample in LuxonisLoader(dataset, view="train")
    }
    expected_keys = {
        "detection/boundingbox",
        "detection/classification",
        "embeddings/array",
        "embeddings/classification",
    }
    assert all(
        set(sample.labels) == expected_keys for sample in samples.values()
    )

    detection_negative = samples["embedding"]
    assert detection_negative.labels["detection/boundingbox"].shape == (0, 5)
    assert not detection_negative.labels["detection/classification"].any()
    assert detection_negative.to_ldf().annotation["detection"] == []

    array_negative = samples["detection"]
    assert array_negative.labels["embeddings/array"].shape == (0,)
    assert not array_negative.labels["embeddings/classification"].any()
    assert array_negative.to_ldf().annotation["embeddings"] == []


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

    record = sample.to_ldf()
    assert "detection" in record.annotation

    detections = _detections(record, "detection")
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
        record = sample.to_ldf()
        for detections in record.annotation.values():
            for det in detections:
                if det.boundingbox is None and det.metadata:
                    boxless_with_metadata.append(det.metadata)
    assert boxless_with_metadata == []


def test_metadata_only_task_yields_detections() -> None:
    """A metadata-only task (e.g. OCR) still produces one detection per entry.

    The instance count is derived from the metadata array length, not just
    spatial annotations, so box-less metadata is not dropped.
    """
    labels = {"text/metadata/text": np.array(["HELLO", "WORLD"], dtype=object)}
    record = _record(labels, classes={"text": {}})

    detections = _detections(record, "text")
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
        _record(labels, classes={"det": {"car": 0}}), "det"
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
        _record(boxes, classes={"det": {" background ": 0, "  car ": 1}}),
        "det",
    )
    assert [d.class_name for d in dets] == [None, "car"]

    # A set background classification bit produces no chip.
    cls = {"cls/classification": np.array([1, 1, 0])}
    dets2 = _detections(
        _record(
            cls,
            classes={"cls": {"background": 0, " cat": 1, "dog": 2}},
        ),
        "cls",
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
    default = _detections(_record(labels, classes=classes), "seg")
    assert [d.class_name for d in default] == ["road"]

    # With render_background it becomes a drawable, stripped-name mask.
    shown = _detections(
        _record(labels, classes=classes, render_background=True), "seg"
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

    record = _record(labels, classes=classes)

    assert set(record.annotation) == {"/driver", "/passenger", "det/face"}
    assert {
        task: detections[0].class_name
        for task, detections in record.annotation.items()
    } == {
        "/driver": "driver",
        "/passenger": "passenger",
        "det/face": "face",
    }
    driver_box = _detections(record, "/driver")[0].boundingbox
    passenger_box = _detections(record, "/passenger")[0].boundingbox
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

    record = _record(labels, classes=classes)

    pose = _detections(record, "pose")
    assert any(detection.keypoints is not None for detection in pose)
    assert any(detection.class_name == "person" for detection in pose)

    objects = _detections(record, "objects")
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
        _record(labels, classes={"det": {"car": 0}}), "det"
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

    record = sample.to_ldf()
    assert [d.metadata for d in _detections(record, "detection")] == [
        {"weather": "sunny"},
        {},
        {"weather": "rainy"},
    ]


def test_to_ldf_without_classes_is_an_error() -> None:
    labels = {"det/boundingbox": np.array([[0, 0.1, 0.1, 0.2, 0.2]])}
    sample = LoaderOutput({}, labels, {})

    with pytest.raises(ValueError, match="without a class mapping"):
        sample.to_ldf()


def test_explicit_classes_override_the_attached_ones() -> None:
    labels = {"det/boundingbox": np.array([[0, 0.1, 0.1, 0.2, 0.2]])}
    sample = LoaderOutput({}, labels, {}, classes={"det": {"car": 0}})

    renamed = sample.to_ldf(classes={"det": {"truck": 0}})
    assert [d.class_name for d in _detections(renamed, "det")] == ["truck"]
    assert [d.class_name for d in _detections(sample.to_ldf(), "det")] == [
        "car"
    ]


def test_to_ldf_attaches_images(dataset_name: str, tempdir: Path) -> None:
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
    record = sample.to_ldf()

    assert set(record.annotation) == {"detection", "segmentation"}
    image = record.files["image"]
    assert isinstance(image, np.ndarray)
    assert np.array_equal(image, sample.image)


def test_array_annotations_round_trip(
    dataset_name: str, tempdir: Path
) -> None:
    """Conversion removes the class-slot padding added by the loader."""
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
    detections = _detections(sample.to_ldf(), "embeddings")

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
    record = next(iter(LuxonisLoader(dataset, view="train"))).to_ldf()

    with pytest.raises(NotImplementedError, match="in-memory image"):
        dataset.add(iter([record]))
    with pytest.raises(NotImplementedError, match="in-memory image"):
        list(record.to_parquet_rows())
