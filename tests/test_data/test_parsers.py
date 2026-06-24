import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from logot import Capturer, Logot, logged
from logot.loguru import LoguruCapturer

from luxonis_ml.data import (
    BaseDataset,
    LuxonisLoader,
    LuxonisParser,
)
from luxonis_ml.data.parsers.base_parser import BaseParser, ParserOutput
from luxonis_ml.data.utils import get_task_type
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import environ

from .utils import create_image


@pytest.fixture
def logot_capturer() -> Callable[[], Capturer]:
    return LoguruCapturer


@pytest.mark.parametrize(
    ("url", "expected_task_types"),
    [
        (
            "COCO_people_subset.zip",
            {
                "boundingbox",
                "keypoints",
                "segmentation",
                "classification",
                "instance_segmentation",
            },
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            {"boundingbox", "classification"},
        ),
        (
            "roboflow://team-roboflow/coco-128/2/coco",
            {"boundingbox", "classification"},
        ),
        (
            "ultralytics://ultralytics/datasets/coco8",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.voc.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.darknet.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.yolov4pytorch.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.mt-yolov6.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.createml.zip",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.tensorflow.zip",
            {"boundingbox", "classification"},
        ),
        (
            "D2_Tile.png-mask-semantic.zip",
            {"segmentation", "classification"},
        ),
        (
            "Flowers_Classification.v2-raw.folder.zip",
            {"classification"},
        ),
        (
            "D2_ParkingLot.zip",
            {"boundingbox", "segmentation", "classification", "keypoints"},
        ),
        (
            "D2_ParkingLot_Native.zip",
            {
                "boundingbox",
                "instance_segmentation",
                "classification",
                "keypoints",
                "metadata/color",
                "metadata/brand",
            },
        ),
        (
            "horse_pose.v8i.yolov8.zip",
            {"boundingbox", "classification", "keypoints"},
        ),
        (
            "medical-pills.zip",
            {"boundingbox", "classification"},
        ),
        (
            "crack-seg.zip",
            {
                "boundingbox",
                "classification",
                "instance_segmentation",
            },
        ),
    ],
)
def test_dir_parser(
    dataset_name: str,
    url: str,
    expected_task_types: set[str],
    storage_url: str,
    tempdir: Path,
):
    if not url.startswith(("roboflow://", "ultralytics://")):
        url = f"{storage_url}/{url}"
    elif url.startswith("roboflow://") and environ.ROBOFLOW_API_KEY is None:
        pytest.skip("Roboflow API key is not set")
    elif (
        url.startswith("ultralytics://")
        and environ.ULTRALYTICS_API_KEY is None
    ):
        pytest.skip("Ultralytics API key is not set")

    parser = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    dataset = parser.parse()
    assert len(dataset) > 0
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == expected_task_types
    dataset.delete_dataset(delete_local=True)


@pytest.mark.parametrize(
    ("url", "dataset_type", "expected_task_types"),
    [
        (
            "COCO_people_subset.zip",
            "coco",
            {
                "boundingbox",
                "keypoints",
                "segmentation",
                "classification",
                "instance_segmentation",
            },
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            "coco",
            {"boundingbox", "classification"},
        ),
        (
            "roboflow://team-roboflow/coco-128/2/coco",
            "coco",
            {"boundingbox", "classification"},
        ),
        (
            "ultralytics://ultralytics/datasets/coco8",
            "ultralytics-ndjson",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.voc.zip",
            "voc",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.darknet.zip",
            "darknet",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.yolov4pytorch.zip",
            "yolov4",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.mt-yolov6.zip",
            "yolov6",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.createml.zip",
            "createml",
            {"boundingbox", "classification"},
        ),
        (
            "Thermal_Dogs_and_People.v1-resize-416x416.tensorflow.zip",
            "tfcsv",
            {"boundingbox", "classification"},
        ),
        (
            "D2_Tile.png-mask-semantic.zip",
            "segmask",
            {"segmentation", "classification"},
        ),
        (
            "Flowers_Classification.v2-raw.folder.zip",
            "clsdir",
            {"classification"},
        ),
        (
            "D2_ParkingLot.zip",
            "solo",
            {"boundingbox", "segmentation", "classification", "keypoints"},
        ),
        (
            "D2_ParkingLot_Native.zip",
            "native",
            {
                "boundingbox",
                "instance_segmentation",
                "classification",
                "keypoints",
                "metadata/color",
                "metadata/brand",
            },
        ),
        (
            "horse_pose.v8i.yolov8.zip",
            "yolov8",
            {"boundingbox", "classification", "keypoints"},
        ),
        (
            "medical-pills.zip",
            "yolov8",
            {"boundingbox", "classification"},
        ),
        (
            "crack-seg.zip",
            "yolov8",
            {
                "boundingbox",
                "classification",
                "instance_segmentation",
            },
        ),
    ],
)
def test_dir_parser_explicit_type(
    dataset_name: str,
    url: str,
    dataset_type: str,
    expected_task_types: set[str],
    storage_url: str,
    tempdir: Path,
):
    if not url.startswith(("roboflow://", "ultralytics://")):
        url = f"{storage_url}/{url}"
    elif url.startswith("roboflow://") and environ.ROBOFLOW_API_KEY is None:
        pytest.skip("Roboflow API key is not set")
    elif (
        url.startswith("ultralytics://")
        and environ.ULTRALYTICS_API_KEY is None
    ):
        pytest.skip("Ultralytics API key is not set")

    parser = LuxonisParser(
        url,
        dataset_name=dataset_name,
        dataset_type=dataset_type,  # type: ignore
        delete_local=True,
        save_dir=tempdir,
    )
    dataset = parser.parse()
    assert len(dataset) > 0
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == expected_task_types
    dataset.delete_dataset(delete_local=True)


def test_skipped_annotation_warnings_include_context(
    dataset_name: str, tempdir: Path, logot: Logot
):
    dataset_dir = tempdir / "coco_issues"
    split_dir = dataset_dir / "train"
    image_dir = split_dir / "data"
    image_dir.mkdir(parents=True)

    valid_image = image_dir / "valid.jpg"
    crowd_image = image_dir / "crowd.jpg"
    valid_image.write_bytes(b"")
    crowd_image.write_bytes(b"")

    labels_path = split_dir / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": valid_image.name,
                        "width": 100,
                        "height": 100,
                    },
                    {
                        "id": 2,
                        "file_name": crowd_image.name,
                        "width": 100,
                        "height": 100,
                    },
                    {
                        "id": 3,
                        "file_name": "missing.jpg",
                        "width": 100,
                        "height": 100,
                    },
                ],
                "annotations": [
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                    },
                    {
                        "id": 11,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [15, 15, 10, 10],
                        "iscrowd": 1,
                    },
                    {
                        "id": 13,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, "inf", 20, 20],
                    },
                    {
                        "id": 12,
                        "image_id": 3,
                        "category_id": 1,
                        "bbox": [5, 5, 10, 10],
                    },
                ],
                "categories": [{"id": 1, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )

    parser = LuxonisParser(
        str(split_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    dataset = parser.parse()
    try:
        assert len(dataset) == 1

        logot.assert_logged(
            logged.warning(
                "Skipping annotation: Annotation contains non-numeric "
                "bbox values "
                f"(annotation_id=13, source={labels_path}, "
                f"image={valid_image.resolve()})"
            )
        )
        logot.assert_logged(
            logged.warning(
                "Skipping annotation: COCO annotation has iscrowd=1 "
                f"(annotation_id=11, source={labels_path}, "
                f"image={crowd_image.resolve()})"
            )
        )
        logot.assert_logged(
            logged.warning(
                "Skipping annotation: referenced image file does not exist "
                f"(source={labels_path}, "
                f"image={(image_dir / 'missing.jpg').resolve()})"
            )
        )
    finally:
        dataset.delete_dataset(delete_local=True)


def test_ultralytics_ndjson_parser(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson.zip"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert len(dataset) > 0
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == {
        "boundingbox",
        "classification",
    }
    dataset.delete_dataset(delete_local=True)


class _DummyDataset:
    def add(self, _generator: Iterator[object]) -> None:
        return None

    def make_splits(self, _splits: object) -> None:
        return None


class _WarningParser(BaseParser):
    def __init__(
        self,
        warning_count: int,
        *,
        reason: str = "dummy skipped annotation",
        full_warnings: bool = False,
    ):
        super().__init__(
            cast(BaseDataset, _DummyDataset()),
            DatasetType.COCO,
            None,
            full_warnings=full_warnings,
        )
        self.warning_count = warning_count
        self.reason = reason

    @staticmethod
    def validate_split(_split_path: Path) -> dict[str, Any]:
        return {}

    def from_dir(
        self, dataset_dir: Path, **kwargs: Any
    ) -> tuple[list[Path], list[Path], list[Path]]:
        return [], [], []

    def from_split(self, **kwargs: Any) -> ParserOutput:
        for annotation_id in range(self.warning_count):
            self._warn_skipped_annotation(
                self.reason,
                annotation_id=annotation_id,
            )
        return iter(()), {}, []


def _assert_skipped_annotation_warning(
    logot: Logot, reason: str, annotation_id: int
) -> None:
    logot.assert_logged(
        logged.warning(
            f"Skipping annotation: {reason} (annotation_id={annotation_id})"
        )
    )


def test_skipped_annotation_warnings_are_capped(logot: Logot):
    parser = _WarningParser(BaseParser.SKIPPED_WARNING_LIMIT + 5)
    parser.parse_split()

    for annotation_id in range(BaseParser.SKIPPED_WARNING_LIMIT):
        _assert_skipped_annotation_warning(logot, parser.reason, annotation_id)
    logot.assert_not_logged(
        logged.warning(
            f"Skipping annotation: {parser.reason} "
            f"(annotation_id={BaseParser.SKIPPED_WARNING_LIMIT})"
        )
    )

    logot.clear()
    parser = _WarningParser(BaseParser.SKIPPED_WARNING_LIMIT + 5)
    parser.parse_split()
    logot.assert_logged(
        logged.warning(
            "Skipped logging 5 additional warnings. Enable the "
            "`--log-all-warnings` flag to see the full list."
        )
    )
    logot.assert_logged(
        logged.warning(
            "Skipped annotations: dummy skipped annotation (55 records)"
        )
    )


def test_full_warnings_logs_all_skipped_annotation_warnings(logot: Logot):
    parser = _WarningParser(
        BaseParser.SKIPPED_WARNING_LIMIT + 5, full_warnings=True
    )
    parser.parse_split()

    for annotation_id in range(BaseParser.SKIPPED_WARNING_LIMIT + 5):
        _assert_skipped_annotation_warning(logot, parser.reason, annotation_id)
    logot.assert_not_logged(
        logged.warning(
            "Skipped logging %d additional warnings. Enable the "
            "`--log-all-warnings` flag to see the full list."
        )
    )


def test_ultralytics_ndjson_parser_explicit_type(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson.zip"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        dataset_type="ultralytics-ndjson",  # type: ignore[arg-type]
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert len(dataset) > 0
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == {
        "boundingbox",
        "classification",
    }
    dataset.delete_dataset(delete_local=True)


def test_ultralytics_ndjson_remote_urls_parser(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson_remote/fruit.ndjson"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert len(dataset) > 0
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == {
        "boundingbox",
        "classification",
    }
    dataset.delete_dataset(delete_local=True)


def test_ultralytics_ndjson_remote_urls_parser_reuses_existing_remote_dir(
    dataset_name: str,
    tempdir: Path,
):
    source = create_image(10, tempdir)
    ndjson_path = tempdir / "budgie.ndjson"
    ndjson_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "dataset",
                        "class_names": {"0": "budgie"},
                    }
                ),
                json.dumps(
                    {
                        "type": "image",
                        "file": "train/img1.jpg",
                        "url": source.resolve().as_uri(),
                        "split": "train",
                        "width": 512,
                        "height": 512,
                        "annotations": {"boxes": [[0, 0.5, 0.5, 0.4, 0.4]]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (tempdir / "budgie").mkdir()

    dataset = LuxonisParser(
        str(ndjson_path),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()
    try:
        assert len(dataset) == 1
    finally:
        dataset.delete_dataset(delete_local=True)


def test_ultralytics_ndjson_remote_urls_parser_rejects_existing_remote_dir_when_cache_disabled(
    dataset_name: str,
    tempdir: Path,
):
    source = create_image(10, tempdir)
    ndjson_path = tempdir / "budgie.ndjson"
    ndjson_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "dataset",
                        "class_names": {"0": "budgie"},
                    }
                ),
                json.dumps(
                    {
                        "type": "image",
                        "file": "train/img1.jpg",
                        "url": source.resolve().as_uri(),
                        "split": "train",
                        "width": 512,
                        "height": 512,
                        "annotations": {"boxes": [[0, 0.5, 0.5, 0.4, 0.4]]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (tempdir / "budgie").mkdir()

    with pytest.raises(
        ValueError,
        match=r"Remote NDJSON image directory '.*budgie' already exists",
    ):
        LuxonisParser(
            str(ndjson_path),
            dataset_name=dataset_name,
            delete_local=True,
            save_dir=tempdir,
        ).parse(reuse_cached=False)


def test_partial_split_clsdir_is_preserved(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "clsdir_partial"
    split_dir = dataset_dir / "valid" / "budgie"
    split_dir.mkdir(parents=True)
    create_image(16, split_dir)

    dataset = LuxonisParser(
        str(dataset_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 0
    dataset.delete_dataset(delete_local=True)


def test_partial_split_clsdir_explicit_type_uses_dir_mode(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "clsdir_partial_explicit"
    split_dir = dataset_dir / "test" / "finch"
    split_dir.mkdir(parents=True)
    create_image(16, split_dir)

    dataset = LuxonisParser(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="clsdir",  # type: ignore[arg-type]
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 1
    dataset.delete_dataset(delete_local=True)


@pytest.mark.parametrize(
    ("url", "expected_split_sizes", "loader_view"),
    [
        (
            "coco_valid_only_debug.zip",
            {"train": 0, "val": 2, "test": 1},
            "val",
        ),
        (
            "native_val_only_debug.zip",
            {"train": 0, "val": 3, "test": 0},
            "val",
        ),
    ],
)
def test_partial_split_fixture_is_preserved(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    expected_split_sizes: dict[str, int],
    loader_view: str,
):
    dataset = LuxonisParser(
        f"{storage_url.rstrip('/')}/{url}",
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert {
        split_name: len(group_ids) for split_name, group_ids in splits.items()
    } == expected_split_sizes

    loader = LuxonisLoader(dataset, view=loader_view)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == {"boundingbox", "classification"}
    dataset.delete_dataset(delete_local=True)


def test_partial_ultralytics_layout_reports_yolov6_yolov8_ambiguity(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "yolo_partial"
    image_dir = dataset_dir / "images" / "test"
    label_dir = dataset_dir / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    create_image(16, image_dir)
    (label_dir / "img_16.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    with pytest.raises(
        ValueError,
        match=(
            r"ambiguous between YOLOv6 and YOLOv8\. Please specify dataset_type\."
        ),
    ):
        LuxonisParser(
            str(dataset_dir),
            dataset_name=dataset_name,
            delete_local=True,
            save_dir=tempdir,
        ).parse()


def test_partial_split_train_only_roboflow_coco_keeps_format_detection(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "coco_train_only_roboflow"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True)
    create_image(16, train_dir)
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "img_16.jpg",
                        "width": 512,
                        "height": 512,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [128, 128, 256, 256],
                        "area": 65536,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 0, "name": "budgie"}],
            }
        )
    )

    dataset = LuxonisParser(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="coco",  # type: ignore[arg-type]
        delete_local=True,
        save_dir=tempdir,
    ).parse(use_keypoint_ann=True)

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 0
    dataset.delete_dataset(delete_local=True)
