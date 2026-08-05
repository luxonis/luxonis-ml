import json
import zipfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from loguru import logger

import luxonis_ml.data as data_module
from luxonis_ml.data import (
    PARSERS_REGISTRY,
    DatasetIterator,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
    ParsedDataset,
    ParseIssueCollector,
    ParserIssue,
    ParserPlugin,
    register_parser_plugin,
)
from luxonis_ml.data.datasets.annotation import DatasetRecord, Detection
from luxonis_ml.data.datasets.base_dataset import _prepare_import_records
from luxonis_ml.data.parsers import (
    COCOParser,
    SOLOParser,
    UltralyticsNDJSONParser,
    YOLOv8Parser,
)
from luxonis_ml.data.parsers.parser_plugin import get_parser_plugin
from luxonis_ml.data.parsers.source import prepare_source
from luxonis_ml.data.utils import get_task_type
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import environ

from .utils import create_image


def test_parser_entry_point_loading(monkeypatch: pytest.MonkeyPatch):
    class EntryPointParser(ParserPlugin):
        dataset_types = ("test-entry-point",)

        @classmethod
        def supports(cls, source: Path) -> bool:
            return False

        def parse(
            self,
            source: Path,
            *,
            dataset_type: str,
            **kwargs: Any,
        ) -> ParsedDataset:
            raise AssertionError("Plugin should only be registered")

    class EntryPoint:
        @staticmethod
        def load() -> type[ParserPlugin]:
            return EntryPointParser

    monkeypatch.setattr(
        data_module,
        "_get_entry_points_subset",
        lambda group: [EntryPoint()] if group == "parser_plugins" else [],
    )
    data_module._load_parser_plugins()
    assert PARSERS_REGISTRY.get("test-entry-point") is EntryPointParser


def test_custom_parser_plugin_import(
    dataset_name: str,
    tempdir: Path,
):
    source = tempdir / "sample.plugin"
    source.write_text("custom")
    image = create_image(0, tempdir)

    class CustomParser(ParserPlugin):
        dataset_types = ("test-custom", "test-custom-alias")

        @classmethod
        def supports(cls, source: Path) -> bool:
            return source.suffix == ".plugin"

        def parse(
            self,
            source: Path,
            *,
            dataset_type: str,
            **kwargs: Any,
        ) -> ParsedDataset:
            assert source.suffix == ".plugin"
            assert dataset_type == "test-custom-alias"
            assert kwargs == {"label": "budgie"}
            return ParsedDataset(
                iter(
                    [
                        {
                            "file": image,
                            "annotation": {"class": "budgie"},
                        }
                    ]
                ),
                {},
                [image],
            )

    register_parser_plugin(CustomParser, force=True)
    with pytest.raises(KeyError, match="already registered"):
        register_parser_plugin(CustomParser)
    dataset = LuxonisDataset.import_dataset(
        source,
        dataset_name=dataset_name,
        dataset_type="test-custom-alias",
        parser_kwargs={"label": "budgie"},
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 1
        assert dataset.get_parser_issue_messages() == []
    finally:
        dataset.delete_dataset(delete_local=True)


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

    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    assert len(dataset) > 0
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == expected_task_types
    dataset.delete_dataset(delete_local=True)


def test_split_parser_creates_default_splits(dataset_name: str, tempdir: Path):
    class_dir = tempdir / "flat_cls"
    image_dir = class_dir / "class_a"
    image_dir.mkdir(parents=True)
    create_image(0, image_dir)

    dataset = LuxonisDataset.import_dataset(
        str(class_dir),
        dataset_name=dataset_name,
        dataset_type=DatasetType.CLSDIR,
        delete_local=True,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert set(splits) == {"train", "val", "test"}
        assert sum(len(group_ids) for group_ids in splits.values()) == 1

        loader = LuxonisLoader(dataset)
        next(iter(loader))
    finally:
        dataset.delete_dataset(delete_local=True)


def test_count_split_filters_unselected_records(
    dataset_name: str,
    tempdir: Path,
):
    class_dir = tempdir / "counted_cls" / "class_a"
    class_dir.mkdir(parents=True)
    for index in range(5):
        create_image(index, class_dir)

    dataset = LuxonisDataset.import_dataset(
        str(class_dir.parent),
        dataset_name=dataset_name,
        dataset_type=DatasetType.CLSDIR,
        split_ratios={"train": 2, "val": 1, "test": 1},
        delete_local=True,
    )
    try:
        assert len(dataset) == 4
        splits = dataset.get_splits()
        assert splits is not None
        assert {name: len(ids) for name, ids in splits.items()} == {
            "train": 2,
            "val": 1,
            "test": 1,
        }
    finally:
        dataset.delete_dataset(delete_local=True)


def test_count_split_matches_relative_fiftyone_paths(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "fiftyone_counts"
    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True)
    images = [create_image(index, data_dir) for index in range(4)]
    (dataset_dir / "labels.json").write_text(
        json.dumps(
            {
                "classes": ["class_a"],
                "labels": {image.stem: 0 for image in images},
            }
        )
    )

    dataset = LuxonisDataset.import_dataset(
        dataset_dir,
        dataset_name=dataset_name,
        dataset_type=DatasetType.FIFTYONECLS,
        split_ratios={"train": 2, "val": 1, "test": 1},
        delete_local=True,
    )
    try:
        assert len(dataset) == 4
        splits = dataset.get_splits()
        assert splits is not None
        assert {name: len(ids) for name, ids in splits.items()} == {
            "train": 2,
            "val": 1,
            "test": 1,
        }
    finally:
        dataset.delete_dataset(delete_local=True)


def test_classification_directory_does_not_claim_data_directory(
    tempdir: Path,
):
    data_dir = tempdir / "coco" / "test" / "data"
    data_dir.mkdir(parents=True)
    create_image(0, data_dir)

    plugin = PARSERS_REGISTRY.get(DatasetType.CLSDIR.value)
    assert not plugin.supports(data_dir.parent.parent)


def test_coco_count_splits_do_not_create_synthetic_test_split(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "coco_counts"
    for split in ("train", "validation"):
        data_dir = dataset_dir / split / "data"
        data_dir.mkdir(parents=True)
        image = create_image(0, data_dir)
        (dataset_dir / split / "labels.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": image.name,
                            "width": 640,
                            "height": 480,
                        }
                    ],
                    "annotations": [],
                    "categories": [],
                }
            )
        )
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir()
    (raw_dir / "person_keypoints.json").write_text("{}")

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        split_ratios={"train": 1, "val": 1, "test": 1},
        delete_local=True,
    )
    try:
        assert len(dataset) == 2
        splits = dataset.get_splits()
        assert splits is not None
        assert {name: len(ids) for name, ids in splits.items()} == {
            "train": 1,
            "val": 1,
            "test": 0,
        }
    finally:
        dataset.delete_dataset(delete_local=True)


def test_fiftyone_classification_parser_discovers_validation_split(
    dataset_name: str, tempdir: Path
):
    dataset_dir = tempdir / "fiftyone_cls"
    for i, split in enumerate(["train", "validation", "test"]):
        data_dir = dataset_dir / split / "data"
        data_dir.mkdir(parents=True)
        image_path = create_image(i, data_dir)
        labels = {
            "classes": ["daisy", "dandelion"],
            "labels": {image_path.stem: i % 2},
        }
        (dataset_dir / split / "labels.json").write_text(json.dumps(labels))

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type=DatasetType.FIFTYONECLS,
        delete_local=True,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert {name: len(ids) for name, ids in splits.items()} == {
            "train": 1,
            "val": 1,
            "test": 1,
        }
    finally:
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

    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        delete_local=True,
        save_dir=tempdir,
    )
    assert len(dataset) > 0
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    task_types = {get_task_type(task) for task in ann}
    assert task_types == expected_task_types
    dataset.delete_dataset(delete_local=True)


def test_parser_issue_messages_collect_skipped_annotations(
    dataset_name: str, tempdir: Path
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

    dataset = LuxonisDataset.import_dataset(
        str(split_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 1

        issues = dataset.get_parser_issue_messages()
        assert len(issues) == 3
        assert {issue.parser_issue for issue in issues} == {
            ParserIssue.COCO_ISCROWD,
            ParserIssue.MISSING_IMAGE,
            ParserIssue.NON_NUMERIC_ANNOTATION,
        }

        crowd_issue = next(
            issue
            for issue in issues
            if issue.parser_issue is ParserIssue.COCO_ISCROWD
        )
        assert crowd_issue.reason == "COCO annotation has iscrowd=1"
        assert crowd_issue.source == labels_path
        assert crowd_issue.image == crowd_image.resolve()
        assert crowd_issue.annotation_id == 11

        non_numeric_issue = next(
            issue
            for issue in issues
            if issue.parser_issue is ParserIssue.NON_NUMERIC_ANNOTATION
        )
        assert (
            non_numeric_issue.reason
            == "Annotation contains non-numeric bbox values"
        )
        assert non_numeric_issue.source == labels_path
        assert non_numeric_issue.image == valid_image.resolve()
        assert non_numeric_issue.annotation_id == 13

        missing_image_issue = next(
            issue
            for issue in issues
            if issue.parser_issue is ParserIssue.MISSING_IMAGE
        )
        assert (
            missing_image_issue.reason
            == "referenced image file does not exist"
        )
        assert missing_image_issue.source == labels_path
        assert (
            missing_image_issue.image == (image_dir / "missing.jpg").resolve()
        )
        assert missing_image_issue.annotation_id is None

        issues.pop()
        assert len(dataset.get_parser_issue_messages()) == 3
    finally:
        dataset.delete_dataset(delete_local=True)


def test_ultralytics_ndjson_parser(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson.zip"
    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )

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


def test_skipped_annotation_warnings_are_capped():
    warning_limit = 10
    warning_count = warning_limit + 5
    collector = ParseIssueCollector(warning_limit=warning_limit)
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )

    try:
        for annotation_id in range(warning_count):
            collector.warn(
                ParserIssue.NON_NUMERIC_ANNOTATION,
                "dummy skipped annotation",
                annotation_id=annotation_id,
            )
        collector.log_summary()
    finally:
        logger.remove(sink_id)

    assert len(collector.messages) == warning_count
    assert (
        sum(message.startswith("Skipping annotation:") for message in messages)
        == warning_limit
    )
    assert (
        "Skipped logging 5 additional warnings. Enable the "
        "`--log-all-warnings` flag to see the full list."
    ) in messages
    assert (
        f"Skipped annotations: dummy skipped annotation ({warning_count} records)"
        in messages
    )


def test_full_warnings_logs_all_skipped_annotation_warnings():
    warning_limit = 10
    warning_count = warning_limit + 5
    collector = ParseIssueCollector(
        full_warnings=True,
        warning_limit=warning_limit,
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )

    try:
        for annotation_id in range(warning_count):
            collector.warn(
                ParserIssue.NON_NUMERIC_ANNOTATION,
                "dummy skipped annotation",
                annotation_id=annotation_id,
            )
        collector.log_summary()
    finally:
        logger.remove(sink_id)

    assert len(collector.messages) == warning_count
    assert (
        sum(message.startswith("Skipping annotation:") for message in messages)
        == warning_count
    )
    assert not any(
        message.startswith("Skipped logging ") for message in messages
    )


def test_ultralytics_ndjson_parser_explicit_type(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    url = f"{storage_url.rstrip('/')}/fruit_ndjson.zip"
    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        dataset_type="ultralytics-ndjson",
        delete_local=True,
        save_dir=tempdir,
    )

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
    dataset = LuxonisDataset.import_dataset(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )

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

    dataset = LuxonisDataset.import_dataset(
        str(ndjson_path),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
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
        LuxonisDataset.import_dataset(
            str(ndjson_path),
            dataset_name=dataset_name,
            delete_local=True,
            save_dir=tempdir,
            parser_kwargs={"reuse_cached": False},
        )


def test_partial_split_clsdir_is_preserved(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "clsdir_partial"
    split_dir = dataset_dir / "valid" / "budgie"
    split_dir.mkdir(parents=True)
    create_image(16, split_dir)

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )

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

    with pytest.warns(DeprecationWarning, match="LuxonisParser"):
        parser = LuxonisParser(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="clsdir",
            delete_local=True,
            save_dir=tempdir,
        )
    dataset = parser.parse()

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 1
    assert parser.get_parser_issue_messages() == []
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
    dataset = LuxonisDataset.import_dataset(
        f"{storage_url.rstrip('/')}/{url}",
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )

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
            r"compatible with multiple parsers: yolov6, yolov8\. "
            r"Please specify `dataset_type`\."
        ),
    ):
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            delete_local=True,
            save_dir=tempdir,
        )


def test_ultralytics_layout_with_val_split_detects_yolov8(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "yolo_ultralytics"
    for index, split_name in enumerate(("train", "val")):
        image_dir = dataset_dir / "images" / split_name
        label_dir = dataset_dir / "labels" / split_name
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        create_image(index, image_dir)
        (label_dir / f"img_{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    # `images/valid` is absent, so only the YOLOv8 parser recognizes both
    # splits even though the YOLOv6 parser also recognizes `images/train`.
    plugin, dataset_type = get_parser_plugin(dataset_dir, None)
    assert dataset_type == "yolov8"
    assert plugin is YOLOv8Parser

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 2
    finally:
        dataset.delete_dataset(delete_local=True)


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

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="coco",
        delete_local=True,
        save_dir=tempdir,
        parser_kwargs={"use_keypoint_ann": True},
    )

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 0
    dataset.delete_dataset(delete_local=True)


def _collect_dataset_records(records: DatasetIterator) -> list[DatasetRecord]:
    """Collect records that `_prepare_import_records` has already parsed."""
    collected = []
    for record in records:
        assert isinstance(record, DatasetRecord)
        collected.append(record)
    return collected


def _collect_raw_records(records: DatasetIterator) -> list[dict[str, Any]]:
    """Collect parser output, which plugins emit as plain dictionaries."""
    collected = []
    for record in records:
        assert isinstance(record, dict)
        collected.append(record)
    return collected


def _write_yolov8_split(
    split_path: Path,
    image_indices: Sequence[int],
    *,
    annotate: Callable[[int], str | None] = lambda _: "0 0.5 0.5 0.4 0.4\n",
) -> None:
    """Write a Roboflow-style YOLOv8 split.

    Args:
        split_path: Split directory to populate.
        image_indices: Indices passed to `create_image`.
        annotate: Label file content per index. ``None`` writes no label file
            at all, which is how parsers see a background image.

    """
    image_dir = split_path / "images"
    label_dir = split_path / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in image_indices:
        create_image(index, image_dir)
        content = annotate(index)
        if content is not None:
            (label_dir / f"img_{index}.txt").write_text(content)


def test_prepare_import_records_keeps_unannotated_records_in_any_order(
    tempdir: Path,
):
    """Records without annotations must survive a string ``task_name``.

    Regression: a string ``task_name`` was wrapped in an empty
    ``defaultdict``, which only materializes keys on lookup, so the fan-out
    set for annotation-less records was empty until some annotated record had
    already been seen. Background images were silently dropped, and which ones
    depended on iteration order — hence both orderings are checked here.
    """
    unannotated = DatasetRecord(
        files={"image": create_image(0, tempdir)}, annotation=None
    )
    annotated = DatasetRecord(
        files={"image": create_image(1, tempdir)},
        annotation=Detection.model_validate({"class": "budgie"}),
    )

    for records in ([unannotated, annotated], [annotated, unannotated]):
        prepared = _collect_dataset_records(
            _prepare_import_records(
                iter(records),
                task_name="birds",
                selected_files=None,
            )
        )
        assert len(prepared) == 2
        assert {record.task_name for record in prepared} == {"birds"}

    # A class-to-task mapping instead fans an annotation-less record out over
    # every distinct task name, so it is not lost from any of them either.
    prepared = _collect_dataset_records(
        _prepare_import_records(
            iter([unannotated]),
            task_name={"budgie": "birds", "dog": "mammals"},
            selected_files=None,
        )
    )
    assert {record.task_name for record in prepared} == {"birds", "mammals"}


def test_prepare_import_records_does_not_copy_annotations(tempdir: Path):
    """Assigning a task name must not duplicate annotation payloads.

    Regression: the task name was applied with
    ``model_copy(update=..., deep=True)``, deep-copying every polygon list and
    mask array once per record. Only ``task_name`` changes, so a shallow copy
    is enough; the annotation object is shared and the input record is left
    untouched.
    """
    record = DatasetRecord(
        files={"image": create_image(0, tempdir)},
        annotation=Detection.model_validate({"class": "budgie"}),
    )

    (prepared,) = _collect_dataset_records(
        _prepare_import_records(
            iter([record]),
            task_name="birds",
            selected_files=None,
        )
    )

    assert prepared.task_name == "birds"
    assert prepared.annotation is record.annotation
    assert record.task_name != "birds"


@pytest.mark.parametrize(
    ("split_ratios", "expected_sizes"),
    [
        ({"train": 2}, {"train": 2, "val": 0, "test": 0}),
        ({"val": 1, "test": 1}, {"train": 0, "val": 1, "test": 1}),
    ],
)
def test_count_split_ratios_may_omit_splits(
    dataset_name: str,
    tempdir: Path,
    split_ratios: dict[str, float | int],
    expected_sizes: dict[str, int],
):
    """Count-based ``split_ratios`` may name only some of the splits.

    Regression: the count helpers indexed ``split_ratios["train"]``,
    ``["val"]`` and ``["test"]`` unconditionally, so a partial mapping raised a
    bare ``KeyError`` — after the dataset had already been created on disk.
    Percentage-based ratios always allowed partial mappings. Splits left out of
    the mapping are treated as :math:`0`.
    """
    dataset_dir = tempdir / "yolo_counts"
    _write_yolov8_split(dataset_dir / "train", range(4))
    _write_yolov8_split(dataset_dir / "valid", range(4, 6))
    _write_yolov8_split(dataset_dir / "test", range(6, 8))
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="yolov8",
        split_ratios=split_ratios,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert {
            name: len(group_ids) for name, group_ids in splits.items()
        } == expected_sizes
    finally:
        dataset.delete_dataset(delete_local=True)


@pytest.mark.parametrize(
    "split_ratios", [{"train": 0, "val": 0, "test": 0}, {}]
)
def test_zero_count_split_ratios_fail_before_creating_dataset(
    dataset_name: str,
    tempdir: Path,
    split_ratios: dict[str, float | int],
):
    """Counts selecting nothing must fail loudly and leave nothing behind.

    Regression: zero counts filtered out every record, so ``make_splits``
    raised ``FileNotFoundError: Dataset is empty`` — but only after the dataset
    had been created and registered, leaving an orphaned empty dataset on
    disk. An empty mapping hit the same path, because ``all()`` over no values
    classifies it as count-based.
    """
    dataset_dir = tempdir / "yolo_zero_counts"
    _write_yolov8_split(dataset_dir / "train", range(2))
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    with pytest.raises(ValueError, match="must request at least one sample"):
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="yolov8",
            split_ratios=split_ratios,
            delete_local=True,
            save_dir=tempdir,
        )

    assert not LuxonisDataset.exists(dataset_name)


def test_uppercase_image_extensions_are_recognized(
    dataset_name: str,
    tempdir: Path,
):
    """Images with uppercase extensions must not be invisible to parsers.

    Regression: ``_list_images`` matched ``image.suffix`` against a
    case-sensitive set, so ``.JPG`` was unknown. Every ``validate_split`` bails
    out on an empty image list, which rejected the whole dataset; a mixed-case
    dataset was worse, importing while silently omitting the uppercase files.
    """
    dataset_dir = tempdir / "yolo_uppercase"
    _write_yolov8_split(dataset_dir / "train", [0])
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    image_dir = dataset_dir / "train" / "images"
    (image_dir / "img_0.jpg").rename(image_dir / "IMG_1.JPG")
    (dataset_dir / "train" / "labels" / "img_0.txt").rename(
        dataset_dir / "train" / "labels" / "IMG_1.txt"
    )

    # Auto-detection is used deliberately: an empty image list previously made
    # every parser reject the layout, not just the intended one.
    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 1
    finally:
        dataset.delete_dataset(delete_local=True)


def test_yolov8_truncated_annotation_line_is_skipped(
    dataset_name: str,
    tempdir: Path,
):
    """A too-short YOLOv8 label line must be reported, not fatal.

    Regression: ``task_type`` was only assigned for lines with exactly 5 or
    more than 5 values, then read unconditionally. A truncated line killed the
    whole import with ``UnboundLocalError`` — or, if an earlier line had set
    ``task_type``, with ``ValueError: not enough values to unpack``.
    """
    dataset_dir = tempdir / "yolo_truncated"
    _write_yolov8_split(
        dataset_dir / "train",
        [0, 1],
        annotate=lambda index: (
            "0 0.5 0.5 0.2\n" if index == 0 else "0 0.5 0.5 0.4 0.4\n"
        ),
    )
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="yolov8",
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        issues = dataset.get_parser_issue_messages()
        assert [issue.parser_issue for issue in issues] == [
            ParserIssue.MALFORMED_ANNOTATION
        ]
        # The import completes and the well-formed image is kept. The image
        # whose only label line was malformed yields no record at all, the
        # same as any other fully-skipped annotation.
        assert len(dataset) == 1
    finally:
        dataset.delete_dataset(delete_local=True)


def test_coco_single_split_rejects_keypoint_options(
    dataset_name: str,
    tempdir: Path,
):
    """Keypoint options must not be silently dropped for a single split.

    Regression: for a source without split directories, ``COCOParser.parse``
    delegated through ``**kwargs``, but ``use_keypoint_ann`` and
    ``keypoint_ann_paths`` are bound to named parameters and never reach it.
    They vanished without a warning, so a pose model could be trained on a
    dataset holding no keypoints at all. Previously this raised ``TypeError``.
    """
    dataset_dir = tempdir / "coco_single_split"
    image_dir = dataset_dir / "data"
    image_dir.mkdir(parents=True)
    create_image(0, image_dir)
    (dataset_dir / "labels.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "img_0.jpg",
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

    with pytest.raises(ValueError, match="use_keypoint_ann"):
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="coco",
            parser_kwargs={"use_keypoint_ann": True},
            delete_local=True,
            save_dir=tempdir,
        )

    # Without the keypoint options the same single-split source still imports.
    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="coco",
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 1
    finally:
        dataset.delete_dataset(delete_local=True)


def test_clsdir_ignores_reserved_directory_names(
    dataset_name: str,
    tempdir: Path,
):
    """Reserved directory names must never be ingested as classes.

    Regression: ``validate_split`` skips directories belonging to other
    layouts (``data``, ``raw``, ``masks``, split names, ``images``,
    ``labels``), but ``_parse_split`` listed every subdirectory, so a source
    validated on its real class folders and then gained a bogus ``data``
    class. Both now share one reserved-name set.
    """
    dataset_dir = tempdir / "clsdir_reserved"
    for index, class_name in enumerate(("budgie", "parrot", "data")):
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True)
        create_image(index, class_dir)

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="clsdir",
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        class_names = {
            name
            for names in dataset.get_class_names().values()
            for name in names
        }
        assert class_names == {"budgie", "parrot"}
    finally:
        dataset.delete_dataset(delete_local=True)


def test_parser_entry_point_name_collision_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
):
    """A colliding entry-point plugin must not break ``import``.

    Regression: ``_load_parser_plugins`` runs unguarded at module scope, so a
    third-party plugin declaring a built-in ``dataset_types`` name made plain
    ``import luxonis_ml.data`` raise ``KeyError``, taking down every downstream
    consumer as soon as that package was installed. Registration is also
    all-or-nothing now: the collision is checked before any name is claimed, so
    the type declared ahead of ``"coco"`` must not be left behind.
    """

    class CollidingParser(ParserPlugin):
        dataset_types = ("test-before-collision", "coco")

        @classmethod
        def supports(cls, source: Path) -> bool:
            return False

        def parse(
            self,
            source: Path,
            *,
            dataset_type: str,
            **kwargs: Any,
        ) -> ParsedDataset:
            raise AssertionError("Plugin should never be selected")

    class EntryPoint:
        name = "colliding-plugin"

        @staticmethod
        def load() -> type[ParserPlugin]:
            return CollidingParser

    monkeypatch.setattr(
        data_module,
        "_get_entry_points_subset",
        lambda group: [EntryPoint()] if group == "parser_plugins" else [],
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        data_module._load_parser_plugins()
    finally:
        logger.remove(sink_id)

    assert any("colliding-plugin" in message for message in messages)
    assert PARSERS_REGISTRY.get("coco") is COCOParser
    assert "test-before-collision" not in PARSERS_REGISTRY


def test_trailing_separator_in_source_keeps_dataset_name(tempdir: Path):
    """A trailing separator must not erase the derived dataset name.

    Regression: the name came from ``rsplit("/", 1)[-1]``, which is ``""`` for
    a path written with a trailing slash. The dataset was then created as
    ``""``, writing its storage directories straight into the datasets root and
    merging into whatever a previous trailing-slash import had left there, and
    the remote download target collapsed to the working directory.
    """
    dataset_dir = tempdir / "trailing_source"
    dataset_dir.mkdir()

    assert prepare_source(f"{dataset_dir}/", None) == (
        dataset_dir,
        "trailing_source",
    )
    assert prepare_source(str(dataset_dir), None) == (
        dataset_dir,
        "trailing_source",
    )

    # A path that is nothing but separators has no name to derive at all.
    with pytest.raises(ValueError, match="Could not derive a dataset name"):
        prepare_source("/", None)


def test_luxonis_parser_rejects_unsupported_source_at_construction(
    tempdir: Path,
):
    """`LuxonisParser` must fail at construction, as it did before deprecation.

    Regression: the deprecated wrapper deferred source acquisition and format
    detection into ``parse``, so ``try: LuxonisParser(d) except ValueError``
    — the documented way to probe whether a directory is a recognized layout
    — always succeeded and the error surfaced much later.
    """
    unrecognized = tempdir / "not_a_dataset"
    unrecognized.mkdir()

    with (
        pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"),
        pytest.raises(
            ValueError,
            match="not in expected format for any registered parser",
        ),
    ):
        LuxonisParser(str(unrecognized))


def test_luxonis_parser_prepares_source_once(
    dataset_name: str,
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Repeated ``parse`` calls must not re-acquire the source.

    Regression: with acquisition deferred into ``parse``, every call
    re-downloaded the Roboflow or S3 source and re-extracted the ZIP. A local
    ZIP stands in for the remote case here, since extraction is the observable
    half of the same work.
    """
    source_dir = tempdir / "clsdir_zip_source"
    for index, class_name in enumerate(("budgie", "parrot")):
        class_dir = source_dir / class_name
        class_dir.mkdir(parents=True)
        create_image(index, class_dir)

    archive_path = tempdir / "clsdir_zip_source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for image_path in source_dir.rglob("*.jpg"):
            archive.write(
                image_path, image_path.relative_to(source_dir.parent)
            )

    extractions: list[Path] = []
    original_extractall = zipfile.ZipFile.extractall

    def counting_extractall(  # noqa: ANN202
        self,  # noqa: ANN001
        path=None,  # noqa: ANN001
        *args: Any,
        **kwargs: Any,
    ):
        extractions.append(Path(str(path)))
        return original_extractall(self, path, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "extractall", counting_extractall)

    with pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"):
        parser = LuxonisParser(
            str(archive_path),
            dataset_name=dataset_name,
            delete_local=True,
        )
    assert len(extractions) == 1

    first = parser.parse()
    second = parser.parse()
    try:
        assert len(extractions) == 1
        assert first.identifier == second.identifier
        assert len(first) == 2
    finally:
        first.delete_dataset(delete_local=True)


def _write_ultralytics_ndjson(
    ndjson_path: Path,
    image_dir: Path,
    *,
    missing_image: str,
) -> None:
    """Write a manifest holding a missing image and a non-image line."""
    image_dir.mkdir(parents=True)
    for index in range(3):
        create_image(index, image_dir)

    relative = image_dir.name
    lines: list[dict[str, Any]] = [
        {"type": "dataset", "class_names": ["budgie", "parrot"]},
        {
            "type": "image",
            "file": f"{relative}/img_0.jpg",
            "split": "train",
            "width": 512,
            "height": 512,
            "annotations": {"boxes": [[0, 0.5, 0.5, 0.2, 0.2]]},
        },
        {
            "type": "image",
            "file": f"{relative}/{missing_image}",
            "split": "train",
            "width": 512,
            "height": 512,
            "annotations": {"boxes": [[1, 0.5, 0.5, 0.2, 0.2]]},
        },
        {"type": "annotation-definitions", "note": "not an image record"},
        {
            "type": "image",
            "file": f"{relative}/img_1.jpg",
            "split": "val",
            "width": 512,
            "height": 512,
            "annotations": {},
        },
        {
            "type": "image",
            "file": f"{relative}/img_2.jpg",
            "split": "test",
            "width": 512,
            "height": 512,
            "annotations": {"boxes": [[1, 0.5, 0.5, 0.2, 0.2]]},
        },
    ]
    ndjson_path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )


def test_ultralytics_ndjson_skips_missing_images_without_shifting_records(
    tempdir: Path,
):
    """Skipped and non-image lines must stay aligned across both passes.

    ``ParsedDataset`` requires ``files`` and ``splits`` to be complete before
    ``records`` is consumed, so image paths are resolved in an eager pass while
    the annotations are re-read lazily in a second one. This guards that split:
    a record whose image is missing, and a line that is not an image at all,
    must be dropped by both passes, or every following record would be paired
    with the wrong image. It is a safety net for the streaming design rather
    than a past bug — the earlier implementation paired records with paths in
    one pass and could not drift.
    """
    ndjson_path = tempdir / "ndjson_alignment" / "dataset.ndjson"
    ndjson_path.parent.mkdir(parents=True)
    _write_ultralytics_ndjson(
        ndjson_path,
        ndjson_path.parent / "images",
        missing_image="img_missing.jpg",
    )

    issues = ParseIssueCollector()
    parsed = UltralyticsNDJSONParser(issues).parse(
        ndjson_path, dataset_type="ultralytics-ndjson"
    )

    assert [path.name for path in parsed.files] == [
        "img_0.jpg",
        "img_1.jpg",
        "img_2.jpg",
    ]
    assert parsed.splits is not None
    assert {
        name: [path.name for path in paths]
        for name, paths in parsed.splits.items()
    } == {"train": ["img_0.jpg"], "val": ["img_1.jpg"], "test": ["img_2.jpg"]}

    records = _collect_raw_records(parsed.records)
    assert [
        (Path(record["file"]).name, (record["annotation"] or {}).get("class"))
        for record in records
    ] == [
        ("img_0.jpg", "budgie"),
        ("img_1.jpg", None),
        ("img_2.jpg", "parrot"),
    ]
    assert [issue.parser_issue for issue in issues.messages] == [
        ParserIssue.MISSING_IMAGE
    ]


def test_ultralytics_ndjson_records_are_streamed(
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The manifest must not be materialized before the first record.

    Regression: the whole file was decoded into an in-memory list of records
    before anything was yielded, so a multi-gigabyte export — every ``segments``
    polygon list and ``pose`` array held alive at once — died of ``MemoryError``
    before a single record reached ``dataset.add``. Counting walks of the file
    is what distinguishes streaming from materializing: a materialized parser
    walks it once, up front, and never again.
    """
    ndjson_path = tempdir / "ndjson_streaming" / "dataset.ndjson"
    ndjson_path.parent.mkdir(parents=True)
    _write_ultralytics_ndjson(
        ndjson_path,
        ndjson_path.parent / "images",
        missing_image="img_missing.jpg",
    )

    walks: list[Path] = []
    original = UltralyticsNDJSONParser._iter_image_records

    def counting_iter_image_records(path: Path) -> Iterator[dict[str, Any]]:
        walks.append(path)
        yield from original(path)

    monkeypatch.setattr(
        UltralyticsNDJSONParser,
        "_iter_image_records",
        staticmethod(counting_iter_image_records),
    )

    parsed = UltralyticsNDJSONParser(ParseIssueCollector()).parse(
        ndjson_path, dataset_type="ultralytics-ndjson"
    )
    assert len(walks) == 1, "eager pass resolves image paths only"

    assert len(list(parsed.records)) == 3
    assert len(walks) == 2, "annotations are read on consumption, not up front"


def _write_solo_split(split_path: Path) -> None:
    """Write a one-sequence SOLO split with box, instance and semantic masks."""
    sequence_path = split_path / "sequence.0"
    sequence_path.mkdir(parents=True)

    cv2.imwrite(
        str(sequence_path / "step0.camera.jpg"),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )
    for mask_name, colour in (
        ("step0.camera.instance.png", (0, 0, 255)),
        ("step0.camera.semantic.png", (0, 255, 0)),
    ):
        mask = np.zeros((8, 8, 3), dtype=np.uint8)
        mask[:4, :4] = colour
        cv2.imwrite(str(sequence_path / mask_name), mask)

    prefix = "type.unity.com/unity.solo."
    box_type = f"{prefix}BoundingBox2DAnnotation"
    instance_type = f"{prefix}InstanceSegmentationAnnotation"
    semantic_type = f"{prefix}SemanticSegmentationAnnotation"

    (split_path / "annotation_definitions.json").write_text(
        json.dumps(
            {
                "annotationDefinitions": [
                    {
                        "@type": box_type,
                        "spec": [{"label_name": "budgie", "label_id": 1}],
                    }
                ]
            }
        )
    )
    (split_path / "metadata.json").write_text(
        json.dumps({"totalSequences": 1})
    )
    (split_path / "metric_definitions.json").write_text("{}")
    (split_path / "sensor_definitions.json").write_text("{}")
    (sequence_path / "step0.frame_data.json").write_text(
        json.dumps(
            {
                "step": 0,
                "captures": [
                    {
                        "filename": "step0.camera.jpg",
                        "dimension": [8, 8],
                        "annotations": [
                            {
                                "@type": box_type,
                                "values": [
                                    {
                                        "labelName": "budgie",
                                        "origin": [0, 0],
                                        "dimension": [4, 4],
                                        "instanceId": 1,
                                    }
                                ],
                            },
                            {
                                "@type": instance_type,
                                "filename": "step0.camera.instance.png",
                                "instances": [
                                    {
                                        "color": [255, 0, 0, 255],
                                        "instanceId": 1,
                                    }
                                ],
                            },
                            {
                                "@type": semantic_type,
                                "filename": "step0.camera.semantic.png",
                                "instances": [
                                    {
                                        "labelName": "budgie",
                                        "pixelValue": [0, 255, 0, 255],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        )
    )


def test_solo_file_enumeration_does_not_decode_masks(
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Listing SOLO images must not decode every mask a second time.

    Regression: ``files`` was built with ``_get_added_images(generator())`` and
    the record stream then ran the same generator again, so every semantic and
    instance mask PNG was read and every per-instance mask rebuilt twice —
    roughly doubling import time for the heaviest supported format. The
    enumeration pass now skips mask decoding, and the masks reaching the
    records still have full image dimensions.
    """
    split_path = tempdir / "solo" / "train"
    _write_solo_split(split_path)

    decoded: list[str] = []
    original_imread = cv2.imread

    def counting_imread(
        path,  # noqa: ANN001
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        decoded.append(str(path))
        return original_imread(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counting_imread)

    parsed = SOLOParser(ParseIssueCollector())._parse_split(split_path)
    assert [path.name for path in parsed.files] == ["step0.camera.jpg"]
    assert decoded == [], "enumerating files must not decode any mask"

    records = _collect_raw_records(parsed.records)
    assert len(decoded) == 2, "each mask is decoded exactly once"
    masks = [
        value["mask"]
        for record in records
        for value in record["annotation"].values()
        if isinstance(value, dict) and "mask" in value
    ]
    assert [mask.shape for mask in masks] == [(8, 8), (8, 8)]
