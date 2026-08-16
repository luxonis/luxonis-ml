import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from loguru import logger
from pydantic import SecretStr

from luxonis_ml.data import (
    BaseDataset,
    LuxonisLoader,
    LuxonisParser,
    ParserIssue,
)
from luxonis_ml.data.parsers import luxonis_parser
from luxonis_ml.data.parsers.base_parser import BaseParser, ParserOutput
from luxonis_ml.data.utils import get_task_type
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import environ

from .utils import create_image


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


def test_split_parser_creates_default_splits(dataset_name: str, tempdir: Path):
    class_dir = tempdir / "flat_cls"
    image_dir = class_dir / "class_a"
    image_dir.mkdir(parents=True)
    create_image(0, image_dir)

    dataset = LuxonisParser(
        str(class_dir),
        dataset_name=dataset_name,
        dataset_type=DatasetType.CLSDIR,
        delete_local=True,
    ).parse()
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert set(splits) == {"train", "val", "test"}
        assert sum(len(group_ids) for group_ids in splits.values()) == 1

        loader = LuxonisLoader(dataset)
        next(iter(loader))
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

    dataset = LuxonisParser(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type=DatasetType.FIFTYONECLS,
        delete_local=True,
    ).parse()
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

    parser = LuxonisParser(
        str(split_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    dataset = parser.parse()
    try:
        assert len(dataset) == 1

        issues = parser._get_parser_issue_messages()
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
        assert len(parser._get_parser_issue_messages()) == 3
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
                ParserIssue.NON_NUMERIC_ANNOTATION,
                self.reason,
                annotation_id=annotation_id,
            )
        return iter(()), {}, []


def test_skipped_annotation_warnings_are_capped():
    warning_count = BaseParser._SKIPPED_WARNING_LIMIT + 5
    parser = _WarningParser(warning_count)
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )

    try:
        parser.parse_split()
    finally:
        logger.remove(sink_id)

    assert len(parser._get_parser_issue_messages()) == warning_count
    assert (
        sum(message.startswith("Skipping annotation:") for message in messages)
        == BaseParser._SKIPPED_WARNING_LIMIT
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
    parser = _WarningParser(
        BaseParser._SKIPPED_WARNING_LIMIT + 5, full_warnings=True
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )

    try:
        parser.parse_split()
    finally:
        logger.remove(sink_id)

    assert len(parser._get_parser_issue_messages()) == (
        BaseParser._SKIPPED_WARNING_LIMIT + 5
    )
    assert (
        sum(message.startswith("Skipping annotation:") for message in messages)
        == BaseParser._SKIPPED_WARNING_LIMIT + 5
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


def test_parser_scopes_keypoint_metadata_to_the_task_of_its_class(
    dataset_name: str,
    tempdir: Path,
):
    """The parser gave every task the keypoints of the last class.

    `_parse_split` keys the parser output by source class name, but it
    called `set_keypoint_metadata` with no task. The last keypoint
    category thus overwrote the metadata of every task, including a task
    that holds no keypoints. Each task must keep the keypoint labels of
    its own class.
    """
    dataset_dir = tempdir / "coco_two_keypoint_classes"

    def annotations(image_name: str) -> str:
        return json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": image_name,
                        "width": 512,
                        "height": 512,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [128, 128, 256, 256],
                        "keypoints": [256, 200, 2, 230, 180, 2, 280, 180, 2],
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 2,
                        "bbox": [10, 10, 60, 60],
                        "keypoints": [20, 20, 2, 50, 50, 2],
                    },
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "person",
                        "keypoints": ["nose", "left_eye", "right_eye"],
                        "skeleton": [[1, 2], [1, 3]],
                    },
                    {
                        "id": 2,
                        "name": "hand",
                        "keypoints": ["thumb", "index"],
                        "skeleton": [[1, 2]],
                    },
                ],
            }
        )

    for split, index in [("train", 16), ("valid", 17)]:
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True)
        image = create_image(index, split_dir)
        (split_dir / "_annotations.coco.json").write_text(
            annotations(image.name)
        )

    dataset = LuxonisParser(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type=DatasetType.COCO,
        task_name={"person": "pose", "hand": "hands"},
        delete_local=True,
        save_dir=tempdir,
    ).parse()
    try:
        keypoint_metadata = dataset.get_keypoint_metadata()
        assert keypoint_metadata["pose"].labels == [
            "nose",
            "left_eye",
            "right_eye",
        ]
        assert keypoint_metadata["hands"].labels == ["thumb", "index"]
        assert dataset.get_n_keypoints() == {"pose": 3, "hands": 2}
    finally:
        dataset.delete_dataset(delete_local=True)


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


class _FakeExportResponse:
    """Stand-in for the export response of the Ultralytics API."""

    ok = True

    def json(self) -> dict[str, str]:
        return {"downloadUrl": "https://storage.example.com/export.ndjson"}


@pytest.fixture
def ultralytics_requests(
    monkeypatch: pytest.MonkeyPatch, tempdir: Path
) -> list[dict[str, Any]]:
    """Capture the requests that `_download_ultralytics_dataset` makes.

    The live parser test skips without an API key, so these mocked
    calls are the only coverage of the URL contract.
    """
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs) -> _FakeExportResponse:
        calls.append({"url": url, "params": kwargs.get("params")})
        return _FakeExportResponse()

    monkeypatch.setattr(
        environ, "ULTRALYTICS_API_KEY", SecretStr("ul_" + "0" * 40)
    )
    monkeypatch.setattr(luxonis_parser.requests, "get", fake_get)
    monkeypatch.setattr(
        luxonis_parser,
        "download_remote_file",
        lambda url, destination, **_: destination.write_text("{}"),
    )
    return calls


def test_ultralytics_url_uses_owner_and_name(
    ultralytics_requests: list[dict[str, Any]], tempdir: Path
):
    """The API dropped the lookup by username and slug.

    A `GET` on `/api/datasets` now answers 405, so the export must be
    addressed by owner and name.
    """
    destination, name = LuxonisParser._download_ultralytics_dataset(
        "ultralytics://acme-vision/datasets/warehouse", tempdir
    )

    assert len(ultralytics_requests) == 1
    assert ultralytics_requests[0]["url"] == (
        "https://platform.ultralytics.com/api"
        "/datasets/acme-vision/warehouse/export"
    )
    assert ultralytics_requests[0]["params"] is None
    assert name == "warehouse"
    assert destination == tempdir / "warehouse.ndjson"


def test_ultralytics_url_escapes_both_segments(
    ultralytics_requests: list[dict[str, Any]], tempdir: Path
):
    LuxonisParser._download_ultralytics_dataset(
        "ultralytics://a b/datasets/c d", tempdir
    )

    assert ultralytics_requests[0]["url"].endswith(
        "/datasets/a%20b/c%20d/export"
    )


def test_ultralytics_version_selects_an_export(
    ultralytics_requests: list[dict[str, Any]], tempdir: Path
):
    destination, _ = LuxonisParser._download_ultralytics_dataset(
        "ultralytics://acme-vision/datasets/warehouse?v=3", tempdir
    )

    assert ultralytics_requests[0]["params"] == {"v": 3}
    assert destination == tempdir / "warehouse.v3.ndjson"
