"""Behaviour shared by all parsers: file discovery, split helpers,
annotation-issue reporting and count-based split maths.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import (
    example,
    given,
    settings,
)
from hypothesis import (
    strategies as st,
)
from loguru import logger

from luxonis_ml.data import (
    LuxonisDataset,
    ParseIssueCollector,
    ParserIssue,
    ParserPlugin,
)
from luxonis_ml.data.datasets.annotation import (
    DatasetRecord,
)
from luxonis_ml.data.datasets.base_dataset import _record_files
from luxonis_ml.data.parsers import (
    ClassificationDirectoryParser,
)
from luxonis_ml.data.parsers.parser_plugin import (
    apply_counts_to_pool,
    apply_counts_to_splits,
)
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _write_yolov8_split,
)


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


def test_parse_issue_collector_deduplicates_and_summarizes():
    collector = ParseIssueCollector(warning_limit=1)
    # Asserted through a dedicated sink, like the capped-warnings test:
    # where the global logging setup routes its output - stdout, stderr,
    # a rich handler - is not this test's concern.
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        collector.warn(
            ParserIssue.MISSING_IMAGE,
            "missing",
            source="annotations.json",
            image="missing.jpg",
            annotation_id=1,
        )
        collector.warn(
            ParserIssue.MISSING_IMAGE,
            "missing",
            source="annotations.json",
            image="missing.jpg",
            annotation_id=1,
        )
        collector.warn(ParserIssue.MISSING_IMAGE, "other")
        collector.log_summary()
    finally:
        logger.remove(sink_id)

    assert len(collector.messages) == 2
    output = "\n".join(messages)
    assert "annotation_id=1" in output
    assert "Skipped logging 1 additional warnings" in output
    assert "Skipped annotations: missing (1 records)" in output
    assert "Skipped annotations: other (1 records)" in output

    reported = collector.messages
    reported.clear()
    assert len(collector.messages) == 2

    full = ParseIssueCollector(full_warnings=True, warning_limit=0)
    full.warn(ParserIssue.MISSING_IMAGE, "visible")
    full.log_summary()
    assert len(full.messages) == 1


def test_parser_plugin_file_helpers(tmp_path: Path):
    image = _image(tmp_path / "image.jpg")
    second = _image(tmp_path / "second.png")
    (tmp_path / "notes.txt").write_text("not an image")

    assert set(ParserPlugin._list_images(tmp_path)) == {image, second}

    record = DatasetRecord(files={"image": image})
    assert list(_record_files(record)) == [image.absolute()]
    assert list(_record_files({"file": image})) == [image]
    assert list(
        _record_files({"files": {"image": image, "depth": second}})
    ) == [image, second]
    assert list(_record_files({})) == []


def _dicts(
    tagged: list[tuple[str | None, Any]],
) -> list[tuple[str | None, dict[str, Any]]]:
    """Narrow split-tagged parser output, which plugins emit as dicts."""
    for _, record in tagged:
        assert isinstance(record, dict)
    return cast("list[tuple[str | None, dict[str, Any]]]", tagged)


def test_split_plugin_helpers_and_errors(tmp_path: Path):
    parser = _plugin(ClassificationDirectoryParser)
    source_file = tmp_path / "source.txt"
    source_file.write_text("x")

    assert type(parser).detect(source_file) is None
    assert parser._canonicalize_split_name("valid") == "val"
    assert parser._canonicalize_split_name("validation") == "val"
    assert parser._canonicalize_split_name("train") == "train"
    assert type(parser).detect(tmp_path / "missing") is None

    train_image = _image(tmp_path / "dataset" / "train" / "bird" / "a.jpg")
    val_image = _image(tmp_path / "dataset" / "valid" / "cat" / "b.png")
    layout = type(parser).detect(tmp_path / "dataset")
    assert layout is not None
    assert layout.split_names == ["train", "val"]

    parsed = parser.parse(tmp_path / "dataset", layout)
    tagged = list(parsed.records)
    assert [split for split, _ in tagged] == ["train", "val"]
    assert [Path(record["file"]) for _, record in _dicts(tagged)] == [
        train_image.resolve(),
        val_image.resolve(),
    ]
    assert {record["annotation"]["class"] for _, record in _dicts(tagged)} == {
        "bird",
        "cat",
    }


def test_split_plugin_streams_every_split_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Records carry their split instead of a file list being published.

    Regression: the base class used to parse each split into its own
    result and merge the file lists afterwards, which is what forced
    every parser to walk its source a second time.
    """
    parser = _plugin(ClassificationDirectoryParser)
    _image(tmp_path / "dataset" / "train" / "bird" / "a.jpg")
    _image(tmp_path / "dataset" / "test" / "cat" / "b.png")
    layout = type(parser).detect(tmp_path / "dataset")
    assert layout is not None

    listings = 0
    original = ParserPlugin._list_images

    def counting_list_images(image_dir: Path) -> list[Path]:
        nonlocal listings
        listings += 1
        return original(image_dir)

    parsed = parser.parse(tmp_path / "dataset", layout)
    assert isinstance(parsed.records, Iterator)

    # `monkeypatch` rather than a manual try/finally: it reverts at
    # teardown however the test exits, so a failed assertion cannot
    # leave the patched staticmethod on the shared base class.
    monkeypatch.setattr(
        ParserPlugin, "_list_images", staticmethod(counting_list_images)
    )
    tagged = list(parsed.records)

    assert [split for split, _ in tagged] == ["train", "test"]
    # One listing per split, not two: nothing walks the source again to
    # find out which files the records named.
    assert listings == 2


@settings(max_examples=50, deadline=None)
@example(image_ids=[0, 1, 2], counts=(1, 1, 1))
@example(image_ids=[0, 1], counts=(3, 2, 1))
@given(
    image_ids=st.lists(
        st.integers(min_value=0, max_value=1000),
        min_size=0,
        max_size=20,
        unique=True,
    ),
    counts=st.tuples(
        st.integers(min_value=0, max_value=20),
        st.integers(min_value=0, max_value=20),
        st.integers(min_value=0, max_value=20),
    ),
)
def test_apply_counts_to_pool_property(
    image_ids: list[int], counts: tuple[int, int, int]
):
    images = [Path(f"{image_id}.jpg") for image_id in image_ids]
    ratios: dict[str, int] = dict(
        zip(("train", "val", "test"), counts, strict=True)
    )
    sampled = apply_counts_to_pool(images, ratios)
    selected = [image for values in sampled.values() for image in values]

    assert set(sampled) == {"train", "val", "test"}
    assert len(selected) == min(sum(counts), len(images))
    assert len(selected) == len(set(selected))
    assert set(selected) <= set(images)


@settings(max_examples=50, deadline=None)
@example(counts=(0, 1, 4))
@given(
    counts=st.tuples(
        st.integers(min_value=0, max_value=6),
        st.integers(min_value=0, max_value=6),
        st.integers(min_value=0, max_value=6),
    )
)
def test_apply_counts_to_splits_property(counts: tuple[int, int, int]):
    original = {
        "train": [Path(f"train-{index}.jpg") for index in range(3)],
        "val": [Path(f"val-{index}.jpg") for index in range(2)],
    }
    ratios: dict[str, int] = dict(
        zip(("train", "val", "test"), counts, strict=True)
    )
    sampled = apply_counts_to_splits(original, ratios)

    for split_name, requested in ratios.items():
        available = original.get(split_name, [])
        assert len(sampled[split_name]) == min(requested, len(available))
        assert set(sampled[split_name]) <= set(available)
