import json
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import pytest
import yaml
from defusedxml.ElementTree import fromstring
from hypothesis import example, given, settings
from hypothesis import strategies as st
from PIL import Image
from pydantic import SecretStr

import luxonis_ml.data.parsers.source as parser_source
from luxonis_ml.data.datasets import LuxonisDataset
from luxonis_ml.data.datasets.annotation import DatasetRecord
from luxonis_ml.data.parsers import (
    ClassificationDirectoryParser,
    COCOParser,
    CreateMLParser,
    DarknetParser,
    FiftyOneClassificationParser,
    LuxonisParser,
    NativeParser,
    ParsedDataset,
    ParseIssueCollector,
    ParserPlugin,
    SegmentationMaskDirectoryParser,
    SOLOParser,
    TensorflowCSVParser,
    UltralyticsNDJSONParser,
    VOCParser,
    YoloV4Parser,
    YoloV6Parser,
    YOLOv8Parser,
)
from luxonis_ml.data.parsers.coco_parser import clean_annotations
from luxonis_ml.data.parsers.fiftyone_classification_parser import (
    clean_imagenet_annotations,
)
from luxonis_ml.data.parsers.parser_plugin import (
    PARSERS_REGISTRY,
    SplitParserPlugin,
    _record_files,
    apply_counts_to_pool,
    apply_counts_to_splits,
    combine_split_outputs,
    get_parser_plugin,
    register_parser_plugin,
)
from luxonis_ml.data.parsers.yolov8_parser import Format
from luxonis_ml.data.utils.enums import COCOFormat, ParserIssue

T = TypeVar("T", bound=ParserPlugin)


def _image(
    path: Path,
    *,
    size: tuple[int, int] = (20, 10),
    value: int = 0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)
    return path


def _plugin(parser_type: type[T]) -> T:
    return parser_type(ParseIssueCollector())


def _records(parsed: ParsedDataset) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], list(parsed.records))


def test_parse_issue_collector_deduplicates_and_summarizes(
    capsys: pytest.CaptureFixture[str],
):
    collector = ParseIssueCollector(warning_limit=1)
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

    assert len(collector.messages) == 2
    output = capsys.readouterr().out
    assert "annotation_id=1" in output
    assert "Skipped logging 1 additional warnings" in output
    assert "Skipped annotations: missing (1 records)" in output
    assert "Skipped annotations: other (1 records)" in output

    messages = collector.messages
    messages.clear()
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
    assert ParserPlugin._compare_stem_files(
        [tmp_path / "a.jpg"], [tmp_path / "a.xml"]
    )
    assert not ParserPlugin._compare_stem_files([], [])
    assert not ParserPlugin._compare_stem_files(
        [tmp_path / "a.jpg"], [tmp_path / "b.xml"]
    )

    record = DatasetRecord(files={"image": image})
    assert list(_record_files(record)) == [image.absolute()]
    assert list(_record_files({"file": image})) == [image]
    assert list(
        _record_files({"files": {"image": image, "depth": second}})
    ) == [image, second]
    assert list(_record_files({})) == []

    added = ParserPlugin._get_added_images(
        iter(
            [
                {"file": image},
                {"files": {"image": image, "depth": second}},
            ]
        )
    )
    assert added == [image, second]


def test_split_plugin_helpers_and_errors(tmp_path: Path):
    parser = _plugin(ClassificationDirectoryParser)
    source_file = tmp_path / "source.txt"
    source_file.write_text("x")

    assert parser.discover_splits(source_file) == {}
    assert not parser.supports(source_file)
    assert parser._canonicalize_split_name("valid") == "val"
    assert parser._canonicalize_split_name("validation") == "val"
    assert parser._canonicalize_split_name("train") == "train"
    with pytest.raises(ValueError, match="expected format"):
        parser.parse(tmp_path / "missing", dataset_type="clsdir")

    train_image = _image(tmp_path / "dataset" / "train" / "bird" / "a.jpg")
    val_image = _image(tmp_path / "dataset" / "valid" / "cat" / "b.png")
    parsed = parser.parse(tmp_path / "dataset", dataset_type="clsdir")
    assert parsed.files == [train_image.resolve(), val_image.resolve()]
    assert parsed.splits == {
        "train": [train_image.resolve()],
        "val": [val_image.resolve()],
        "test": [],
    }
    assert {record["annotation"]["class"] for record in _records(parsed)} == {
        "bird",
        "cat",
    }


def test_combine_split_outputs_merges_files_and_skeletons(tmp_path: Path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    parsed = combine_split_outputs(
        {
            "train": ParsedDataset(
                iter([{"file": first}]),
                {"bird": {"labels": ["head"]}},
                [first],
            ),
            "test": ParsedDataset(
                iter([{"file": second}]),
                {"cat": {"labels": ["tail"]}},
                [first, second],
            ),
        }
    )

    assert parsed.files == [first, second]
    assert parsed.skeletons == {
        "bird": {"labels": ["head"]},
        "cat": {"labels": ["tail"]},
    }
    assert parsed.splits == {
        "train": [first],
        "val": [],
        "test": [first, second],
    }
    assert len(_records(parsed)) == 2


def test_parser_registration_forms_and_validation():
    class MissingTypes(ParserPlugin):
        @classmethod
        def supports(cls, source: Path) -> bool:
            return False

        def parse(
            self, source: Path, *, dataset_type: str, **kwargs: Any
        ) -> ParsedDataset:
            return ParsedDataset(iter(()), {}, [])

    with pytest.raises(ValueError, match="declare `dataset_types`"):
        register_parser_plugin(MissingTypes)

    @register_parser_plugin(force=True)
    class DecoratedParser(MissingTypes):
        dataset_types = ("synthetic-decorated",)

    assert PARSERS_REGISTRY.get("synthetic-decorated") is DecoratedParser


class _SyntheticSplitParser(SplitParserPlugin):
    dataset_types = ("synthetic-split",)
    recognized = False
    splits: dict[str, dict[str, Any]] = {}

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        return (
            {"source": split_path}
            if _SyntheticSplitParser.recognized
            else None
        )

    @classmethod
    def discover_splits(cls, source: Path) -> dict[str, dict[str, Any]]:
        return cls.splits

    def _parse_split(self, **kwargs: Any) -> ParsedDataset:
        return ParsedDataset(iter(()), {}, [])


def test_get_parser_plugin_resolution_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class NeverParser(_SyntheticSplitParser):
        dataset_types = ("never",)

        @classmethod
        def supports(cls, source: Path) -> bool:
            return False

    class FirstParser(NeverParser):
        dataset_types = ("first",)

        @classmethod
        def supports(cls, source: Path) -> bool:
            return True

    class SecondParser(FirstParser):
        dataset_types = ("second",)

    monkeypatch.setattr(PARSERS_REGISTRY, "get", lambda name: NeverParser)
    with pytest.raises(ValueError, match="never parser"):
        get_parser_plugin(tmp_path, "never")

    monkeypatch.setattr(PARSERS_REGISTRY, "get", lambda name: FirstParser)
    assert get_parser_plugin(tmp_path, "first") == (FirstParser, "first")

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [NeverParser])
    with pytest.raises(ValueError, match="any registered parser"):
        get_parser_plugin(tmp_path, None)

    monkeypatch.setattr(
        PARSERS_REGISTRY, "values", lambda: [FirstParser, SecondParser]
    )
    with pytest.raises(ValueError, match="multiple parsers: first, second"):
        get_parser_plugin(tmp_path, None)

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [FirstParser])
    assert get_parser_plugin(tmp_path, None) == (FirstParser, "first")


def test_get_parser_plugin_resolves_by_split_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Multiple matches are broken by how many splits each one recognizes.

    YOLOv6 and Ultralytics YOLOv8 layouts differ only in split naming
    (``images/valid`` against ``images/val``), so both parsers report support
    for either tree — but only the right one recognizes every split present.
    Equal coverage stays genuinely ambiguous and must be reported, since
    guessing would silently parse the source with the wrong parser.
    """

    class SyntheticYoloV6(_SyntheticSplitParser):
        dataset_types = ("yolov6",)
        splits: dict[str, dict[str, Any]] = {}

        @classmethod
        def supports(cls, source: Path) -> bool:
            return True

    class SyntheticYoloV8(SyntheticYoloV6):
        dataset_types = ("yolov8",)
        splits: dict[str, dict[str, Any]] = {}

    monkeypatch.setattr(
        PARSERS_REGISTRY,
        "values",
        lambda: [SyntheticYoloV6, SyntheticYoloV8],
    )

    SyntheticYoloV6.splits = {"train": {}, "valid": {}, "test": {}}
    SyntheticYoloV8.splits = {"train": {}}
    assert get_parser_plugin(tmp_path, None) == (SyntheticYoloV6, "yolov6")

    SyntheticYoloV6.splits = {"train": {}}
    SyntheticYoloV8.splits = {"train": {}, "val": {}}
    assert get_parser_plugin(tmp_path, None) == (SyntheticYoloV8, "yolov8")

    SyntheticYoloV6.splits = {"test": {}}
    SyntheticYoloV8.splits = {"test": {}}
    with pytest.raises(ValueError, match="multiple parsers: yolov6, yolov8"):
        get_parser_plugin(tmp_path, None)

    # No split at all is no evidence either way, so it stays ambiguous too.
    SyntheticYoloV6.splits = {}
    SyntheticYoloV8.splits = {}
    with pytest.raises(ValueError, match="multiple parsers: yolov6, yolov8"):
        get_parser_plugin(tmp_path, None)


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


def test_classification_directory_validation(tmp_path: Path):
    parser = _plugin(ClassificationDirectoryParser)
    assert parser.validate_split(tmp_path / "missing") is None

    reserved = tmp_path / "reserved"
    for name in (
        "train",
        "valid",
        "test",
        "val",
        "validation",
        "images",
        "labels",
        "data",
        "raw",
        "masks",
    ):
        (reserved / name).mkdir(parents=True)
    assert parser.validate_split(reserved) is None

    class_dir = tmp_path / "classes"
    _image(class_dir / "bird" / "bird.jpg")
    (class_dir / "unexpected.txt").write_text("x")
    assert parser.validate_split(class_dir) is None
    (class_dir / "unexpected.txt").unlink()
    (class_dir / "info.json").write_text("{}")
    assert parser.validate_split(class_dir) == {"class_dir": class_dir}


def test_create_ml_parser_with_missing_and_valid_images(tmp_path: Path):
    parser = _plugin(CreateMLParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    annotations = split / "_annotations.createml.json"
    annotations.write_text("[]")
    assert parser.validate_split(split) is None

    image = _image(split / "image.jpg")
    annotations.write_text(
        json.dumps(
            [
                {
                    "image": image.name,
                    "annotations": [
                        {
                            "label": "bird",
                            "coordinates": {
                                "x": 10,
                                "y": 5,
                                "width": 10,
                                "height": 4,
                            },
                        }
                    ],
                },
                {"image": "missing.jpg", "annotations": []},
            ]
        )
    )
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    parsed = parser._parse_split(**kwargs)
    records = _records(parsed)
    assert parsed.files == [image.resolve()]
    assert records[0]["annotation"] == {
        "class": "bird",
        "boundingbox": {"x": 0.25, "y": 0.3, "w": 0.5, "h": 0.4},
    }
    assert len(parser._issues.messages) == 1


def test_darknet_parser_with_labeled_and_unlabeled_images(tmp_path: Path):
    parser = _plugin(DarknetParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    classes = split / "_darknet.labels"
    classes.write_text("bird\n")
    assert parser.validate_split(split) is None

    labeled = _image(split / "labeled.jpg")
    unlabeled = _image(split / "unlabeled.jpg")
    labeled.with_suffix(".txt").write_text("0 0.5 0.5 0.4 0.2\n")
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._parse_split(**kwargs))

    assert {Path(record["file"]).name for record in records} == {
        labeled.name,
        unlabeled.name,
    }
    labeled_record = next(
        record
        for record in records
        if Path(record["file"]).name == labeled.name
    )
    assert labeled_record["annotation"]["boundingbox"] == {
        "x": 0.3,
        "y": 0.4,
        "w": 0.4,
        "h": 0.2,
    }
    assert (
        next(
            record
            for record in records
            if Path(record["file"]).name == unlabeled.name
        )["annotation"]
        is None
    )


def test_segmentation_mask_directory_parser(tmp_path: Path):
    parser = _plugin(SegmentationMaskDirectoryParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    classes = split / "_classes.csv"
    classes.write_text("Pixel Value, Class\n0,background\n1,bird\n")
    orphan_mask = split / "orphan_mask.png"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(orphan_mask)
    assert parser.validate_split(split) is None
    orphan_mask.unlink()

    image = _image(split / "sample.jpg", size=(2, 2))
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(split / "sample_mask.png")
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._parse_split(**kwargs))

    assert len(records) == 2
    assert {record["annotation"]["class"] for record in records} == {
        "background",
        "bird",
    }
    assert {Path(record["file"]) for record in records} == {image.resolve()}

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "_classes.csv").write_text("Pixel Value, Class\n0,bird\n")
    _image(broken / "image.jpg")
    (broken / "image_mask.png").write_text("not an image")
    with pytest.raises(ValueError, match="Failed to read mask"):
        parser._parse_split(broken, broken, broken / "_classes.csv")


def test_tensorflow_csv_parser(tmp_path: Path):
    parser = _plugin(TensorflowCSVParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    _image(split / "annotated.jpg", size=(20, 10))
    assert parser.validate_split(split) is None

    unannotated = _image(split / "unannotated.jpg")
    annotations = split / "_annotations.csv"
    annotations.write_text(
        "filename,width,height,class,xmin,ymin,xmax,ymax\n"
        "annotated.jpg,20,10,bird,2,1,12,6\n"
        ",20,10,bird,0,0,1,1\n"
    )
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._parse_split(**kwargs))

    assert len(records) == 2
    annotated = next(record for record in records if record["annotation"])
    assert annotated["annotation"]["boundingbox"] == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.5,
        "h": 0.5,
    }
    assert (
        next(
            record
            for record in records
            if Path(record["file"]).name == unannotated.name
        )["annotation"]
        is None
    )


def _voc_xml(
    filename: str,
    *,
    with_bbox: bool = True,
) -> str:
    bbox = (
        "<bndbox><xmin>2</xmin><ymin>1</ymin>"
        "<xmax>12</xmax><ymax>6</ymax></bndbox>"
        if with_bbox
        else ""
    )
    return (
        "<annotation>"
        f"<filename>{filename}</filename>"
        "<size><width>20</width><height>10</height></size>"
        f"<object><name>bird</name>{bbox}</object>"
        "</annotation>"
    )


def test_voc_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parser = _plugin(VOCParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    _image(split / "mismatch.jpg")
    (split / "other.xml").write_text(_voc_xml("other.jpg"))
    assert parser.validate_split(split) is None

    (split / "mismatch.jpg").unlink()
    (split / "other.xml").unlink()
    bbox_image = _image(split / "bbox.jpg", size=(20, 10))
    empty_image = _image(split / "empty.jpg", size=(20, 10))
    (split / "bbox.xml").write_text(_voc_xml(bbox_image.name))
    (split / "empty.xml").write_text(
        _voc_xml(empty_image.name, with_bbox=False)
    )
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._parse_split(**kwargs))

    assert len(records) == 2
    bbox_record = next(record for record in records if record["annotation"])
    assert bbox_record["annotation"]["boundingbox"] == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.5,
        "h": 0.5,
    }
    assert (
        next(
            record
            for record in records
            if Path(record["file"]).name == empty_image.name
        )["annotation"]
        is None
    )

    missing_xml = split / "missing.xml"
    missing_xml.write_text(_voc_xml("missing.jpg"))
    _records(parser._parse_split(split, split))
    assert parser._issues.messages
    missing_xml.unlink()

    class EmptyTree:
        @staticmethod
        def getroot() -> None:
            return None

    monkeypatch.setattr(
        "luxonis_ml.data.parsers.voc_parser.parse",
        lambda path: EmptyTree(),
    )
    with pytest.raises(ValueError, match="Could not parse"):
        parser._parse_split(split, split)

    with pytest.raises(ValueError, match="Could not find missing"):
        parser._xml_find(fromstring("<root />"), "missing")


def test_yolov4_parser(tmp_path: Path):
    parser = _plugin(YoloV4Parser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    (split / "_annotations.txt").write_text("")
    assert parser.validate_split(split) is None
    (split / "_classes.txt").write_text("bird\n")

    bbox = _image(split / "bbox.jpg", size=(20, 10))
    empty = _image(split / "empty.jpg")
    unlisted = _image(split / "unlisted.jpg")
    (split / "_annotations.txt").write_text(
        f"{bbox.name} 2,1,12,6,0\n{empty.name}\nmissing.jpg 0,0,1,1,0\n"
    )
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._parse_split(**kwargs))

    assert {Path(record["file"]).name for record in records} == {
        bbox.name,
        empty.name,
        unlisted.name,
    }
    assert next(
        record for record in records if Path(record["file"]).name == bbox.name
    )["annotation"]["boundingbox"] == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.5,
        "h": 0.5,
    }
    assert len(parser._issues.messages) == 1


def test_yolov6_parser(tmp_path: Path):
    parser = _plugin(YoloV6Parser)
    root = tmp_path / "dataset"
    assert parser.discover_splits(root) == {}

    image_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True)
    assert parser.validate_split(image_dir) is None
    labels_dir.mkdir(parents=True)
    assert parser.validate_split(image_dir) is None
    labeled = _image(image_dir / "labeled.jpg")
    assert parser.validate_split(image_dir) is None
    (root / "data.yaml").write_text("names: [bird]\n")
    unlabeled = _image(image_dir / "unlabeled.jpg")
    (labels_dir / "labeled.txt").write_text("0 0.5 0.5 0.4 0.2\n")

    parsed = parser.parse(root, dataset_type="yolov6")
    records = _records(parsed)
    assert parsed.splits == {
        "train": [labeled, unlabeled],
        "val": [],
        "test": [],
    }
    assert (
        next(
            record
            for record in records
            if Path(record["file"]).name == labeled.name
        )["annotation"]["class"]
        == "bird"
    )
    assert (
        next(
            record
            for record in records
            if Path(record["file"]).name == unlabeled.name
        )["annotation"]
        is None
    )


def test_native_parser_resolves_files_and_masks(tmp_path: Path):
    parser = _plugin(NativeParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    image = _image(split / "image.jpg")
    depth = split / "depth.bin"
    depth.write_bytes(b"depth")
    mask = split / "mask.png"
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(mask)
    annotations = split / "annotations.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "file": image.name,
                    "annotation": {
                        "segmentation": {"mask": mask.name},
                        "instance_segmentation": {"mask": 123},
                    },
                },
                {
                    "files": {
                        "image": image.name,
                        "depth": depth.name,
                    },
                    "annotation": None,
                },
                {"annotation": None},
            ]
        )
    )

    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = _records(parser._parse_split(**kwargs))
    assert records[0]["file"] == image.resolve()
    assert records[0]["annotation"]["segmentation"]["mask"] == mask.resolve()
    assert records[0]["annotation"]["instance_segmentation"]["mask"] == 123
    assert records[1]["files"] == {
        "image": image.resolve(),
        "depth": depth.resolve(),
    }


def test_fiftyone_classification_validation_and_missing_stem(tmp_path: Path):
    parser = _plugin(FiftyOneClassificationParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    (split / "labels.json").write_text("{}")
    (split / "data").write_text("not a directory")
    assert parser.validate_split(split) is None
    (split / "data").unlink()
    (split / "data").mkdir()
    assert parser.validate_split(split) is None
    (split / "labels.json").write_text("{")
    assert parser.validate_split(split) is None

    image = _image(split / "data" / "present.jpg")
    (split / "labels.json").write_text(
        json.dumps(
            {
                "classes": ["bird"],
                "labels": {"present": 0, "missing": 0},
            }
        )
    )
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    parsed = parser._parse_split(**kwargs)
    assert _records(parsed) == [
        {"file": image, "annotation": {"class": "bird"}}
    ]
    assert len(parser._issues.messages) == 1

    flat = tmp_path / "flat"
    flat_image = _image(flat / "data" / "flat.jpg")
    (flat / "labels.json").write_text(
        json.dumps({"classes": ["bird"], "labels": {"flat": 0}})
    )
    assert _records(parser._parse_split(flat)) == [
        {"file": flat_image, "annotation": {"class": "bird"}}
    ]


def test_clean_imagenet_annotations(tmp_path: Path):
    untouched = tmp_path / "untouched.json"
    untouched.write_text(
        json.dumps({"classes": ["bird"], "labels": {"image": 0}})
    )
    assert clean_imagenet_annotations(untouched) == untouched

    labels_path = tmp_path / "labels.json"
    classes = [f"class-{index}" for index in range(640)]
    classes[10] = "crane"
    classes[20] = "maillot"
    classes[21] = "maillot"
    labels_path.write_text(
        json.dumps(
            {
                "classes": classes,
                "labels": {"006742": 517, "031933": 639},
            }
        )
    )
    cleaned = clean_imagenet_annotations(labels_path)
    cleaned_data = json.loads(cleaned.read_text())
    assert cleaned.name == "labels_fixed.json"
    assert cleaned_data["classes"][10] == "crane bird"
    assert cleaned_data["classes"][21] == "maillot swim suit"
    assert cleaned_data["labels"] == {"006742": 134, "031933": 638}


def _coco_data(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "images": images,
        "annotations": annotations or [],
        "categories": categories or [],
    }


def _write_coco_split(
    split: Path,
    *,
    roboflow: bool,
    data: dict[str, Any],
) -> tuple[Path, Path]:
    split.mkdir(parents=True, exist_ok=True)
    image_dir = split if roboflow else split / "data"
    image_dir.mkdir(exist_ok=True)
    annotation_path = split / (
        "_annotations.coco.json" if roboflow else "labels.json"
    )
    annotation_path.write_text(json.dumps(data))
    return image_dir, annotation_path


def test_coco_format_detection_and_validation(tmp_path: Path):
    parser = _plugin(COCOParser)
    missing = tmp_path / "missing"
    assert parser._detect_dataset_dir_format(missing) == (None, [])
    assert parser.validate_split(missing) is None

    native = tmp_path / "native"
    (native / "val").mkdir(parents=True)
    assert parser._detect_dataset_dir_format(native) == (None, [])

    fiftyone = tmp_path / "fiftyone"
    (fiftyone / "validation").mkdir(parents=True)
    assert parser._detect_dataset_dir_format(fiftyone) == (
        COCOFormat.FIFTYONE,
        ["validation"],
    )

    both_validation_names = tmp_path / "both-validation-names"
    (both_validation_names / "validation").mkdir(parents=True)
    (both_validation_names / "valid").mkdir()
    assert parser._detect_dataset_dir_format(both_validation_names) == (
        COCOFormat.FIFTYONE,
        ["validation"],
    )

    roboflow = tmp_path / "roboflow"
    (roboflow / "valid").mkdir(parents=True)
    assert parser._detect_dataset_dir_format(roboflow) == (
        COCOFormat.ROBOFLOW,
        ["valid"],
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    assert parser._detect_dataset_dir_format(empty) == (None, [])
    assert parser.discover_splits(empty) == {}

    train_only = tmp_path / "train-only"
    image_dir, annotation_path = _write_coco_split(
        train_only / "train",
        roboflow=True,
        data=_coco_data([], categories=[]),
    )
    assert parser._detect_dataset_dir_format(train_only) == (
        COCOFormat.ROBOFLOW,
        ["train"],
    )
    assert parser.validate_split(train_only / "train") == {
        "image_dir": image_dir,
        "annotation_path": annotation_path,
    }

    fiftyone_train = tmp_path / "fiftyone-train"
    image_dir, annotation_path = _write_coco_split(
        fiftyone_train / "train",
        roboflow=False,
        data=_coco_data([], categories=[]),
    )
    assert parser._detect_dataset_dir_format(fiftyone_train) == (
        COCOFormat.FIFTYONE,
        ["train"],
    )
    assert parser.validate_split(fiftyone_train / "train") == {
        "image_dir": image_dir,
        "annotation_path": annotation_path,
    }

    no_json = tmp_path / "no-json"
    no_json.mkdir()
    assert parser.validate_split(no_json) is None
    invalid_roboflow = tmp_path / "invalid-roboflow"
    invalid_roboflow.mkdir()
    (invalid_roboflow / "_annotations.coco.json").write_text("{}")
    assert parser.validate_split(invalid_roboflow) is None
    invalid_fiftyone = tmp_path / "invalid-fiftyone"
    invalid_fiftyone.mkdir()
    (invalid_fiftyone / "labels.json").write_text("{}")
    assert parser.validate_split(invalid_fiftyone) is None

    too_many_dirs = tmp_path / "too-many-dirs"
    too_many_dirs.mkdir()
    (too_many_dirs / "labels.json").write_text(
        json.dumps(_coco_data([], categories=[]))
    )
    (too_many_dirs / "one").mkdir()
    (too_many_dirs / "two").mkdir()
    assert parser.validate_split(too_many_dirs) is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("[]", False),
        ("{}", False),
        ('{"images": []}', False),
        ('{"images": [], "categories": []}', True),
        ('{"images": [], "info": {"categories": []}}', True),
        ("{", False),
    ],
)
def test_coco_json_detection(tmp_path: Path, content: str, expected: bool):
    path = tmp_path / "labels.json"
    path.write_text(content)
    assert COCOParser._is_coco_json(path) is expected
    if content == "{":
        path.unlink()
        assert not COCOParser._is_coco_json(path)


def test_coco_split_finalization_and_resolution(tmp_path: Path):
    parser = _plugin(COCOParser)
    with pytest.raises(ValueError, match="expected format"):
        parser._resolve_dir_format_and_keypoint_paths(
            tmp_path, use_keypoint_ann=False, keypoint_ann_paths=None
        )

    roboflow = tmp_path / "roboflow"
    (roboflow / "valid").mkdir(parents=True)
    assert parser._resolve_dir_format_and_keypoint_paths(
        roboflow, use_keypoint_ann=True, keypoint_ann_paths={"test": "x"}
    ) == (COCOFormat.ROBOFLOW, ["valid"], {"test": "x"})

    fiftyone = tmp_path / "fiftyone"
    (fiftyone / "validation").mkdir(parents=True)
    _, _, keypoint_paths = parser._resolve_dir_format_and_keypoint_paths(
        fiftyone, use_keypoint_ann=True, keypoint_ann_paths=None
    )
    assert keypoint_paths == {
        "train": "raw/person_keypoints_train2017.json",
        "val": "raw/person_keypoints_val2017.json",
        "test": "raw/person_keypoints_test2017.json",
    }

    images = [tmp_path / f"{index}.jpg" for index in range(3)]
    assert parser._finalize_split_definitions(
        {"val": images, "test": []}, split_val_to_test=True
    ) == {"val": images[:2], "test": images[2:]}
    assert parser._finalize_split_definitions(
        {"val": images, "test": []}, split_val_to_test=False
    ) == {"val": images, "test": []}
    assert parser._finalize_split_definitions(
        {"val": images, "test": [images[0]]}, split_val_to_test=True
    ) == {"val": images, "test": [images[0]]}


def test_coco_parser_all_annotation_types(tmp_path: Path):
    parser = _plugin(COCOParser)
    image_dir = tmp_path / "images"
    image = _image(image_dir / "image.jpg", size=(20, 10))
    unlabeled = _image(image_dir / "unlabeled.jpg", size=(20, 10))
    annotation_path = tmp_path / "labels.json"
    annotation_path.write_text(
        json.dumps(
            _coco_data(
                [
                    {
                        "id": 1,
                        "file_name": image.name,
                        "width": 20,
                        "height": 10,
                    },
                    {
                        "id": 2,
                        "file_name": unlabeled.name,
                        "width": 20,
                        "height": 10,
                    },
                    {
                        "id": 3,
                        "file_name": "missing.jpg",
                        "width": 20,
                        "height": 10,
                    },
                ],
                [
                    {
                        "id": 0,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "iscrowd": 1,
                    },
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [2, 1, 10, 5],
                        "segmentation": [[2, 1, 12, 1, 12, 6, 2, 6]],
                    },
                    {
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 20, 10],
                        "segmentation": {
                            "size": [10, 20],
                            "counts": "encoded",
                        },
                        "keypoints": [-1, 20, 2, 10, 5, 1],
                    },
                    {
                        "id": 3,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": ["bad", 0, 1, 1],
                    },
                    {
                        "id": 4,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [float("nan"), 0, 1, 1],
                    },
                ],
                [
                    {
                        "id": 1,
                        "name": "bird",
                        "keypoints": ["head", "tail"],
                        "skeleton": [[1, 2]],
                    }
                ],
            )
        )
    )

    parsed = parser._parse_split(image_dir, annotation_path)
    records = _records(parsed)
    assert parsed.skeletons == {
        "bird": {"labels": ["head", "tail"], "edges": [(0, 1)]}
    }
    assert len(records) == 2
    assert records[0]["annotation"]["instance_segmentation"]
    assert records[1]["annotation"]["instance_id"] == 2
    assert records[1]["annotation"]["keypoints"]["keypoints"] == [
        (0.0, 1.0, 2),
        (0.5, 0.5, 1),
    ]
    assert unlabeled not in parsed.files
    assert len(parser._issues.messages) == 4

    plain_annotations = tmp_path / "plain.json"
    plain_annotations.write_text(
        json.dumps(
            _coco_data(
                [
                    {
                        "id": 1,
                        "file_name": unlabeled.name,
                        "width": 20,
                        "height": 10,
                    }
                ]
            )
        )
    )
    assert _records(parser._parse_split(image_dir, plain_annotations)) == [
        {"file": unlabeled.resolve(), "annotation": None}
    ]


def test_coco_directory_parse_and_keypoint_paths(tmp_path: Path):
    parser = _plugin(COCOParser)
    root = tmp_path / "dataset"
    for split_name in ("train", "validation"):
        image_dir, _ = _write_coco_split(
            root / split_name,
            roboflow=False,
            data=_coco_data(
                [
                    {
                        "id": 1,
                        "file_name": f"{split_name}.jpg",
                        "width": 20,
                        "height": 10,
                    }
                ]
            ),
        )
        _image(image_dir / f"{split_name}.jpg")

    parsed = parser.parse(root, dataset_type="coco")
    assert parsed.splits is not None
    assert len(parsed.splits["val"]) + len(parsed.splits["test"]) == 1

    (root / "test").mkdir()
    with pytest.raises(ValueError, match="Test split"):
        parser.parse(root, dataset_type="coco")
    (root / "test").rmdir()

    single_split = tmp_path / "single"
    image_dir, _ = _write_coco_split(
        single_split,
        roboflow=True,
        data=_coco_data(
            [
                {
                    "id": 1,
                    "file_name": "single.jpg",
                    "width": 20,
                    "height": 10,
                }
            ]
        ),
    )
    _image(image_dir / "single.jpg")
    assert _records(parser.parse(single_split, dataset_type="coco"))

    keypoint_root = tmp_path / "keypoints"
    raw = keypoint_root / "raw"
    raw.mkdir(parents=True)
    paths = {
        "train": "raw/train.json",
        "val": "raw/val.json",
        "test": "raw/test.json",
    }
    for split_name, canonical in (("train", "train"), ("validation", "val")):
        image_dir, labels_path = _write_coco_split(
            keypoint_root / split_name,
            roboflow=False,
            data=_coco_data(
                [
                    {
                        "id": 1,
                        "file_name": f"{canonical}.jpg",
                        "width": 20,
                        "height": 10,
                    }
                ],
                categories=[{"id": 1, "name": "bird"}],
            ),
        )
        _image(image_dir / f"{canonical}.jpg")
        keypoint_data = _coco_data(
            [
                {
                    "id": 1,
                    "file_name": f"{canonical}.jpg",
                    "width": 20,
                    "height": 10,
                }
            ],
            [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 5],
                    "keypoints": [1, 1, 2],
                }
            ],
            [
                {
                    "id": 1,
                    "name": "bird",
                    "keypoints": ["head"],
                    "skeleton": [],
                }
            ],
        )
        (keypoint_root / paths[canonical]).write_text(
            json.dumps(keypoint_data)
        )
        assert labels_path.exists()

    parsed = parser.parse(
        keypoint_root,
        dataset_type="coco",
        use_keypoint_ann=True,
        keypoint_ann_paths=paths,
        split_val_to_test=False,
    )
    assert len(_records(parsed)) == 2

    test_image_dir, _ = _write_coco_split(
        keypoint_root / "test",
        roboflow=False,
        data=_coco_data(
            [
                {
                    "id": 1,
                    "file_name": "test.jpg",
                    "width": 20,
                    "height": 10,
                }
            ],
            categories=[{"id": 1, "name": "bird"}],
        ),
    )
    _image(test_image_dir / "test.jpg")
    parsed = parser.parse(
        keypoint_root,
        dataset_type="coco",
        use_keypoint_ann=True,
        keypoint_ann_paths=paths,
        split_val_to_test=False,
    )
    assert parsed.splits is not None
    assert parsed.splits["test"] == []

    test_keypoints = _coco_data(
        [
            {
                "id": 1,
                "file_name": "test.jpg",
                "width": 20,
                "height": 10,
            }
        ],
        [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 5],
                "keypoints": [1, 1, 2],
            }
        ],
        [
            {
                "id": 1,
                "name": "bird",
                "keypoints": ["head"],
                "skeleton": [],
            }
        ],
    )
    (keypoint_root / paths["test"]).write_text(json.dumps(test_keypoints))
    parsed = parser.parse(
        keypoint_root,
        dataset_type="coco",
        use_keypoint_ann=True,
        keypoint_ann_paths=paths,
    )
    assert parsed.splits is not None
    assert len(parsed.splits["test"]) == 1


def test_clean_coco_annotations(tmp_path: Path):
    annotation_path = tmp_path / "labels.json"
    untouched = _coco_data(
        [{"id": 1, "file_name": "safe.jpg", "width": 1, "height": 1}]
    )
    annotation_path.write_text(json.dumps(untouched))
    assert clean_annotations(annotation_path) == annotation_path

    annotation_path.write_text(
        json.dumps(
            _coco_data(
                [
                    {
                        "id": 1,
                        "file_name": "000000341448.jpg",
                        "width": 1,
                        "height": 1,
                    },
                    {
                        "id": 2,
                        "file_name": "safe.jpg",
                        "width": 1,
                        "height": 1,
                    },
                ],
                [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                    },
                ],
            )
        )
    )
    cleaned = clean_annotations(annotation_path)
    data = json.loads(cleaned.read_text())
    assert [image["id"] for image in data["images"]] == [2]
    assert [annotation["id"] for annotation in data["annotations"]] == [2]


def test_yolov8_format_detection_and_validation(tmp_path: Path):
    parser = _plugin(YOLOv8Parser)
    assert parser._detect_dataset_dir_format(tmp_path / "missing") == (
        None,
        [],
    )

    roboflow = tmp_path / "roboflow"
    (roboflow / "train").mkdir(parents=True)
    assert parser._detect_dataset_dir_format(roboflow) == (
        Format.ROBOFLOW,
        ["train"],
    )

    ultralytics = tmp_path / "ultralytics"
    (ultralytics / "images").mkdir(parents=True)
    (ultralytics / "labels").mkdir()
    assert parser._detect_dataset_dir_format(ultralytics) == (
        Format.ULTRALYTICS,
        ["images", "labels"],
    )
    empty = tmp_path / "empty-yolo"
    empty.mkdir()
    assert parser._detect_dataset_dir_format(empty) == (None, [])
    assert parser.discover_splits(empty) == {}

    split = tmp_path / "split"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    (split / "images").mkdir()
    (split / "labels").mkdir()
    assert parser.validate_split(split) is None
    image = _image(split / "images" / "image.jpg")
    assert parser.validate_split(split) is None
    (tmp_path / "dataset.yaml").write_text("names: [bird]\n")
    assert parser.validate_split(split) == {
        "image_dir": split / "images",
        "annotation_dir": split / "labels",
        "classes_path": tmp_path / "dataset.yaml",
    }
    assert image.exists()


def _write_yolo8_split(
    root: Path,
    annotations: dict[str, str],
    *,
    classes: list[str] | dict[int, str] | None = None,
    kpt_shape: list[int] | None = None,
) -> tuple[Path, Path, Path]:
    image_dir = root / "images"
    annotation_dir = root / "labels"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    for stem, annotation in annotations.items():
        _image(image_dir / f"{stem}.jpg")
        (annotation_dir / f"{stem}.txt").write_text(annotation)
    data: dict[str, Any] = {"names": classes or ["bird"]}
    if kpt_shape is not None:
        data["kpt_shape"] = kpt_shape
    classes_path = root.parent / f"{root.name}.yaml"
    classes_path.write_text(cast(str, yaml.safe_dump(data)))
    return image_dir, annotation_dir, classes_path


def test_yolov8_detection_and_segmentation(tmp_path: Path):
    parser = _plugin(YOLOv8Parser)
    detection = tmp_path / "detection" / "train"
    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        detection,
        {
            "detection": "0 0.5 0.5 0.4 0.2\n\n",
            "unlabeled": "\n",
        },
    )
    records = _records(
        parser._parse_split(image_dir, annotation_dir, classes_path)
    )
    assert len(records) == 2
    assert next(record for record in records if record["annotation"])[
        "annotation"
    ]["boundingbox"] == {"x": 0.3, "y": 0.4, "w": 0.4, "h": 0.2}

    segmentation = tmp_path / "segmentation" / "train"
    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        segmentation,
        {"segment": "0 0.1 0.2 0.8 0.2 0.8 0.9\n"},
        classes={0: "bird"},
    )
    records = _records(
        parser._parse_split(image_dir, annotation_dir, classes_path)
    )
    annotation = records[0]["annotation"]
    assert annotation["boundingbox"] == {
        "x": 0.1,
        "y": 0.2,
        "w": 0.7000000000000001,
        "h": 0.7,
    }
    assert annotation["instance_segmentation"]["points"] == [
        (0.1, 0.2),
        (0.8, 0.2),
        (0.8, 0.9),
    ]

    broken = tmp_path / "broken-yolo8"
    broken_images, broken_labels, broken_yaml = _write_yolo8_split(
        broken,
        {"broken": "0 0.1 0.2 0.8 0.2 0.8 0.9\n"},
    )
    (broken_images / "broken.jpg").write_text("broken")
    with pytest.raises(ValueError, match="Failed to read image"):
        parser._parse_split(broken_images, broken_labels, broken_yaml)


@pytest.mark.parametrize("kpt_dim", [2, 3])
def test_yolov8_keypoints(tmp_path: Path, kpt_dim: int):
    parser = _plugin(YOLOv8Parser)
    keypoint_values = (
        "0.1 0.2 0.3 0.4" if kpt_dim == 2 else "0.1 0.2 1 0.3 0.4 2"
    )
    root = tmp_path / f"keypoints-{kpt_dim}" / "train"
    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        root,
        {"pose": f"0 0.5 0.5 0.4 0.2 {keypoint_values}\n"},
        kpt_shape=[2, kpt_dim],
    )
    records = _records(
        parser._parse_split(image_dir, annotation_dir, classes_path)
    )
    keypoints = records[0]["annotation"]["keypoints"]["keypoints"]
    assert len(keypoints) == 2
    assert all(len(point) == 3 for point in keypoints)
    if kpt_dim == 2:
        assert all(point[2] == 2 for point in keypoints)


def test_yolov8_discovers_both_directory_layouts(tmp_path: Path):
    roboflow = tmp_path / "roboflow"
    _write_yolo8_split(roboflow / "train", {"image": ""})
    assert set(YOLOv8Parser.discover_splits(roboflow)) == {"train"}

    ultralytics = tmp_path / "ultralytics"
    image_dir = ultralytics / "images" / "val"
    labels_dir = ultralytics / "labels" / "val"
    image_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    _image(image_dir / "image.jpg")
    (ultralytics / "data.yml").write_text("names: [bird]\n")
    assert set(YOLOv8Parser.discover_splits(ultralytics)) == {"val"}


def _write_ndjson(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n" + "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_ultralytics_ndjson_local_annotations(tmp_path: Path):
    parser = _plugin(UltralyticsNDJSONParser)
    box_image = _image(tmp_path / "box.jpg")
    segment_image = _image(tmp_path / "segment.jpg")
    pose_image = _image(tmp_path / "pose.jpg")
    empty_image = _image(tmp_path / "empty.jpg")
    ndjson = _write_ndjson(
        tmp_path / "dataset.ndjson",
        [
            {
                "type": "dataset",
                "class_names": ["bird"],
                "kpt_shape": [2, 2],
            },
            {"type": "metadata"},
            {
                "type": "image",
                "file": box_image.name,
                "split": "train",
                "width": 20,
                "height": 10,
                "annotations": {"boxes": [[0, 0.5, 0.5, 0.4, 0.2]]},
            },
            {
                "type": "image",
                "file": str(segment_image.resolve()),
                "split": "validation",
                "width": 20,
                "height": 10,
                "annotations": {
                    "segments": [[0, 0.1, 0.2, 0.8, 0.2, 0.8, 0.9]]
                },
            },
            {
                "type": "image",
                "file": pose_image.name,
                "split": "test",
                "width": 20,
                "height": 10,
                "annotations": {
                    "pose": [
                        [
                            0,
                            0.5,
                            0.5,
                            0.4,
                            0.2,
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                        ]
                    ]
                },
            },
            {
                "type": "image",
                "file": empty_image.name,
                "width": 20,
                "height": 10,
            },
            {
                "type": "image",
                "file": "missing.jpg",
                "split": "mystery",
                "width": 20,
                "height": 10,
            },
            {
                "type": "image",
                "file": box_image.name,
                "split": "train",
                "width": 20,
                "height": 10,
            },
        ],
    )

    assert parser.supports(ndjson)
    assert parser.supports(tmp_path)
    parsed = parser.parse(tmp_path, dataset_type="ultralytics-ndjson")
    records = _records(parsed)

    assert len(records) == 5
    assert parsed.files == [box_image, segment_image, pose_image, empty_image]
    assert parsed.splits == {
        "train": [box_image, empty_image],
        "val": [segment_image],
        "test": [pose_image],
    }
    assert any(
        record["annotation"]
        and "instance_segmentation" in record["annotation"]
        for record in records
    )
    pose = next(
        record
        for record in records
        if record["annotation"] and "keypoints" in record["annotation"]
    )
    assert pose["annotation"]["keypoints"]["keypoints"] == [
        (0.1, 0.2, 2),
        (0.3, 0.4, 2),
    ]
    assert len(parser._issues.messages) == 1


def test_ultralytics_ndjson_pose_inference_and_errors(tmp_path: Path):
    parser = _plugin(UltralyticsNDJSONParser)
    image = _image(tmp_path / "pose.jpg")
    valid = _write_ndjson(
        tmp_path / "valid.ndjson",
        [
            {"type": "dataset", "class_names": {"0": "bird"}},
            {
                "type": "image",
                "file": image.name,
                "width": 20,
                "height": 10,
                "annotations": {
                    "pose": [
                        [
                            0,
                            0.5,
                            0.5,
                            0.4,
                            0.2,
                            0.1,
                            0.2,
                            1,
                            0.3,
                            0.4,
                            2,
                        ]
                    ]
                },
            },
        ],
    )
    parsed = parser.parse(valid, dataset_type="ultralytics-ndjson")
    assert _records(parsed)[0]["annotation"]["keypoints"]["keypoints"] == [
        (0.1, 0.2, 1),
        (0.3, 0.4, 2),
    ]

    invalid = _write_ndjson(
        tmp_path / "invalid-pose.ndjson",
        [
            {"type": "dataset", "class_names": ["bird"]},
            {
                "type": "image",
                "file": image.name,
                "width": 20,
                "height": 10,
                "annotations": {"pose": [[0, 0.5, 0.5, 0.4, 0.2, 0.1, 0.2]]},
            },
        ],
    )
    parsed = parser.parse(invalid, dataset_type="ultralytics-ndjson")
    with pytest.raises(ValueError, match="dimensionality is not inferable"):
        _records(parsed)

    wrong_suffix = tmp_path / "dataset.txt"
    wrong_suffix.write_text("x")
    with pytest.raises(ValueError, match="dataset file not found"):
        parser.parse(wrong_suffix, dataset_type="ultralytics-ndjson")

    invalid_header = _write_ndjson(
        tmp_path / "invalid-header.ndjson",
        [{"type": "dataset", "class_names": ["bird"]}],
    )
    with pytest.raises(ValueError, match="Invalid Ultralytics"):
        parser._build_record_stream(invalid_header)


def test_ultralytics_ndjson_path_and_header_validation(tmp_path: Path):
    parser = _plugin(UltralyticsNDJSONParser)
    assert parser._resolve_ndjson_path(tmp_path / "missing") is None

    directory = tmp_path / "directory"
    directory.mkdir()
    assert parser._resolve_ndjson_path(directory) is None
    first = _write_ndjson(directory / "first.ndjson", [])
    assert parser._resolve_ndjson_path(directory) == first.resolve()
    _write_ndjson(directory / "second.ndjson", [])
    assert parser._resolve_ndjson_path(directory) is None

    malformed = tmp_path / "malformed.ndjson"
    malformed.write_text("{")
    assert parser._load_header(malformed) is None

    wrong_first = _write_ndjson(
        tmp_path / "wrong-first.ndjson",
        [
            {"type": "metadata", "class_names": ["bird"]},
            {"type": "image", "file": "image.jpg"},
        ],
    )
    assert parser._load_header(wrong_first) is None

    no_classes = _write_ndjson(
        tmp_path / "no-classes.ndjson",
        [
            {"type": "dataset"},
            {"type": "image", "file": "image.jpg"},
        ],
    )
    assert parser._load_header(no_classes) is None
    assert parser._load_header(tmp_path / "missing") is None


def test_ultralytics_ndjson_remote_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    parser = _plugin(UltralyticsNDJSONParser)
    destinations: list[Path] = []

    class Downloader:
        @staticmethod
        def download(
            url: str,
            destination: Path,
            *,
            validate_image: bool,
        ) -> Path:
            assert validate_image
            assert url.startswith("https://")
            destinations.append(destination)
            return _image(destination)

    monkeypatch.setattr(
        UltralyticsNDJSONParser,
        "_remote_file_downloader",
        Downloader(),
    )
    ndjson = _write_ndjson(
        tmp_path / "remote.ndjson",
        [
            {"type": "dataset", "class_names": ["bird"]},
            {
                "type": "image",
                "file": "remote",
                "url": "https://example.com/image.png",
                "split": "valid",
                "width": 20,
                "height": 10,
            },
        ],
    )
    parsed = parser.parse(ndjson, dataset_type="ultralytics-ndjson")
    assert len(_records(parsed)) == 1
    assert destinations[0].parent.name == "val"
    assert destinations[0].suffix == ".png"

    with pytest.raises(ValueError, match="already exists"):
        parser.parse(
            ndjson,
            dataset_type="ultralytics-ndjson",
            reuse_cached=False,
        )
    assert _records(
        parser.parse(
            ndjson,
            dataset_type="ultralytics-ndjson",
            reuse_cached=True,
        )
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("train", "train"),
        ("val", "val"),
        ("test", "test"),
        ("valid", "val"),
        ("validation", "val"),
        (None, "train"),
        ("unknown", "train"),
    ],
)
def test_ultralytics_ndjson_split_normalization(
    name: str | None,
    expected: str,
):
    assert UltralyticsNDJSONParser._normalize_split_name(name) == expected


def test_ultralytics_ndjson_small_helpers():
    assert UltralyticsNDJSONParser._get_class_names(["bird"]) == {0: "bird"}
    assert UltralyticsNDJSONParser._get_class_names({"1": "cat"}) == {1: "cat"}
    assert UltralyticsNDJSONParser._fit_boundingbox(
        np.array([[0.1, 0.2], [0.8, 0.9]])
    ) == {"x": 0.1, "y": 0.2, "w": 0.7000000000000001, "h": 0.7}


def _solo_definitions(*, include_bbox: bool = True) -> dict[str, Any]:
    definitions: list[dict[str, Any]] = [
        {
            "@type": "type.unity.com/unity.solo.KeypointAnnotation",
            "template": {
                "keypoints": [
                    {"label": "tail", "index": 1},
                    {"label": "head", "index": 0},
                ]
            },
        },
        {
            "@type": (
                "type.unity.com/unity.solo.SemanticSegmentationAnnotation"
            )
        },
    ]
    if include_bbox:
        definitions.append(
            {
                "@type": ("type.unity.com/unity.solo.BoundingBox2DAnnotation"),
                "spec": [
                    {"label_name": "cat", "label_id": 2},
                    {"label_name": "bird", "label_id": 1},
                ],
            }
        )
    return {"annotationDefinitions": definitions}


def _write_solo_frame(
    split: Path,
    annotations: list[dict[str, Any]],
    *,
    image_name: str = "step0.camera.jpg",
    create_image: bool = True,
) -> Path:
    sequence = split / "sequence.0"
    sequence.mkdir(parents=True, exist_ok=True)
    if create_image:
        _image(sequence / image_name, size=(20, 10))
    frame = {
        "step": "0",
        "captures": [
            {
                "filename": image_name,
                "dimension": [20, 10],
                "annotations": annotations,
            },
            {
                "filename": image_name,
                "dimension": [20, 10],
                "annotations": [],
            },
        ],
    }
    frame_path = sequence / "step0.frame_data.json"
    frame_path.write_text(json.dumps(frame))
    return frame_path


def _write_solo_metadata(split: Path, *, total_sequences: int = 1) -> None:
    split.mkdir(parents=True, exist_ok=True)
    (split / "annotation_definitions.json").write_text(
        json.dumps(_solo_definitions())
    )
    (split / "metadata.json").write_text(
        json.dumps({"totalSequences": total_sequences})
    )
    (split / "metric_definitions.json").write_text("{}")
    (split / "sensor_definitions.json").write_text("{}")


def test_solo_parser_all_annotation_types(tmp_path: Path):
    parser = _plugin(SOLOParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    _write_solo_metadata(split, total_sequences=2)

    sequence = split / "sequence.0"
    sequence.mkdir()
    semantic_mask = sequence / "semantic.png"
    instance_mask = sequence / "instance.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(semantic_mask)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(instance_mask)
    annotations = [
        {
            "@type": (
                "type.unity.com/unity.solo.SemanticSegmentationAnnotation"
            ),
            "filename": semantic_mask.name,
            "instances": [
                {
                    "labelName": "bird",
                    "pixelValue": [255, 0, 0, 255],
                }
            ],
        },
        {
            "@type": "type.unity.com/unity.solo.BoundingBox2DAnnotation",
            "values": [
                {
                    "labelName": "bird",
                    "origin": [2, 1],
                    "dimension": [10, 5],
                    "instanceId": 1,
                }
            ],
        },
        {
            "@type": (
                "type.unity.com/unity.solo.InstanceSegmentationAnnotation"
            ),
            "filename": instance_mask.name,
            "instances": [
                {
                    "color": [255, 0, 0, 255],
                    "instanceId": 1,
                }
            ],
        },
        {
            "@type": "type.unity.com/unity.solo.KeypointAnnotation",
            "values": [
                {
                    "instanceId": 1,
                    "keypoints": [
                        {"location": [2, 1], "state": 2},
                        {"location": [10, 5], "state": 1},
                    ],
                }
            ],
        },
    ]
    _write_solo_frame(split, annotations)

    assert parser.validate_split(split) == {"split_path": split}
    parsed = parser._parse_split(split)
    records = _records(parsed)
    assert parsed.skeletons == {
        "bird": {"labels": ["head", "tail"]},
        "cat": {"labels": ["head", "tail"]},
    }
    assert len(records) == 2
    assert records[0]["annotation"]["class"] == "bird"
    combined = records[1]["annotation"]
    assert combined["boundingbox"] == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.5,
        "h": 0.5,
    }
    assert combined["keypoints"]["keypoints"] == [
        (0.1, 0.1, 2),
        (0.5, 0.5, 1),
    ]
    assert "instance_segmentation" in combined

    definitions = _solo_definitions()
    assert parser._get_solo_annotation_types(definitions) == [
        "KeypointAnnotation",
        "SemanticSegmentationAnnotation",
        "BoundingBox2DAnnotation",
    ]
    assert parser._get_solo_bbox_class_names(definitions) == ["bird", "cat"]
    assert parser._get_solo_keypoint_names(definitions) == ["head", "tail"]


def test_solo_parser_structure_errors(tmp_path: Path):
    parser = _plugin(SOLOParser)
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="path non-existent"):
        parser._parse_split(missing)

    no_definitions = tmp_path / "no-definitions"
    no_definitions.mkdir()
    with pytest.raises(FileNotFoundError, match="annotation_definitions"):
        parser._parse_split(no_definitions)

    no_bbox = tmp_path / "no-bbox"
    no_bbox.mkdir()
    (no_bbox / "annotation_definitions.json").write_text(
        json.dumps(_solo_definitions(include_bbox=False))
    )
    with pytest.raises(ValueError, match="No class_names"):
        parser._parse_split(no_bbox)


@pytest.mark.parametrize(
    ("annotation_type", "mask_name"),
    [
        ("SemanticSegmentationAnnotation", "semantic.png"),
        ("InstanceSegmentationAnnotation", "instance.png"),
    ],
)
def test_solo_parser_missing_masks(
    tmp_path: Path,
    annotation_type: str,
    mask_name: str,
):
    parser = _plugin(SOLOParser)
    split = tmp_path / annotation_type
    _write_solo_metadata(split)
    _write_solo_frame(
        split,
        [
            {
                "@type": f"type.unity.com/unity.solo.{annotation_type}",
                "filename": mask_name,
                "instances": [],
            }
        ],
    )
    with pytest.raises(FileNotFoundError, match="not existent"):
        parser._parse_split(split)

    mask = split / "sequence.0" / mask_name
    mask.write_text("broken")
    with pytest.raises(ValueError, match="Failed to read mask image"):
        parser._parse_split(split)


def test_solo_parser_missing_image(tmp_path: Path):
    parser = _plugin(SOLOParser)
    split = tmp_path / "missing-image"
    _write_solo_metadata(split)
    _write_solo_frame(split, [], create_image=False)
    with pytest.raises(FileNotFoundError, match="not existent"):
        parser._parse_split(split)


def test_prepare_source_routes_and_extracts_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    local = tmp_path / "local"
    local.mkdir()

    class FileSystem:
        source_path = local

        @classmethod
        def download(cls, source: str, destination: Path) -> Path:
            assert source
            assert destination
            return cls.source_path

    monkeypatch.setattr(parser_source, "LuxonisFileSystem", FileSystem)
    source, name = parser_source.prepare_source(
        "https://example.com/local", tmp_path
    )
    assert (source, name) == (local, "local")

    monkeypatch.setattr(
        parser_source,
        "_download_roboflow_dataset",
        lambda source, local_path: (local, "roboflow-name"),
    )
    assert parser_source.prepare_source(
        "roboflow://workspace/project/1/coco", tmp_path
    ) == (local, "roboflow-name")

    monkeypatch.setattr(
        parser_source,
        "_download_ultralytics_dataset",
        lambda source, local_path: (local, "ultralytics-name"),
    )
    assert parser_source.prepare_source(
        "ultralytics://user/datasets/project", tmp_path
    ) == (local, "ultralytics-name")

    archive = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("wrapper/train/bird/image.jpg", b"image")
    FileSystem.source_path = archive
    source, name = parser_source.prepare_source(archive, tmp_path)
    assert name == archive.name
    assert source == tmp_path / "wrapped" / "wrapper"


def test_resolve_extracted_zip_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    multiple = tmp_path / "multiple"
    (multiple / "one").mkdir(parents=True)
    (multiple / "two").mkdir()
    assert parser_source._resolve_extracted_zip_root(multiple) == multiple

    unrecognized = tmp_path / "unrecognized"
    only = unrecognized / "wrapper"
    (only / "content").mkdir(parents=True)

    class NeverParser:
        @classmethod
        def supports(cls, source: Path) -> bool:
            return False

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [NeverParser])
    assert (
        parser_source._resolve_extracted_zip_root(unrecognized) == unrecognized
    )

    marker = tmp_path / "marker"
    wrapped = marker / "wrapper"
    (wrapped / "train").mkdir(parents=True)
    assert parser_source._resolve_extracted_zip_root(marker) == wrapped

    recognized = tmp_path / "recognized"
    recognized_wrapper = recognized / "wrapper"
    (recognized_wrapper / "content").mkdir(parents=True)

    class RecognizedParser:
        @classmethod
        def supports(cls, source: Path) -> bool:
            return source == recognized_wrapper

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [RecognizedParser])
    assert (
        parser_source._resolve_extracted_zip_root(recognized)
        == recognized_wrapper
    )


def _fake_roboflow_module(location: Path) -> types.ModuleType:
    module = types.ModuleType("roboflow")

    class Version:
        @staticmethod
        def download(
            export_format: str, destination: str
        ) -> types.SimpleNamespace:
            assert export_format == "coco"
            # Compared as a path, not as a string: the destination is
            # built with `/`, which is a backslash on Windows.
            assert Path(destination).name == "project"
            return types.SimpleNamespace(location=str(location))

    class Project:
        @staticmethod
        def version(version: int) -> Version:
            assert version == 2
            return Version()

    class Workspace:
        @staticmethod
        def project(project: str) -> Project:
            assert project == "project"
            return Project()

    class Roboflow:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "secret"

        @staticmethod
        def workspace(workspace: str) -> Workspace:
            assert workspace == "workspace"
            return Workspace()

    cast(Any, module).Roboflow = Roboflow
    return module


def test_download_roboflow_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(parser_source.environ, "ROBOFLOW_API_KEY", None)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        parser_source._download_roboflow_dataset(
            "roboflow://workspace/project/2/coco", tmp_path
        )

    monkeypatch.setattr(
        parser_source.environ,
        "ROBOFLOW_API_KEY",
        SecretStr("secret"),
    )
    monkeypatch.setattr(parser_source, "find_spec", lambda name: object())
    monkeypatch.setitem(
        sys.modules,
        "roboflow",
        _fake_roboflow_module(tmp_path / "downloaded"),
    )

    with pytest.raises(ValueError, match="Incorrect Roboflow"):
        parser_source._download_roboflow_dataset(
            "roboflow://workspace/project/2", tmp_path
        )
    with pytest.raises(ValueError, match="must be an integer"):
        parser_source._download_roboflow_dataset(
            "roboflow://workspace/project/latest/coco", tmp_path
        )

    assert parser_source._download_roboflow_dataset(
        "roboflow://workspace/project/2/coco", tmp_path
    ) == (tmp_path / "downloaded", "project")


class _Response:
    def __init__(
        self,
        *,
        ok: bool,
        status_code: int,
        payload: dict[str, Any] | ValueError,
        text: str = "",
        reason: str = "",
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.reason = reason

    def json(self) -> dict[str, Any]:
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


def test_download_ultralytics_reference_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(parser_source.environ, "ULTRALYTICS_API_KEY", None)
    with pytest.raises(RuntimeError, match="ULTRALYTICS_API_KEY"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/datasets/project", tmp_path
        )

    monkeypatch.setattr(
        parser_source.environ,
        "ULTRALYTICS_API_KEY",
        SecretStr("secret"),
    )
    with pytest.raises(ValueError, match="must be an integer"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/datasets/project?v=latest", tmp_path
        )
    with pytest.raises(ValueError, match="must be >= 1"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/datasets/project?v=0", tmp_path
        )
    with pytest.raises(ValueError, match="Incorrect Ultralytics"):
        parser_source._download_ultralytics_dataset(
            "ultralytics://user/projects/project", tmp_path
        )


def test_download_ultralytics_api_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        parser_source.environ,
        "ULTRALYTICS_API_KEY",
        SecretStr("secret"),
    )
    reference = "ultralytics://user/datasets/project"

    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: _Response(
            ok=False,
            status_code=401,
            payload={"error": "unauthorized"},
        ),
    )
    with pytest.raises(RuntimeError, match=r"401.*unauthorized"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)

    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: _Response(
            ok=False,
            status_code=500,
            payload=ValueError(),
            text="server failed",
        ),
    )
    with pytest.raises(RuntimeError, match=r"500.*server failed"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)

    dataset_response = _Response(
        ok=True,
        status_code=200,
        payload={
            "dataset": {
                "_id": "dataset-id",
                "slug": "project",
                "name": "Project",
            }
        },
    )
    export_json_error = _Response(
        ok=False,
        status_code=422,
        payload={"error": "bad export"},
        reason="Unprocessable",
    )
    responses = iter([dataset_response, export_json_error])
    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(RuntimeError, match="422 Unprocessable: bad export"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)

    export_text_error = _Response(
        ok=False,
        status_code=503,
        payload=ValueError(),
        text="",
        reason="Unavailable",
    )
    responses = iter([dataset_response, export_text_error])
    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(RuntimeError, match="503 Unavailable"):
        parser_source._download_ultralytics_dataset(reference, tmp_path)


@pytest.mark.parametrize("version", [None, 3])
def test_download_ultralytics_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int | None,
):
    monkeypatch.setattr(
        parser_source.environ,
        "ULTRALYTICS_API_KEY",
        SecretStr("secret"),
    )
    dataset_response = _Response(
        ok=True,
        status_code=200,
        payload={
            "dataset": {
                "_id": "dataset-id",
                "slug": "project",
                "name": "Project",
            }
        },
    )
    export_response = _Response(
        ok=True,
        status_code=200,
        payload={"downloadUrl": "https://example.com/export.ndjson"},
    )
    responses = iter([dataset_response, export_response])
    monkeypatch.setattr(
        parser_source.requests,
        "get",
        lambda *args, **kwargs: next(responses),
    )

    def download(url: str, destination: Path, *, timeout: float) -> None:
        assert url.endswith("export.ndjson")
        assert timeout == 120.0
        destination.write_text("downloaded")

    monkeypatch.setattr(parser_source, "download_remote_file", download)
    suffix = f"?v={version}" if version is not None else ""
    destination, name = parser_source._download_ultralytics_dataset(
        f"ultralytics://user/datasets/project{suffix}",
        tmp_path,
    )
    assert destination.name == (
        f"project.v{version}.ndjson"
        if version is not None
        else "project.ndjson"
    )
    assert destination.read_text() == "downloaded"
    assert name == "Project"


def test_luxonis_parser_forwards_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The deprecated wrapper forwards arguments to ``import_dataset``.

    Source acquisition and format resolution happen in ``__init__`` — that is
    what makes an unsupported source fail at construction, as it did before the
    deprecation, and what stops each ``parse`` call from downloading the source
    again — so the forwarded ``source`` is the resolved local path and the
    forwarded ``dataset_type`` is the resolved type name.
    """
    calls: list[dict[str, Any]] = []

    class Result:
        @staticmethod
        def get_parser_issue_messages() -> list[str]:
            return ["issue"]

    def import_dataset(cls: type[Any], source: str, **kwargs: Any) -> Result:
        calls.append({"cls": cls, "source": source, **kwargs})
        return Result()

    monkeypatch.setattr(
        "luxonis_ml.data.parsers.luxonis_parser.LuxonisDataset.import_dataset",
        classmethod(import_dataset),
    )
    monkeypatch.setattr(
        "luxonis_ml.data.parsers.luxonis_parser.get_parser_plugin",
        lambda source, dataset_type: (
            _SyntheticSplitParser,
            dataset_type or "synthetic-split",
        ),
    )
    with pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"):
        parser = LuxonisParser(
            str(tmp_path),
            dataset_name="dataset",
            save_dir=tmp_path,
            dataset_type="coco",
            task_name="detection",
            full_warnings=True,
            delete_local=True,
        )
    assert parser.get_parser_issue_messages() == []
    result = parser.parse(
        split="train",
        random_split=False,
        split_ratios={"train": 1, "val": 0, "test": 0},
        use_keypoint_ann=True,
    )
    assert result.get_parser_issue_messages() == ["issue"]
    assert parser._get_parser_issue_messages() == ["issue"]
    assert calls == [
        {
            "cls": LuxonisDataset,
            "source": tmp_path,
            "dataset_name": "dataset",
            "save_dir": tmp_path,
            "dataset_type": "coco",
            "task_name": "detection",
            "full_warnings": True,
            "split": "train",
            "random_split": False,
            "split_ratios": {"train": 1, "val": 0, "test": 0},
            "parser_kwargs": {"use_keypoint_ann": True},
            "delete_local": True,
        }
    ]

    class PluginDataset:
        @classmethod
        def import_dataset(cls, source: str, **kwargs: Any) -> Result:
            return import_dataset(cls, source, **kwargs)

    monkeypatch.setattr(
        "luxonis_ml.data.parsers.luxonis_parser.DATASETS_REGISTRY.get",
        lambda name: PluginDataset,
    )
    with pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"):
        plugin_parser = LuxonisParser(str(tmp_path), dataset_plugin="plugin")
    plugin_parser.parse()
    assert calls[-1]["cls"] is PluginDataset
