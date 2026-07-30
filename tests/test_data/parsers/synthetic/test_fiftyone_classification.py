"""FiftyOne classification parser."""

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from luxonis_ml.data import (
    LuxonisDataset,
)
from luxonis_ml.data.parsers import (
    FiftyOneClassificationParser,
)
from luxonis_ml.data.parsers import (
    fiftyone_classification_parser as parser_module,
)
from luxonis_ml.data.parsers.fiftyone_classification_parser import (
    clean_imagenet_annotations,
)
from luxonis_ml.enums import DatasetType
from tests.test_data.parsers.helpers import (
    _collect_raw_records,
    _image,
    _plugin,
)
from tests.test_data.utils import create_image


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
    assert _collect_raw_records(parser._split_records(**kwargs)) == [
        {"file": image, "annotation": {"class": "bird"}}
    ]
    assert len(parser._issues.messages) == 1

    flat = tmp_path / "flat"
    flat_image = _image(flat / "data" / "flat.jpg")
    (flat / "labels.json").write_text(
        json.dumps({"classes": ["bird"], "labels": {"flat": 0}})
    )
    assert _collect_raw_records(parser._split_records(flat)) == [
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


class _TrackedLabels(dict):
    """Label map that records how often and how far it is walked."""

    def __init__(self, labels: Mapping[str, int]):
        super().__init__(labels)
        self.walks = 0
        self.visited: list[str] = []

    def items(self) -> Iterator[tuple[str, int]]:
        self.walks += 1
        for image_stem, class_idx in super().items():
            self.visited.append(image_stem)
            yield image_stem, class_idx


def _fiftyone_split(
    split_path: Path,
    image_stems: Sequence[str],
    labels: Mapping[str, int],
    classes: Sequence[str] = ("bird",),
) -> Path:
    """Write one FiftyOne split directory and return its path."""
    split_path.mkdir(parents=True, exist_ok=True)
    for image_stem in image_stems:
        _image(split_path / "data" / f"{image_stem}.jpg")
    (split_path / "labels.json").write_text(
        json.dumps({"classes": list(classes), "labels": dict(labels)})
    )
    return split_path


def test_labels_are_walked_once_and_records_stream_lazily(tmp_path: Path):
    """The parse walks the label map exactly once - the pass that existed
    only to publish a file list up front is gone - and it still produces
    the records one at a time instead of materializing them.
    """
    stems = {"a": 0, "b": 0, "c": 0}
    split = _fiftyone_split(tmp_path / "train", list(stems), stems)
    labels = _TrackedLabels(stems)
    parser = _plugin(FiftyOneClassificationParser)

    records = parser._split_records(
        split, {"classes": ["bird"], "labels": labels}
    )

    # Nothing is read before a record is asked for.
    assert labels.walks == 0
    assert labels.visited == []

    first = next(records)
    assert isinstance(first, dict)
    assert first["file"] == split / "data" / "a.jpg"
    # Only the first label was looked at, so the records are not
    # materialized when the iterator is created.
    assert labels.walks == 1
    assert labels.visited == ["a"]

    assert len(_collect_raw_records(records)) == 2
    assert labels.walks == 1
    assert labels.visited == ["a", "b", "c"]


def test_labels_file_is_read_once_per_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Recognizing a split already parses its ``labels.json``, and the
    layout that produced is handed to the parse, so the same file is
    never opened and decoded a second time.
    """
    root = tmp_path / "dataset"
    for split_name in ("train", "validation", "test"):
        _fiftyone_split(
            root / split_name,
            [f"{split_name}_0"],
            {f"{split_name}_0": 0},
        )

    opened: list[str] = []
    real_open = open

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(parser_module, "open", counting_open, raising=False)

    layout = FiftyOneClassificationParser.detect(root)
    assert layout is not None
    parser = _plugin(FiftyOneClassificationParser)
    streamed = list(parser.parse(root, layout).records)

    assert [split_name for split_name, _ in streamed] == [
        "train",
        "val",
        "test",
    ]
    assert [name for name in opened if name.endswith("labels.json")] == [
        str(root / split_name / "labels.json")
        for split_name in ("train", "validation", "test")
    ]


def test_records_and_files_follow_label_order(tmp_path: Path):
    """The records come out in label order, leaving out the images no
    label mentions and the files that are not images at all, and the
    listing that count-based split ratios use reports exactly the files
    those records name - without reporting the split's issues, which the
    parse itself reports as it reaches them.
    """
    split = tmp_path / "train"
    jpg = _image(split / "data" / "labelled_jpg.jpg")
    png = _image(split / "data" / "labelled_png.png")
    _image(split / "data" / "unreferenced.jpg")
    (split / "data" / "notes.txt").write_text("not an image")
    (split / "labels.json").write_text(
        json.dumps(
            {
                "classes": ["bird", "cat"],
                "labels": {
                    "labelled_jpg": 1,
                    "dangling": 0,
                    "labelled_png": 0,
                },
            }
        )
    )

    parser = _plugin(FiftyOneClassificationParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    assert parser._split_files(**kwargs) == [jpg, png]
    assert parser._issues.messages == []

    assert _collect_raw_records(parser._split_records(**kwargs)) == [
        {"file": jpg, "annotation": {"class": "cat"}},
        {"file": png, "annotation": {"class": "bird"}},
    ]
    assert [message.image for message in parser._issues.messages] == [
        "dangling"
    ]


def test_invalid_class_index_raises_while_parsing(tmp_path: Path):
    """A label pointing outside the class list is an error raised by the
    parser rather than a record it quietly drops.

    The records stream, so the error surfaces when the offending label is
    reached instead of before the first record; the labels are only
    walked once now, and checking every index up front would be the
    second walk this change removed.
    """
    split = _fiftyone_split(tmp_path / "train", ["a", "b"], {"a": 0, "b": 7})
    parser = _plugin(FiftyOneClassificationParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    records = parser._split_records(**kwargs)
    first = next(records)
    assert isinstance(first, dict)
    assert first["file"] == split / "data" / "a.jpg"

    with pytest.raises(IndexError):
        next(records)


def test_flat_layout_parses_the_cleaned_labels(tmp_path: Path):
    """The labels handed over by validation are only reusable while the
    file they came from is the one being parsed: a flat layout whose
    labels the ImageNet cleanup rewrites must be parsed from the cleaned
    file, not from what validation happened to read.
    """
    flat = tmp_path / "flat"
    image = _image(flat / "data" / "006742.jpg")
    (flat / "labels.json").write_text(
        json.dumps(
            {
                "classes": [f"class-{index}" for index in range(640)],
                "labels": {"006742": 517},
            }
        )
    )

    parser = _plugin(FiftyOneClassificationParser)
    kwargs = parser.validate_split(flat)
    assert kwargs is not None

    # The cleanup only re-points labels it never adds or removes, so
    # listing the files answers from the labels validation read and
    # writes nothing.
    assert parser._split_files(**kwargs) == [image]
    assert not (flat / "labels_fixed.json").exists()

    assert _collect_raw_records(parser._split_records(**kwargs)) == [
        {"file": image, "annotation": {"class": "class-134"}}
    ]
    assert (flat / "labels_fixed.json").exists()
