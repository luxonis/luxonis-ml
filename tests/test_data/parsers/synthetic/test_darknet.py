"""Darknet parser."""

import builtins
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from luxonis_ml.data.parsers import (
    DarknetParser,
)
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
)


def _darknet_split(
    root: Path,
    labeled: Sequence[str],
    unlabeled: Sequence[str] = (),
    split: str = "train",
) -> Path:
    """Write a Darknet split with one box per labeled image.

    Args:
        root: Directory the split is created in.
        labeled: Stems of images that get a label file.
        unlabeled: Stems of images that get no label file.
        split: Name of the split directory to create.

    Returns:
        The created split directory.

    """
    split_path = root / split
    split_path.mkdir(parents=True, exist_ok=True)
    (split_path / "_darknet.labels").write_text("bird\n")
    for name in labeled:
        _image(split_path / f"{name}.jpg")
        (split_path / f"{name}.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    for name in unlabeled:
        _image(split_path / f"{name}.jpg")
    return split_path


def _count_label_opens(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count how often each label file is opened, by file name."""
    opened: Counter[str] = Counter()
    real_open = builtins.open

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            name = os.fspath(file)
            if name.endswith(".txt"):
                opened[Path(name).name] += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    return opened


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
    records = _records(parser._split_records(**kwargs))

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


def test_darknet_reads_every_label_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A full parse must read the split exactly once.

    Regression: the file list used to be collected by running the record
    generator a second time, so every label file was opened twice and the
    reported files were rebuilt from records that had already been produced.
    The file list now comes from the image listing, and the guard is that a
    complete parse opens each label file once while still reporting the very
    files the records refer to, in the same order.
    """
    split = _darknet_split(tmp_path, ["one", "two"], ["background"])
    parser = _plugin(DarknetParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    opened = _count_label_opens(monkeypatch)
    records = _records(parser._split_records(**kwargs))

    assert opened == Counter({"one.txt": 1, "two.txt": 1})
    assert parser._split_files(**kwargs) == list(
        dict.fromkeys(Path(record["file"]) for record in records)
    )


def test_darknet_files_are_listed_without_reading_any_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Enumerating the files must not parse, and records must stay lazy.

    Regression: the file list used to be built by the parse itself, which
    is why a count-based `split_ratios` paid for a full parse it threw
    away. `_split_files` answers from the image listing instead, so the
    guard checks it from both sides: enumerating the three images reads no
    label file at all, and the record stream reads a label file only once a
    record is actually pulled - never materializing a multi-GB dataset.
    """
    split = _darknet_split(tmp_path, ["one", "two", "three"])
    parser = _plugin(DarknetParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    opened = _count_label_opens(monkeypatch)
    assert len(parser._split_files(**kwargs)) == 3
    assert sum(opened.values()) == 0

    records = parser._split_records(**kwargs)
    assert sum(opened.values()) == 0

    next(iter(records))
    assert sum(opened.values()) == 1


def test_darknet_lists_the_split_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Recognizing and parsing a split must list it a single time.

    Regression: `validate_split` lists the images to recognize the split and
    `_split_records` listed the same directory again for the same answer. The
    listing is passed on as a parse argument instead; the guard is that
    validating, parsing and enumerating the files lists the directory once.
    Callers that pass no listing must still get an identical parse, so the
    fallback is checked to list the directory itself and produce the same
    output.
    """
    listed: list[Path] = []
    real_list_images = DarknetParser._list_images

    def counting_list_images(image_dir: Path) -> list[Path]:
        listed.append(image_dir)
        return real_list_images(image_dir)

    monkeypatch.setattr(
        DarknetParser, "_list_images", staticmethod(counting_list_images)
    )

    split = _darknet_split(tmp_path, ["one", "two"], ["background"])
    parser = _plugin(DarknetParser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    assert listed == [split]

    records = _records(parser._split_records(**kwargs))
    files = parser._split_files(**kwargs)
    assert listed == [split]

    fallback = parser._split_records(
        kwargs["image_dir"], kwargs["classes_path"]
    )
    assert listed == [split, split]
    assert _records(fallback) == records
    assert (
        parser._split_files(kwargs["image_dir"], kwargs["classes_path"])
        == files
    )
    assert listed == [split, split, split]


def test_darknet_streams_records_tagged_with_their_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """`detect` finds the splits and `parse` tags every record with one.

    The layout `detect` returns is handed straight to `parse`, so the
    source is walked once per import: enumerating the files of all three
    splits reads no label file, and the whole parse then reads each label
    file exactly once - once per split, not once per pass.
    """
    for split in ("train", "valid", "test"):
        _darknet_split(tmp_path, ["one", "two"], ["background"], split=split)

    layout = DarknetParser.detect(tmp_path)
    assert layout is not None
    assert layout.split_names == ["train", "val", "test"]

    parser = _plugin(DarknetParser)
    opened = _count_label_opens(monkeypatch)
    enumerated = parser.enumerate_files(tmp_path, layout)
    assert enumerated is not None
    assert {name: len(files) for name, files in enumerated.items()} == {
        "train": 3,
        "val": 3,
        "test": 3,
    }
    assert sum(opened.values()) == 0

    result = parser.parse(tmp_path, layout)
    tagged = cast(
        list[tuple[str | None, dict[str, Any]]], list(result.records)
    )
    assert Counter(split for split, _ in tagged) == Counter(
        {"train": 3, "val": 3, "test": 3}
    )
    assert opened == Counter({"one.txt": 3, "two.txt": 3})
    assert result.skeletons == {}

    for split_name, files in enumerated.items():
        assert files == list(
            dict.fromkeys(
                Path(record["file"])
                for split, record in tagged
                if split == split_name
            )
        )
