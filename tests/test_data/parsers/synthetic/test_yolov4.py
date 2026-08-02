"""YOLOv4 parser."""

import builtins
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from luxonis_ml.data.parsers import (
    YoloV4Parser,
)
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
)
from tests.test_data.utils import create_image


def _yolov4_split(root: Path) -> Path:
    """Write a split holding one image of every kind the parser knows.

    Args:
        root: Directory the ``train`` split is written into.

    Returns:
        The split directory.

    """
    split = root / "train"
    _image(split / "boxed_0.jpg")
    _image(split / "boxed_1.jpg")
    _image(split / "nested" / "boxed_2.jpg")
    _image(split / "listed_without_boxes.jpg")
    _image(split / "unlisted.jpg")
    (split / "_classes.txt").write_text("bird\n")
    (split / "_annotations.txt").write_text(
        "boxed_0.jpg 2,1,12,6,0 4,2,14,7,0\n"
        "boxed_1.jpg 2,1,12,6,0\n"
        "nested/boxed_2.jpg 2,1,12,6,0\n"
        "listed_without_boxes.jpg\n"
        "missing.jpg 2,1,12,6,0\n"
    )
    return split


def _count_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record every image `Image.open` is called on."""
    decoded: list[str] = []
    real_open = Image.open

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        decoded.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(Image, "open", counting_open)
    return decoded


def _count_annotation_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record every time the annotation file is opened."""
    reads: list[str] = []
    real_open = builtins.open

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file).endswith("_annotations.txt"):
            reads.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    return reads


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
    records = _records(parser._split_records(**kwargs))

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


def test_yolov4_walks_the_split_once_and_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The records are the only walk of the split.

    Regression: the file list had to be complete before the records were
    consumed, so the annotation file was walked once to build it and a
    second time to stream the records - and an even earlier version built
    the list by running the record generator and throwing every record
    away, decoding every annotated image twice. The files now come from
    the records themselves, so the annotation file is opened exactly once
    and nothing is read until a record is asked for.
    """
    split = _yolov4_split(tmp_path)

    decoded = _count_decodes(monkeypatch)
    annotation_reads = _count_annotation_reads(monkeypatch)

    parser = _plugin(YoloV4Parser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    records = parser._split_records(**kwargs)

    assert decoded == [], "no image is read before a record is asked for"
    assert annotation_reads == [], "and neither is the annotation file"

    first = next(records)
    assert isinstance(first, dict)
    assert Path(first["file"]).name == "boxed_0.jpg"
    assert decoded == [str(split / "boxed_0.jpg")], "records stream"

    rest = _records(records)
    assert len(rest) == 5

    # The order the records name their files in is the order a split's
    # files are collected in, so it is part of what the parser promises.
    named = list(
        dict.fromkeys(Path(record["file"]).name for record in [first, *rest])
    )
    assert named == [
        "boxed_0.jpg",
        "boxed_1.jpg",
        "boxed_2.jpg",
        "listed_without_boxes.jpg",
        "unlisted.jpg",
    ]

    # `boxed_0.jpg` carries two boxes, so a decode per annotation instead of
    # a decode per image would show up here as well.
    assert decoded == [
        str(split / "boxed_0.jpg"),
        str(split / "boxed_1.jpg"),
        str(split / "nested" / "boxed_2.jpg"),
    ]
    assert annotation_reads == [str(split / "_annotations.txt")]
    assert len(parser._issues.messages) == 1


def test_yolov4_split_files_cost_no_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Count-based split ratios ask for the files, not the annotations.

    `_split_files` is what keeps that cheap, and its answer has to be
    exactly the files the records name, in the same order: a count-based
    import selects from it and then keeps only the records naming one of
    the selected files, so an image listed here but never emitted - or
    emitted in a different order - would silently change what is imported.
    """
    split = _yolov4_split(tmp_path)

    decoded = _count_decodes(monkeypatch)

    parser = _plugin(YoloV4Parser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    files = parser._split_files(**kwargs)

    assert decoded == [], "listing the files reads no image"
    assert [file.name for file in files] == [
        "boxed_0.jpg",
        "boxed_1.jpg",
        "boxed_2.jpg",
        "listed_without_boxes.jpg",
        "unlisted.jpg",
    ]

    records = _records(_plugin(YoloV4Parser)._split_records(**kwargs))
    assert files == list(
        dict.fromkeys(Path(record["file"]) for record in records)
    )

    layout = YoloV4Parser.detect(tmp_path)
    assert layout is not None
    assert parser.enumerate_files(tmp_path, layout) == {"train": files}


def test_yolov4_resolves_each_annotated_path_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Resolving costs a syscall per path component, so it is done once.

    Regression: the split directory was resolved again for every line of the
    annotation file, every line's image was resolved a second time to fill
    the set of annotated images, and the whole loop ran twice because the
    file list came from a discarded parse - 38 resolves for this split. What
    is left is one resolve for the split directory, one per annotation line,
    and one per image of the directory that no annotation claims by name.
    """
    split = _yolov4_split(tmp_path)

    resolved: list[str] = []
    real_resolve = Path.resolve

    def counting_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        resolved.append(str(self))
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting_resolve)

    parser = _plugin(YoloV4Parser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None
    assert len(_records(parser._split_records(**kwargs))) == 6

    assert len(resolved) <= 1 + 5 + 1


@pytest.mark.skipif(
    os.name == "nt", reason="symlinks need extra privileges on Windows"
)
def test_yolov4_symlinked_image_is_not_reported_as_unlisted(tmp_path: Path):
    """A symlink to an annotated image is that same annotated image.

    The name of a directory image may stand in for a resolved annotated path
    only because resolving never returns a symlink. An image no annotation
    names still has to be resolved before it counts as unlisted; comparing
    names alone would emit ``link.jpg`` as an extra annotation-less record
    for an image that is already annotated through its target.
    """
    split = tmp_path / "train"
    target = _image(split / "target.jpg")
    (split / "link.jpg").symlink_to(target)
    (split / "_classes.txt").write_text("bird\n")
    (split / "_annotations.txt").write_text("link.jpg 2,1,12,6,0\n")

    parser = _plugin(YoloV4Parser)
    kwargs = parser.validate_split(split)
    assert kwargs is not None

    records = _records(parser._split_records(**kwargs))
    assert [Path(record["file"]).name for record in records] == ["target.jpg"]
    assert [file.name for file in parser._split_files(**kwargs)] == [
        "target.jpg"
    ]


def test_yolov4_parser_keeps_unlabeled_image_with_duplicate_basename(
    tempdir: Path,
):
    split_dir = tempdir / "train"
    split_dir.mkdir()
    nested_dir = split_dir / "nested"
    nested_dir.mkdir()

    unlabeled_image = create_image(0, split_dir)
    annotated_image = create_image(0, nested_dir)

    annotations_path = split_dir / "_annotations.txt"
    annotations_path.write_text(
        "nested/img_0.jpg 0,0,10,10,0\n", encoding="utf-8"
    )
    classes_path = split_dir / "_classes.txt"
    classes_path.write_text("class0\n", encoding="utf-8")

    layout = YoloV4Parser.detect(split_dir)
    assert layout is not None
    parsed = _plugin(YoloV4Parser).parse(split_dir, layout)

    records = list(parsed.records)
    assert {split_name for split_name, _ in records} == {None}, (
        "a bare split directory carries no split name"
    )
    files = {
        Path(
            record["file"] if isinstance(record, dict) else record.file
        ).resolve()
        for _, record in records
    }

    assert annotated_image.resolve() in files
    assert unlabeled_image.resolve() in files
