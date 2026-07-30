"""YOLOv6 parser."""

import builtins
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from luxonis_ml.data import ParseResult
from luxonis_ml.data.parsers import (
    YoloV6Parser,
)
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
)

LABEL = "0 0.5 0.5 0.4 0.2\n"


def _parse(parser: YoloV6Parser, source: Path) -> ParseResult:
    """Detect ``source`` and parse the layout detection returned."""
    layout = YoloV6Parser.detect(source)
    assert layout is not None
    return parser.parse(source, layout)


def _files(records: list[dict[str, Any]]) -> list[Path]:
    """Collect the files the records name, the way the importer does."""
    return list(dict.fromkeys(Path(record["file"]) for record in records))


def _dataset(
    root: Path,
    *,
    splits: tuple[str, ...] = ("train",),
    images: int = 4,
    unlabeled: tuple[int, ...] = (0,),
) -> dict[str, list[Path]]:
    """Write a YOLOv6 dataset and return the images of every split.

    Args:
        root: Dataset root to populate.
        splits: Split names to write.
        images: Images per split.
        unlabeled: Image indices written without a label file, which the
            parser reports as a record without an annotation.

    Returns:
        The image paths written for each split, in creation order.

    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("names: [bird]\n")
    written = {}
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        label_dir.mkdir(parents=True)
        paths = []
        for index in range(images):
            paths.append(_image(image_dir / f"{split}_{index}.jpg"))
            if index not in unlabeled:
                (label_dir / f"{split}_{index}.txt").write_text(LABEL)
        written[split] = paths
    return written


def test_yolov6_parser(tmp_path: Path):
    parser = _plugin(YoloV6Parser)
    root = tmp_path / "dataset"
    assert YoloV6Parser.detect(root) is None

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
    (labels_dir / "labeled.txt").write_text(LABEL)

    layout = YoloV6Parser.detect(root)
    assert layout is not None
    assert layout.split_names == ["train"]

    tagged = _split_records(parser.parse(root, layout))
    assert {split_name for split_name, _ in tagged} == {"train"}

    records = [record for _, record in tagged]
    assert _files(records) == [labeled, unlabeled]
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


def test_yolov6_detects_a_single_split_directory(tmp_path: Path):
    """Guard parsing one ``images/<split>`` directory on its own.

    Pointed straight at a split, the parser resolves ``labels/`` and
    ``data.yaml`` from that directory's grandparent, which is the dataset
    root either way. The source names no split, so neither do its records.
    """
    root = tmp_path / "dataset"
    written = _dataset(root, splits=("train",), images=3)
    parser = _plugin(YoloV6Parser)

    split_dir = root / "images" / "train"
    layout = YoloV6Parser.detect(split_dir)
    assert layout is not None
    assert layout.split_names == []

    tagged = _split_records(parser.parse(split_dir, layout))
    assert {split_name for split_name, _ in tagged} == {None}
    assert sorted(_files([record for _, record in tagged])) == sorted(
        written["train"]
    )


def test_yolov6_reads_every_label_file_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard the single walk over the dataset.

    The file list used to be collected by running the whole generator a
    second time, which opened and re-parsed every label file twice. The
    importer collects the files from the records as they stream past now,
    so each label file must be read once - and only while the records are
    consumed, never during `detect` or `parse` themselves.
    """
    root = tmp_path / "dataset"
    _dataset(root, splits=("train", "valid"), images=4)
    parser = _plugin(YoloV6Parser)

    opened: Counter[str] = Counter()
    real_open = builtins.open

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened[str(file)] += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)

    layout = YoloV6Parser.detect(root)
    assert layout is not None
    parsed = parser.parse(root, layout)
    # Not one file is read before the first record is pulled: recognizing a
    # split only lists its images, and the parse is a stream.
    assert opened == Counter()

    records = _records(parsed)
    assert len(_files(records)) == 8
    label_reads = {
        name: count for name, count in opened.items() if name.endswith(".txt")
    }
    assert len(label_reads) == 6
    assert set(label_reads.values()) == {1}


def test_yolov6_lists_each_image_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard the reuse of the listing made while recognizing a split.

    `detect` has to list an image directory to decide whether the split
    holds any image at all, and that listing is what both the records and
    the enumerated files are drawn from. Listing again in `_split_records`,
    in `_split_files`, or once more inside the record generator walks a
    large directory for nothing.
    """
    root = tmp_path / "dataset"
    written = _dataset(root, splits=("train", "test"), images=3)
    parser = _plugin(YoloV6Parser)

    listed: list[Path] = []
    real_list_images: Callable[[Path], list[Path]] = YoloV6Parser._list_images

    def counting_list_images(image_dir: Path) -> list[Path]:
        listed.append(image_dir)
        return real_list_images(image_dir)

    monkeypatch.setattr(
        YoloV6Parser, "_list_images", staticmethod(counting_list_images)
    )

    layout = YoloV6Parser.detect(root)
    assert layout is not None
    records = _records(parser.parse(root, layout))
    enumerated = parser.enumerate_files(root, layout)

    assert listed == [root / "images" / "train", root / "images" / "test"]
    assert sorted(_files(records)) == sorted(
        written["train"] + written["test"]
    )
    assert enumerated is not None
    assert {
        split_name: sorted(files) for split_name, files in enumerated.items()
    } == {
        "train": sorted(written["train"]),
        "test": sorted(written["test"]),
    }


def test_yolov6_enumerated_files_are_the_paths_the_records_name(
    tmp_path: Path,
):
    """Guard the enumerated files against drifting away from the records.

    Count-based `split_ratios` pick their subset from `enumerate_files`
    before anything is imported, and the importer then keeps only the
    records naming a selected file. A file the records never name is a
    sample the import silently drops, so the enumeration has to stay the
    images the records name, per split, de-duplicated, in first-seen order.
    """
    root = tmp_path / "dataset"
    _dataset(root, splits=("train", "valid", "test"), images=5)
    parser = _plugin(YoloV6Parser)

    layout = YoloV6Parser.detect(root)
    assert layout is not None
    enumerated = parser.enumerate_files(root, layout)

    named: dict[str | None, dict[Path, None]] = {}
    for split_name, record in _split_records(parser.parse(root, layout)):
        named.setdefault(split_name, {})[Path(record["file"])] = None

    assert enumerated == {
        split_name: list(files) for split_name, files in named.items()
    }


def test_yolov6_records_ignore_edits_to_the_enumerated_files(tmp_path: Path):
    """Guard the enumeration against being the parse's own image list.

    The enumerated files and the images the records are streamed from come
    from a single listing, kept in the layout; handing that very list out
    would let a caller trimming a split to a count silently drop the
    records of the images it removed.
    """
    root = tmp_path / "dataset"
    _dataset(root, splits=("train",), images=4)
    parser = _plugin(YoloV6Parser)

    layout = YoloV6Parser.detect(root)
    assert layout is not None
    enumerated = parser.enumerate_files(root, layout)
    assert enumerated is not None
    enumerated["train"].clear()

    assert len(_records(parser.parse(root, layout))) == 4


def test_yolov6_split_lists_images_when_not_given(tmp_path: Path):
    """Guard the split parse against requiring the cached listing.

    The listing is an optimization handed over by `validate_split`; a
    caller assembling the arguments itself must still get the same records
    and the same files.
    """
    root = tmp_path / "dataset"
    written = _dataset(root, splits=("train",), images=4)
    parser = _plugin(YoloV6Parser)

    split_kwargs: dict[str, Any] = {
        "image_dir": root / "images" / "train",
        "annotation_dir": root / "labels" / "train",
        "classes_path": root / "data.yaml",
    }
    assert sorted(parser._split_files(**split_kwargs)) == sorted(
        written["train"]
    )
    assert len(_records(parser._split_records(**split_kwargs))) == 4


def test_yolov6_labels_dotted_image_names(tmp_path: Path):
    """Guard the label file name built from the image stem.

    Naming the label file after the image stem only matches replacing the
    image suffix because a stem drops exactly one suffix. An image called
    ``img.v1.2.jpg`` is labelled by ``img.v1.2.txt``, neither by ``img.txt``
    nor by ``img.v1.2.jpg.txt``.
    """
    root = tmp_path / "dataset"
    label_dir = root / "labels" / "train"
    label_dir.mkdir(parents=True)
    (root / "data.yaml").write_text("names: [bird]\n")
    _image(root / "images" / "train" / "img.v1.2.jpg")
    (label_dir / "img.v1.2.txt").write_text(LABEL)
    parser = _plugin(YoloV6Parser)

    records = _records(_parse(parser, root))
    assert [record["annotation"]["class"] for record in records] == ["bird"]


def test_yolov6_records_are_lazy(tmp_path: Path):
    """Guard the streaming contract.

    The records are an iterator over a dataset that is read while it is
    consumed: emptying the label files after the first record must be
    visible to the records that follow, which cannot happen if the parse
    was materialized up front.
    """
    root = tmp_path / "dataset"
    _dataset(root, splits=("train",), images=4, unlabeled=())
    parser = _plugin(YoloV6Parser)

    parsed = _parse(parser, root)
    records = cast(Iterator[tuple[str | None, dict[str, Any]]], parsed.records)
    assert isinstance(records, Iterator)

    split_name, first = next(records)
    assert split_name == "train"
    assert first["annotation"] is not None
    for label in (root / "labels" / "train").glob("*.txt"):
        label.write_text("")

    assert [record["annotation"] for _, record in records] == [
        None,
        None,
        None,
    ]
