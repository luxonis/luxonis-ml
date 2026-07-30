"""Pascal VOC parser."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from defusedxml.ElementTree import fromstring

from luxonis_ml.data.parsers import (
    VOCParser,
)
from luxonis_ml.data.parsers import voc_parser as voc_parser_module
from luxonis_ml.data.parsers.parser_plugin import ParserPlugin
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
)


def _voc_xml(
    filename: str,
    *,
    with_bbox: bool = True,
    size: tuple[int, int] = (20, 10),
    box: tuple[int, int, int, int] = (2, 1, 12, 6),
) -> str:
    width, height = size
    xmin, ymin, xmax, ymax = box
    bbox = (
        f"<bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin>"
        f"<xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>"
        if with_bbox
        else ""
    )
    return (
        "<annotation>"
        f"<filename>{filename}</filename>"
        f"<size><width>{width}</width><height>{height}</height></size>"
        f"<object><name>bird</name>{bbox}</object>"
        "</annotation>"
    )


def _voc_split(root: Path, images: int) -> Path:
    """Write a split of ``images`` images, each with one boxed object.

    Args:
        root: Directory the split is created in.
        images: Number of images to write.

    Returns:
        The created split directory.

    """
    split = root / "train"
    split.mkdir(parents=True, exist_ok=True)
    for index in range(images):
        image = _image(split / f"img_{index}.jpg", size=(20, 10))
        (split / f"img_{index}.xml").write_text(_voc_xml(image.name))
    return split


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
    records = _records(parser._split_records(**kwargs))

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
    _records(parser._split_records(split, split))
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
        _records(parser._split_records(split, split))

    with pytest.raises(ValueError, match="Could not find missing"):
        parser._xml_find(fromstring("<root />"), "missing")


def test_voc_records_stream_lazily_from_a_single_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parse must read nothing until a record is asked for.

    The parser used to publish a file list next to its records, which
    forced the whole split to be read before the first record could be
    consumed - and, before that, forced a second pass that rebuilt every
    record only to read its path back out. The importer now collects the
    files from the records themselves, so the annotations must be read
    one at a time as the records are pulled, each record must carry the
    split it belongs to, and the stream stays one-shot.
    """
    split = _voc_split(tmp_path, 3)
    opened: list[Path] = []
    real_parse = voc_parser_module.parse

    def counting_parse(source: Path) -> Any:
        opened.append(Path(source))
        return real_parse(source)

    monkeypatch.setattr(voc_parser_module, "parse", counting_parse)

    layout = VOCParser.detect(tmp_path)
    assert layout is not None
    assert layout.split_names == ["train"]

    result = _plugin(VOCParser).parse(tmp_path, layout)
    assert isinstance(result.records, Iterator)
    assert opened == [], "the source was walked before a record was asked for"

    first = next(result.records)
    assert len(opened) == 1, "more was read than the first record needed"

    streamed = cast(
        "list[tuple[str | None, dict[str, Any]]]",
        [first, *result.records],
    )
    assert len(opened) == 3

    assert [split_name for split_name, _ in streamed] == ["train"] * 3
    # One boxed object per image, so there is exactly one record per image.
    assert {Path(record["file"]) for _, record in streamed} == {
        split.resolve() / f"img_{index}.jpg" for index in range(3)
    }
    assert list(result.records) == []


def test_voc_does_not_enumerate_files_without_parsing(
    tmp_path: Path,
) -> None:
    """A directory listing must not stand in for a parse.

    A split's images are the ones its annotations name in ``<filename>``,
    and an annotation naming an image that is not there is skipped. A
    listing of the split directory would therefore report images that no
    record ever mentions, and count-based ``split_ratios`` would select
    them and import nothing for them. Answering ``None`` keeps the
    importer on its fallback parse.
    """
    split = tmp_path / "train"
    split.mkdir()
    _image(split / "img_0.jpg", size=(20, 10))
    (split / "img_0.xml").write_text(_voc_xml("gone.jpg"))
    _image(split / "img_1.jpg", size=(20, 10))
    (split / "img_1.xml").write_text(_voc_xml("img_1.jpg"))

    layout = VOCParser.detect(tmp_path)
    assert layout is not None
    parser = _plugin(VOCParser)
    assert parser.enumerate_files(tmp_path, layout) is None

    records = _records(parser.parse(tmp_path, layout))
    assert {Path(record["file"]) for record in records} == {
        split.resolve() / "img_1.jpg"
    }


def test_voc_reads_each_annotation_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``.xml`` file must be opened and parsed exactly once.

    Recovering the file list with a separate pass over the annotation
    directory would parse each document twice, which is the single most
    expensive thing this parser does.
    """
    split = _voc_split(tmp_path, 4)
    parsed_files: list[Path] = []
    real_parse = voc_parser_module.parse

    def counting_parse(source: Path) -> Any:
        parsed_files.append(Path(source))
        return real_parse(source)

    monkeypatch.setattr(voc_parser_module, "parse", counting_parse)

    _records(_plugin(VOCParser)._split_records(split, split))

    assert len(parsed_files) == 4
    assert len(set(parsed_files)) == 4


def test_voc_resolves_the_image_directory_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The image directory must be resolved once per split.

    ``image_dir.absolute().resolve()`` used to run inside the annotation
    loop, so every annotation paid for a full symlink resolution of a
    directory that never changes.
    """
    split = _voc_split(tmp_path, 5)
    calls = 0
    real_absolute = Path.absolute

    def counting_absolute(self: Path) -> Path:
        nonlocal calls
        if self == split:
            calls += 1
        return real_absolute(self)

    monkeypatch.setattr(Path, "absolute", counting_absolute)

    _records(_plugin(VOCParser)._split_records(split, split))

    assert calls == 1


def test_voc_still_reads_the_name_of_a_boxless_object(
    tmp_path: Path,
) -> None:
    """A classification-style object without a name must still raise.

    Every ``<object>`` used to have its ``<name>`` appended to a per-image
    ``classes`` list that nothing ever read. Dropping the list must not
    drop the lookup with it: an object with no ``<bndbox>`` produces no
    record, but a missing ``<name>`` has always been a hard error and stays
    one.
    """
    split = tmp_path / "train"
    split.mkdir()
    image = _image(split / "nameless.jpg", size=(20, 10))
    (split / "nameless.xml").write_text(
        "<annotation>"
        f"<filename>{image.name}</filename>"
        "<size><width>20</width><height>10</height></size>"
        "<object><pose>Unspecified</pose></object>"
        "</annotation>"
    )

    with pytest.raises(ValueError, match="Could not find name"):
        _records(_plugin(VOCParser)._split_records(split, split))


def test_voc_bbox_normalization_matches_the_numpy_original(
    tmp_path: Path,
) -> None:
    """Normalized boxes must stay bit-identical to the array version.

    The corners used to be normalized by dividing a ``float64`` array in
    place, one array per box. They are divided as scalars now, which is the
    same IEEE-754 operation - but only as long as nobody replaces a
    division by a multiplication with a precomputed reciprocal, which this
    test would catch on the deliberately awkward width and height below.
    """
    assert not hasattr(voc_parser_module, "np"), (
        "the parser allocates a numpy array per bounding box again"
    )

    split = tmp_path / "train"
    split.mkdir()
    image = _image(split / "odd.jpg", size=(3, 7))
    (split / "odd.xml").write_text(
        _voc_xml(image.name, size=(3, 7), box=(1, 2, 2, 5))
    )

    (record,) = _records(_plugin(VOCParser)._split_records(split, split))

    expected = np.array([1.0, 2.0, 2.0 - 1.0, 5.0 - 2.0])
    expected[::2] /= 3.0
    expected[1::2] /= 7.0
    x, y, width, height = expected.tolist()
    boundingbox = record["annotation"]["boundingbox"]
    assert (
        boundingbox["x"],
        boundingbox["y"],
        boundingbox["w"],
        boundingbox["h"],
    ) == (x, y, width, height)


def test_voc_validate_split_compares_stems_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split validation must not re-wrap its listings in new ``Path``s.

    ``_compare_stem_files`` builds a fresh ``Path`` for every entry of both
    listings before taking its stem, which doubles the cost of validating a
    large split - and validation is what recognizing a VOC dataset at all
    consists of, so every candidate directory pays it.
    """
    split = _voc_split(tmp_path, 2)

    def _rewrapped(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("the directory listings were re-wrapped")

    monkeypatch.setattr(
        ParserPlugin, "_compare_stem_files", staticmethod(_rewrapped)
    )

    assert VOCParser.validate_split(split) == {
        "image_dir": split,
        "annotation_dir": split,
    }
    assert VOCParser.validate_split(tmp_path / "absent") is None

    (split / "img_0.xml").unlink()
    assert VOCParser.validate_split(split) is None
