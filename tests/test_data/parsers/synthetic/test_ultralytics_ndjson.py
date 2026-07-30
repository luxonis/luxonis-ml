"""Ultralytics NDJSON parser."""

import json
from collections.abc import (
    Callable,
    Iterator,
    Sequence,
)
from pathlib import Path, PurePath
from typing import (
    Any,
)

import numpy as np
import pytest

from luxonis_ml.data import (
    Layout,
    LuxonisDataset,
    ParseIssueCollector,
    ParseResult,
    ParserIssue,
)
from luxonis_ml.data.parsers import (
    UltralyticsNDJSONParser,
)
from luxonis_ml.data.parsers.parser_plugin import (
    get_parser_plugin,
)
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
)
from tests.test_data.utils import create_image

SplitRecords = list[tuple[str | None, Any]]


def _detect(source: Path) -> Layout:
    """Return the layout the parser recognizes for ``source``."""
    layout = UltralyticsNDJSONParser.detect(source)
    assert layout is not None
    return layout


def _parse(
    parser: UltralyticsNDJSONParser,
    source: Path,
    **kwargs: Any,
) -> ParseResult:
    """Parse ``source`` with the layout detection found for it."""
    return parser.parse(source, _detect(source), **kwargs)


def _files(parsed: SplitRecords) -> list[str]:
    """Collect the files of a parse the way the importer collects them."""
    files: dict[str, None] = {}
    for _, record in parsed:
        files[record["file"]] = None
    return list(files)


def _split_files(parsed: SplitRecords) -> dict[str | None, list[str]]:
    """Collect each split's files as the records stream past."""
    split_files: dict[str | None, dict[str, None]] = {}
    for split_name, record in parsed:
        split_files.setdefault(split_name, {})[record["file"]] = None
    return {name: list(files) for name, files in split_files.items()}


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
    """A dropped record must not take the ones that follow with it.

    The parser walks the manifest once and tags every record with the
    split its image record names, so the importer collects the files —
    and each split's files — as the records stream past. A record whose
    image is missing, and a line that is not an image at all, must be
    dropped without disturbing the records after them: the split tag of
    every surviving record still has to be the one its own line names.
    """
    ndjson_path = tempdir / "ndjson_alignment" / "dataset.ndjson"
    ndjson_path.parent.mkdir(parents=True)
    _write_ultralytics_ndjson(
        ndjson_path,
        ndjson_path.parent / "images",
        missing_image="img_missing.jpg",
    )

    issues = ParseIssueCollector()
    parsed = _split_records(
        _parse(UltralyticsNDJSONParser(issues), ndjson_path)
    )

    assert [Path(file).name for file in _files(parsed)] == [
        "img_0.jpg",
        "img_1.jpg",
        "img_2.jpg",
    ]
    assert {
        name: [Path(file).name for file in files]
        for name, files in _split_files(parsed).items()
    } == {"train": ["img_0.jpg"], "val": ["img_1.jpg"], "test": ["img_2.jpg"]}

    assert [
        (Path(record["file"]).name, (record["annotation"] or {}).get("class"))
        for _, record in parsed
    ] == [
        ("img_0.jpg", "budgie"),
        ("img_1.jpg", None),
        ("img_2.jpg", "parrot"),
    ]
    assert [issue.parser_issue for issue in issues.messages] == [
        ParserIssue.MISSING_IMAGE
    ]


def test_ultralytics_ndjson_detects_the_splits_the_format_carries(
    tempdir: Path,
):
    """Detection must report splits without reading the image records.

    The manifest assigns every image its own split, so the source has
    original splits and an import must preserve them instead of
    reshuffling the images at random. Detection cannot name the splits
    that are actually used without walking the whole file — the walk
    parsing makes — so it claims the ones the format defines and the
    records carry the assignment. The header is the only part read, and
    the manifest it resolved rides along so parsing does not look for it
    again.
    """
    ndjson_path = tempdir / "ndjson_detect" / "dataset.ndjson"
    ndjson_path.parent.mkdir(parents=True)
    _write_ultralytics_ndjson(
        ndjson_path,
        ndjson_path.parent / "images",
        missing_image="img_missing.jpg",
    )

    layout = _detect(ndjson_path.parent)

    assert layout.split_names == ["train", "val", "test"]
    for split_kwargs in layout.splits.values():
        assert split_kwargs["ndjson_path"] == ndjson_path.resolve()
        assert split_kwargs["header"]["class_names"] == ["budgie", "parrot"]


def test_ultralytics_ndjson_import_preserves_the_manifest_splits(
    dataset_name: str,
    tempdir: Path,
):
    """The splits written in the manifest must survive the import.

    Regression guard for the layout: an import keeps the original splits
    only when the parser announces that the source has them, and the
    images land in the right one only because every record carries its
    own split tag. The three splits are deliberately of different sizes,
    which the random split the importer falls back to would not
    reproduce.
    """
    image_dir = tempdir / "ndjson_import_splits" / "images"
    image_dir.mkdir(parents=True)
    sizes = {"train": 5, "valid": 3, "test": 2}
    lines: list[dict[str, Any]] = [
        {"type": "dataset", "class_names": ["budgie"]}
    ]
    index = 0
    for split_name, size in sizes.items():
        for _ in range(size):
            create_image(index, image_dir)
            lines.append(
                {
                    "type": "image",
                    "file": f"images/img_{index}.jpg",
                    "split": split_name,
                    "width": 512,
                    "height": 512,
                    "annotations": {"boxes": [[0, 0.5, 0.5, 0.2, 0.2]]},
                }
            )
            index += 1
    ndjson_path = image_dir.parent / "dataset.ndjson"
    ndjson_path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )

    dataset = LuxonisDataset.import_dataset(
        str(ndjson_path),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        splits = dataset.get_splits()
        assert splits is not None
        assert {name: len(files) for name, files in splits.items()} == {
            "train": 5,
            "val": 3,
            "test": 2,
        }
    finally:
        dataset.delete_dataset(delete_local=True)


def test_ultralytics_ndjson_records_are_streamed(
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The manifest must be walked once, and only on consumption.

    Regression: the whole file was decoded into an in-memory list of
    records before anything was yielded, so a multi-gigabyte export —
    every ``segments`` polygon list and ``pose`` array held alive at
    once — died of ``MemoryError`` before a single record reached
    ``dataset.add``. Counting walks of the file is what distinguishes
    streaming from materializing: a materialized parser walks it once,
    up front, and never again.

    The old contract needed a second walk of its own, because the file
    list it had to publish before the records were consumed could only
    be built by resolving every image path first. Nothing is published
    up front any more, so one walk answers both.
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

    parser = _plugin(UltralyticsNDJSONParser)
    parsed = parser.parse(ndjson_path, _detect(ndjson_path))
    assert walks == [], "nothing is read before the first record is pulled"

    records = parsed.records
    first = next(records)
    assert len(walks) == 1
    assert len([first, *records]) == 3
    assert len(walks) == 1, "one walk answers both the records and the files"


def _write_ndjson(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n" + "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _count_calls(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *names: str,
) -> list[str]:
    """Record calls to ``names`` on ``module`` without suppressing them.

    The wrappers delegate to the originals so that a parser which lost an
    optimization still produces correct records and fails on the call
    count rather than on a broken stub.
    """
    calls: list[str] = []

    def wrap(name: str) -> Callable[..., Any]:
        original = getattr(module, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            return original(*args, **kwargs)

        return wrapper

    for name in names:
        monkeypatch.setattr(module, name, wrap(name))
    return calls


def test_ultralytics_ndjson_fits_polygon_boxes_in_one_pass_per_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Polygon bounding boxes must not be reduced one corner at a time.

    Regression: `_fit_boundingbox` called `np.min`/`np.max` once per
    corner on a column slice, so every polygon paid for four column
    slices and four NumPy dispatches where two whole-array reductions
    say the same thing — `min` and `max` select an element instead of
    computing one, so reducing both columns at once yields the identical
    floats. It was the single most expensive step of the segmentation
    branch. Counting the column-wise calls is what distinguishes the two:
    the reduced form makes none of them.
    """
    parser = _plugin(UltralyticsNDJSONParser)
    image = _image(tmp_path / "segment.jpg")
    column_reductions = _count_calls(monkeypatch, np, "min", "max")

    ndjson = _write_ndjson(
        tmp_path / "segments.ndjson",
        [
            {"type": "dataset", "class_names": ["bird"]},
            {
                "type": "image",
                "file": image.name,
                "split": "train",
                "width": 20,
                "height": 10,
                "annotations": {
                    "segments": [[0, 0.1, 0.2, 0.8, 0.2, 0.8, 0.9]]
                },
            },
        ],
    )
    records = _records(_parse(parser, ndjson))

    assert column_reductions == []
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


def test_ultralytics_ndjson_two_column_keypoints_skip_the_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A ``[n, 2]`` ``kpt_shape`` must not build a visibility column.

    Regression: a 2-column pose layout carries no visibility flag, so the
    parser concatenated a column of ``2.0`` onto the keypoint array only
    for the very next line to cast it back to the literal ``2``. That is
    two array allocations and a copy of every coordinate per annotation,
    for a constant. Counting the NumPy calls is what distinguishes
    emitting the constant from rebuilding it.
    """
    parser = _plugin(UltralyticsNDJSONParser)
    image = _image(tmp_path / "pose.jpg")
    padding_calls = _count_calls(monkeypatch, np, "ones", "concatenate")

    ndjson = _write_ndjson(
        tmp_path / "pose-2d.ndjson",
        [
            {
                "type": "dataset",
                "class_names": ["bird"],
                "kpt_shape": [3, 2],
            },
            {
                "type": "image",
                "file": image.name,
                "split": "train",
                "width": 20,
                "height": 10,
                "annotations": {
                    "pose": [
                        [0, 0.5, 0.5, 0.4, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                    ]
                },
            },
        ],
    )
    records = _records(_parse(parser, ndjson))

    assert padding_calls == []
    assert records[0]["annotation"]["keypoints"]["keypoints"] == [
        (0.1, 0.2, 2),
        (0.3, 0.4, 2),
        (0.5, 0.6, 2),
    ]


def _count_image_conversions(
    ndjson: Path,
    images: Sequence[Path],
    expected_records: int,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    """Count how often parsing ``ndjson`` stringifies an image path."""
    parser = _plugin(UltralyticsNDJSONParser)
    parsed = parser.parse(ndjson, _detect(ndjson))

    original = PurePath.__str__
    stringified: list[PurePath] = []

    def counting_str(self: PurePath) -> str:
        # Appended unconditionally and filtered afterwards: testing which
        # path this is would hash it, and hashing a path stringifies it.
        stringified.append(self)
        return original(self)

    monkeypatch.setattr(PurePath, "__str__", counting_str)
    try:
        records = list(parsed.records)
    finally:
        monkeypatch.undo()

    assert len(records) == expected_records
    wanted = set(images)
    return len([path for path in stringified if path in wanted])


def test_ultralytics_ndjson_stringifies_each_image_path_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The image path must be stringified per image, not per annotation.

    Regression: every yielded record built its ``file`` value with its
    own ``str(image_path)`` call, so an image carrying thirty instances —
    an ordinary Ultralytics export mixes boxes, polygons and poses on the
    same image — converted the same path thirty times. The path belongs
    to the image and cannot change between its annotations.

    Parsing a path now also resolves it, which converts it a fixed number
    of times per image record; what must not happen is the count growing
    with the annotations on that image. Two manifests over the same
    images, one carrying six times the annotations of the other, are what
    distinguishes hoisting the conversion from repeating it.
    """
    images = [_image(tmp_path / f"img_{index}.jpg") for index in range(3)]
    box = [0, 0.5, 0.5, 0.4, 0.2]

    def manifest(name: str, boxes: list[list[float]]) -> Path:
        return _write_ndjson(
            tmp_path / name,
            [
                {"type": "dataset", "class_names": ["bird"]},
                *(
                    {
                        "type": "image",
                        "file": image.name,
                        "split": "train",
                        "width": 20,
                        "height": 10,
                        "annotations": {"boxes": boxes},
                    }
                    for image in images
                ),
            ],
        )

    one_box = _count_image_conversions(
        manifest("one-box.ndjson", [box]),
        images,
        expected_records=len(images),
        monkeypatch=monkeypatch,
    )
    many_boxes = _count_image_conversions(
        manifest("many-boxes.ndjson", [box] * 6),
        images,
        expected_records=len(images) * 6,
        monkeypatch=monkeypatch,
    )

    assert one_box == many_boxes
    # Sanity check that image paths are converted at all, so that a
    # parser converting none of them cannot pass on equality alone.
    assert one_box >= len(images)


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

    assert UltralyticsNDJSONParser.detect(ndjson) is not None
    assert UltralyticsNDJSONParser.detect(tmp_path) is not None
    parsed = _split_records(_parse(parser, tmp_path))
    records = [record for _, record in parsed]

    assert len(records) == 5
    assert [Path(file) for file in _files(parsed)] == [
        box_image,
        segment_image,
        pose_image,
        empty_image,
    ]
    assert {
        name: [Path(file) for file in files]
        for name, files in _split_files(parsed).items()
    } == {
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
    records = _records(_parse(parser, valid))
    assert records[0]["annotation"]["keypoints"]["keypoints"] == [
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
    with pytest.raises(ValueError, match="dimensionality is not inferable"):
        _records(_parse(parser, invalid))

    # A source the parser cannot read is rejected before an import can
    # start, which is where the check moved to now that detection is
    # what recognizes a source.
    wrong_suffix = tmp_path / "dataset.txt"
    wrong_suffix.write_text("x")
    assert UltralyticsNDJSONParser.detect(wrong_suffix) is None
    with pytest.raises(ValueError, match="not in the expected format"):
        get_parser_plugin(wrong_suffix, "ultralytics-ndjson")

    invalid_header = _write_ndjson(
        tmp_path / "invalid-header.ndjson",
        [{"type": "dataset", "class_names": ["bird"]}],
    )
    assert UltralyticsNDJSONParser.detect(invalid_header) is None
    with pytest.raises(ValueError, match="not in the expected format"):
        get_parser_plugin(invalid_header, "ultralytics-ndjson")


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
    parsed = _split_records(_parse(parser, ndjson))
    assert len(parsed) == 1
    assert parsed[0][0] == "val"
    assert destinations[0].parent.name == "val"
    assert destinations[0].suffix == ".png"

    # The download directory is checked before the record that would
    # reuse it is emitted, so an import that must not reuse it fails
    # before a single record is written.
    records = parser.parse(
        ndjson,
        _detect(ndjson),
        reuse_cached=False,
    ).records
    with pytest.raises(ValueError, match="already exists"):
        next(records)
    assert _records(_parse(parser, ndjson, reuse_cached=True))


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


def test_ultralytics_ndjson_counts_do_not_pre_download_the_images(
    dataset_name: str,
    tempdir: Path,
):
    """Count-based splits must not make the parse trip over its own cache.

    Regression: the parser did not answer ``enumerate_files``, so counting
    the files fell back to a throwaway parse — which downloaded every remote
    image and created the cache directory. The real parse that followed then
    found that directory and refused it, so a first-ever import failed with
    "Remote NDJSON image directory already exists". Enumeration now names the
    downloads without fetching them.
    """
    source_dir = tempdir / "sources"
    source_dir.mkdir()
    ndjson_path = tempdir / "budgie.ndjson"
    lines = [json.dumps({"type": "dataset", "class_names": {"0": "budgie"}})]
    for index, split_name in enumerate(
        ["train", "train", "train", "train", "val", "test"]
    ):
        source = create_image(index, source_dir)
        lines.append(
            json.dumps(
                {
                    "type": "image",
                    "file": f"{split_name}/img{index}.jpg",
                    "url": source.resolve().as_uri(),
                    "split": split_name,
                    "width": 512,
                    "height": 512,
                    "annotations": {"boxes": [[0, 0.5, 0.5, 0.4, 0.4]]},
                }
            )
        )
    ndjson_path.write_text("\n".join(lines), encoding="utf-8")

    remote_image_dir = tempdir / "budgie"
    assert not remote_image_dir.exists()

    dataset = LuxonisDataset.import_dataset(
        str(ndjson_path),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
        split_ratios={"train": 2, "val": 1, "test": 1},
        parser_kwargs={"reuse_cached": False},
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
