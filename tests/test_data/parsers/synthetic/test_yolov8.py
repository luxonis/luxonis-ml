"""YOLOv8 parser (detection, segmentation and keypoints)."""

import builtins
from pathlib import Path
from typing import (
    Any,
    cast,
)

import pytest
import yaml

from luxonis_ml.data import (
    LuxonisDataset,
    ParseIssueCollector,
    ParserIssue,
)
from luxonis_ml.data.parsers import (
    YOLOv8Parser,
)
from luxonis_ml.data.parsers import yolov8_parser as yolov8_module
from luxonis_ml.data.parsers.yolov8_parser import Format
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
    _write_yolov8_split,
)


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
    assert parser.detect(empty) is None

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
        parser._split_records(image_dir, annotation_dir, classes_path)
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
        parser._split_records(image_dir, annotation_dir, classes_path)
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
    # No image is decoded to start a parse, but an image a polygon needs
    # the size of must still fail the parse when it is started: failing
    # once the records are streaming would leave a half-imported dataset
    # behind.
    with pytest.raises(ValueError, match="Failed to read image"):
        parser._split_records(broken_images, broken_labels, broken_yaml)


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
        parser._split_records(image_dir, annotation_dir, classes_path)
    )
    keypoints = records[0]["annotation"]["keypoints"]["keypoints"]
    assert len(keypoints) == 2
    assert all(len(point) == 3 for point in keypoints)
    if kpt_dim == 2:
        assert all(point[2] == 2 for point in keypoints)


def test_yolov8_decodes_each_image_once_and_only_when_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard the single parse pass and the shared image decode.

    Regression: `files` was collected by running the whole generator once
    and discarding its records, and the polygon branch decoded the image
    again for every polygon. Three images with three polygons each were
    decoded eighteen times. Enumerating a split no longer parses it, and
    one decode serves every polygon of an image, so the same split must
    decode exactly three times — none of them before the records are
    consumed, the readability check that starts the parse included.
    """
    polygon = "0.1 0.2 0.8 0.2 0.8 0.9"
    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        tmp_path / "segmentation" / "train",
        dict.fromkeys(("a", "b", "c"), f"0 {polygon}\n" * 3),
    )

    decoded: list[str] = []
    real_imread = yolov8_module.cv2.imread

    def counting_imread(path: str, *args: Any, **kwargs: Any) -> Any:
        decoded.append(path)
        return real_imread(path, *args, **kwargs)

    monkeypatch.setattr(yolov8_module.cv2, "imread", counting_imread)

    parser = _plugin(YOLOv8Parser)
    files = parser._split_files(image_dir, annotation_dir, classes_path)
    assert len(files) == 3

    records = parser._split_records(image_dir, annotation_dir, classes_path)
    assert decoded == []

    collected = _records(records)

    assert len(collected) == 9
    assert sorted(decoded) == sorted(str(file) for file in files)


def test_yolov8_enumerates_without_parsing_and_streams_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard the lazy record stream and the reads each phase costs.

    Regression: the file list had to be final before a record was
    consumed, so the parser scanned every label file to predict which
    images would yield a record. The importer now collects the files from
    the records instead. Enumeration survives only for count-based
    `split_ratios`, which sample from it, so it must still agree with the
    records: an image whose only annotation line is malformed yields
    nothing and must not be listed. That costs one read of each label
    file. Starting a parse reads them again, only to fail up front on an
    image a polygon needs the size of; the annotations themselves are
    read once more and not before the records are consumed.
    """
    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        tmp_path / "mixed" / "train",
        {
            "labeled": "0 0.5 0.5 0.4 0.2\n",
            "blank": "\n\n",
            "malformed": "0 0.5 0.5\n",
        },
    )
    _image(image_dir / "background.jpg")

    opened: list[Path] = []
    real_open = builtins.open

    def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(file))
        return real_open(file, *args, **kwargs)

    def labels_opened() -> list[str]:
        return [path.name for path in opened if path.parent == annotation_dir]

    monkeypatch.setattr(builtins, "open", counting_open)

    issues = ParseIssueCollector()
    parser = YOLOv8Parser(issues)

    files = parser._split_files(image_dir, annotation_dir, classes_path)
    # `malformed` yields no record, so it is not among the files a
    # count-based split may sample.
    assert {file.stem for file in files} == {
        "labeled",
        "blank",
        "background",
    }
    assert sorted(labels_opened()) == [
        "blank.txt",
        "labeled.txt",
        "malformed.txt",
    ]

    opened.clear()
    records = parser._split_records(image_dir, annotation_dir, classes_path)

    # One read per label file buys the up-front readability check, and
    # nothing is parsed yet: a single-pass iterator that has not moved.
    assert sorted(labels_opened()) == [
        "blank.txt",
        "labeled.txt",
        "malformed.txt",
    ]
    assert iter(records) is records
    assert issues.messages == []

    opened.clear()
    collected = _records(records)

    # The annotations are read once, only now, and every image yields its
    # records in listing order — except the one whose only annotation
    # line is malformed, which yields nothing.
    assert sorted(labels_opened()) == [
        "blank.txt",
        "labeled.txt",
        "malformed.txt",
    ]
    assert [Path(record["file"]) for record in collected] == [
        file for file in files if file.stem != "malformed"
    ]
    assert [issue.parser_issue for issue in issues.messages] == [
        ParserIssue.MALFORMED_ANNOTATION
    ]


def test_yolov8_well_formed_annotations_build_no_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard the array-free record paths.

    Regression: every polygon was reshaped into an array and reduced four
    times to fit its bounding box, and every keypoint line was reshaped
    as well. Well-formed annotations are now grouped by striding the
    parsed values, so parsing them must not allocate a single array;
    numpy is reached only for shapes the fast paths refuse.
    """
    arrays: list[object] = []
    real_array = yolov8_module.np.array

    def counting_array(*args: Any, **kwargs: Any) -> Any:
        arrays.append(args)
        return real_array(*args, **kwargs)

    monkeypatch.setattr(yolov8_module.np, "array", counting_array)

    parser = _plugin(YOLOv8Parser)
    cases: list[tuple[str, str, list[int] | None]] = [
        ("detection", "0 0.5 0.5 0.4 0.2\n", None),
        ("segmentation", "0 0.1 0.2 0.8 0.2 0.8 0.9\n", None),
        ("keypoints-2d", "0 0.5 0.5 0.4 0.2 0.1 0.2 0.3 0.4\n", [2, 2]),
        ("keypoints-3d", "0 0.5 0.5 0.4 0.2 0.1 0.2 1 0.3 0.4 2\n", [2, 3]),
    ]
    for name, annotation, kpt_shape in cases:
        image_dir, annotation_dir, classes_path = _write_yolo8_split(
            tmp_path / name / "train",
            {"image": annotation},
            kpt_shape=kpt_shape,
        )
        records = _records(
            parser._split_records(image_dir, annotation_dir, classes_path)
        )
        assert len(records) == 1, name

    assert arrays == []


def test_yolov8_shapes_the_fast_paths_refuse_fall_back_to_numpy(
    tmp_path: Path,
):
    """Guard the numpy fallbacks the fast paths are guarded by.

    Regression: striding the parsed values accepts counts that reshaping
    rejected, and it cannot express a `kpt_shape` wider than three. A
    polygon with an odd number of coordinates and a keypoint line that
    does not fill `kpt_shape` must still raise the reshape error instead
    of silently dropping a value, and a four-valued `kpt_shape` must
    still keep only the first three values of each keypoint.
    """
    parser = _plugin(YOLOv8Parser)

    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        tmp_path / "odd-polygon" / "train",
        {"image": "0 0.1 0.2 0.8 0.2 0.8\n"},
    )
    with pytest.raises(ValueError, match="reshape"):
        _records(
            parser._split_records(image_dir, annotation_dir, classes_path)
        )

    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        tmp_path / "short-keypoints" / "train",
        {"image": "0 0.5 0.5 0.4 0.2 0.1 0.2 1 0.3\n"},
        kpt_shape=[2, 3],
    )
    with pytest.raises(ValueError, match="reshape"):
        _records(
            parser._split_records(image_dir, annotation_dir, classes_path)
        )

    image_dir, annotation_dir, classes_path = _write_yolo8_split(
        tmp_path / "wide-keypoints" / "train",
        {"image": "0 0.5 0.5 0.4 0.2 0.1 0.2 1 9 0.3 0.4 2 9\n"},
        kpt_shape=[2, 4],
    )
    records = _records(
        parser._split_records(image_dir, annotation_dir, classes_path)
    )
    assert records[0]["annotation"]["keypoints"]["keypoints"] == [
        (0.1, 0.2, 1),
        (0.3, 0.4, 2),
    ]


def test_yolov8_detects_both_directory_layouts(tmp_path: Path):
    """Detection recognizes both layouts and hands parsing the result."""
    roboflow = tmp_path / "roboflow"
    _write_yolo8_split(roboflow / "train", {"image": ""})
    layout = YOLOv8Parser.detect(roboflow)
    assert layout is not None
    assert layout.split_names == ["train"]

    # The layout detection produced is what `parse` streams from, so the
    # source is inspected once and every record carries its split.
    parser = _plugin(YOLOv8Parser)
    image = roboflow / "train" / "images" / "image.jpg"
    assert _split_records(parser.parse(roboflow, layout)) == [
        ("train", {"file": str(image), "annotation": None})
    ]

    ultralytics = tmp_path / "ultralytics"
    image_dir = ultralytics / "images" / "val"
    labels_dir = ultralytics / "labels" / "val"
    image_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    _image(image_dir / "image.jpg")
    (ultralytics / "data.yml").write_text("names: [bird]\n")
    layout = YOLOv8Parser.detect(ultralytics)
    assert layout is not None
    assert layout.split_names == ["val"]


def test_yolov8_enumerated_files_exclude_images_that_yield_nothing(
    tmp_path: Path,
):
    """`enumerate_files` must agree with what the records name.

    Regression: `_split_files` returned the plain image listing, but an
    image whose annotation lines are all too short to parse yields no
    record at all. A count-based `split_ratios` samples from this list, so
    over-reporting made it pick an image that contributed nothing and
    leave that split one sample short.
    """
    split = tmp_path / "train"
    images = split / "images"
    labels = split / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names:\n  0: bird\n")

    for name in ("good", "background", "malformed"):
        _image(images / f"{name}.jpg")
    (labels / "good.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    # No label file for `background`: it still yields one empty record.
    (labels / "malformed.txt").write_text("0 0.5\n\n0 0.1\n")

    parser = _plugin(YOLOv8Parser)
    kwargs = {
        "image_dir": images,
        "annotation_dir": labels,
        "classes_path": tmp_path / "data.yaml",
    }
    enumerated = parser._split_files(**kwargs)
    named = list(
        dict.fromkeys(
            Path(record["file"])
            for record in _records(parser._split_records(**kwargs))
        )
    )

    assert enumerated == named
    assert images / "malformed.jpg" not in enumerated
    assert {path.name for path in enumerated} == {"good.jpg", "background.jpg"}
