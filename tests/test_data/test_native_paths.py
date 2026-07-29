import json
from pathlib import Path

from luxonis_ml.data import ParseIssueCollector
from luxonis_ml.data.parsers.native_parser import NativeParser
from luxonis_ml.data.parsers.yolov4_parser import YoloV4Parser

from .utils import create_image


def test_native_parser_accepts_windows_style_file_paths(tempdir: Path):
    image_path = create_image(0, tempdir)
    split_dir = tempdir / "train"
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True)
    copied_image = image_dir / image_path.name
    copied_image.write_bytes(image_path.read_bytes())

    annotations_path = split_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "file": f"images\\{image_path.name}",
                    "task_name": "task",
                    "annotation": {
                        "class": "class0",
                        "boundingbox": {
                            "x": 0.1,
                            "y": 0.2,
                            "w": 0.3,
                            "h": 0.4,
                        },
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    parsed = NativeParser(ParseIssueCollector()).parse(
        split_dir,
        dataset_type="native",
    )

    parsed_record = next(iter(parsed.records))
    parsed_file = (
        parsed_record["file"]
        if isinstance(parsed_record, dict)
        else parsed_record.file
    )
    assert parsed_file == copied_image.resolve()
    assert parsed.files == [copied_image.resolve()]


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

    parsed = YoloV4Parser(ParseIssueCollector()).parse(
        split_dir,
        dataset_type="yolov4",
    )

    records = list(parsed.records)
    files = {
        Path(
            record["file"] if isinstance(record, dict) else record.file
        ).resolve()
        for record in records
    }

    assert annotated_image.resolve() in files
    assert unlabeled_image.resolve() in files
