import json
from pathlib import Path

from luxonis_ml.data.parsers.native_parser import NativeParser
from luxonis_ml.data.parsers.yolov4_parser import YoloV4Parser
from luxonis_ml.enums import DatasetType

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

    generator, _, added_images = NativeParser(
        dataset=None,  # type: ignore[arg-type]
        dataset_type=DatasetType.NATIVE,
        task_name=None,
    ).from_split(annotation_path=annotations_path)

    parsed_record = next(iter(generator))
    parsed_file = (
        parsed_record["file"]
        if isinstance(parsed_record, dict)
        else parsed_record.file
    )
    assert parsed_file == copied_image.resolve()
    assert added_images == [copied_image.resolve()]


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

    generator, _, _ = YoloV4Parser(
        dataset=None,  # type: ignore[arg-type]
        dataset_type=DatasetType.YOLOV4,
        task_name=None,
    ).from_split(
        image_dir=split_dir,
        annotation_path=annotations_path,
        classes_path=classes_path,
    )

    records = list(generator)
    files = {
        Path(
            record["file"] if isinstance(record, dict) else record.file
        ).resolve()
        for record in records
    }

    assert annotated_image.resolve() in files
    assert unlabeled_image.resolve() in files


def test_native_parser_resolves_array_annotation_paths(tempdir: Path):
    import numpy as np

    image_path = create_image(0, tempdir)
    split_dir = tempdir / "train"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "arrays").mkdir(parents=True)
    copied_image = split_dir / "images" / image_path.name
    copied_image.write_bytes(image_path.read_bytes())
    array_path = split_dir / "arrays" / "0.npy"
    np.save(array_path, np.zeros((4, 5), dtype=np.float32))

    annotations_path = split_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "file": f"images/{image_path.name}",
                    "task_name": "stereo",
                    "annotation": {
                        "class": "disparity",
                        "array": {"path": "arrays/0.npy"},
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    generator, _, _ = NativeParser(
        dataset=None,  # type: ignore[arg-type]
        dataset_type=DatasetType.NATIVE,
        task_name=None,
    ).from_split(annotation_path=annotations_path)

    record = next(iter(generator))
    assert isinstance(record, dict)
    assert record["annotation"]["array"]["path"] == array_path.resolve()


def test_native_parser_resolves_paths_for_a_list_of_annotations(
    tempdir: Path,
):
    image_path = create_image(0, tempdir)
    split_dir = tempdir / "train"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "masks").mkdir(parents=True)
    copied_image = split_dir / "images" / image_path.name
    copied_image.write_bytes(image_path.read_bytes())
    mask_paths = [split_dir / "masks" / f"{i}.png" for i in range(2)]
    for mask_path in mask_paths:
        mask_path.write_bytes(image_path.read_bytes())

    annotations_path = split_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "file": f"images/{image_path.name}",
                    "task_name": "task",
                    "annotation": [
                        {
                            "class": "class0",
                            "segmentation": {"mask": "masks/0.png"},
                        },
                        {"class": "class1"},
                        {
                            "class": "class2",
                            "segmentation": {"mask": "masks/1.png"},
                        },
                    ],
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    generator, _, _ = NativeParser(
        dataset=None,  # type: ignore[arg-type]
        dataset_type=DatasetType.NATIVE,
        task_name=None,
    ).from_split(annotation_path=annotations_path)

    record = next(iter(generator))
    assert isinstance(record, dict)
    resolved = [
        detection["segmentation"]["mask"]
        for detection in record["annotation"]
        if "segmentation" in detection
    ]
    assert resolved == [path.resolve() for path in mask_paths]


def test_native_parser_resolves_paths_inside_sub_detections(tempdir: Path):
    image_path = create_image(0, tempdir)
    split_dir = tempdir / "train"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "masks").mkdir(parents=True)
    copied_image = split_dir / "images" / image_path.name
    copied_image.write_bytes(image_path.read_bytes())
    mask_path = split_dir / "masks" / "head.png"
    mask_path.write_bytes(image_path.read_bytes())

    annotations_path = split_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "file": f"images/{image_path.name}",
                    "task_name": "person",
                    "annotation": {
                        "class": "person",
                        "sub_detections": {
                            "head": {
                                "class": "head",
                                "instance_segmentation": {
                                    "mask": "masks/head.png"
                                },
                            }
                        },
                    },
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    generator, _, _ = NativeParser(
        dataset=None,  # type: ignore[arg-type]
        dataset_type=DatasetType.NATIVE,
        task_name=None,
    ).from_split(annotation_path=annotations_path)

    record = next(iter(generator))
    assert isinstance(record, dict)
    nested = record["annotation"]["sub_detections"]["head"]
    assert nested["instance_segmentation"]["mask"] == mask_path.resolve()
