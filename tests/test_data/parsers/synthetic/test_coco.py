"""COCO parser."""

import json
import tracemalloc
from pathlib import Path
from typing import (
    Any,
    TextIO,
    cast,
)

import pycocotools.mask as mask_util
import pytest

from luxonis_ml.data import (
    LuxonisDataset,
)
from luxonis_ml.data.parsers import (
    COCOParser,
)
from luxonis_ml.data.parsers import coco_parser as coco_parser_module
from luxonis_ml.data.parsers.coco_parser import clean_annotations
from luxonis_ml.data.utils.enums import COCOFormat
from tests.test_data.parsers.helpers import (
    _image,
    _plugin,
    _records,
    _split_records,
)
from tests.test_data.utils import create_image


def _parse(
    parser: COCOParser, source: Path, **kwargs: Any
) -> list[tuple[str | None, Any]]:
    """Detect ``source`` and collect the records `parse` tags.

    The layout detection returns is handed straight to `parse`, which is
    how a source is inspected once per import.
    """
    layout = COCOParser.detect(source)
    assert layout is not None
    return _split_records(parser.parse(source, layout, **kwargs))


def _files_by_split(
    tagged: list[tuple[str | None, Any]],
) -> dict[str | None, list[Path]]:
    """Collect the files of each split the way the importer does."""
    files: dict[str | None, dict[Path, None]] = {}
    for split_name, record in tagged:
        files.setdefault(split_name, {})[record["file"]] = None
    return {name: list(paths) for name, paths in files.items()}


def test_coco_count_splits_do_not_create_synthetic_test_split(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "coco_counts"
    for split in ("train", "validation"):
        data_dir = dataset_dir / split / "data"
        data_dir.mkdir(parents=True)
        image = create_image(0, data_dir)
        (dataset_dir / split / "labels.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": image.name,
                            "width": 640,
                            "height": 480,
                        }
                    ],
                    "annotations": [],
                    "categories": [],
                }
            )
        )
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir()
    (raw_dir / "person_keypoints.json").write_text("{}")

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        split_ratios={"train": 1, "val": 1, "test": 1},
        delete_local=True,
    )
    try:
        assert len(dataset) == 2
        splits = dataset.get_splits()
        assert splits is not None
        assert {name: len(ids) for name, ids in splits.items()} == {
            "train": 1,
            "val": 1,
            "test": 0,
        }
    finally:
        dataset.delete_dataset(delete_local=True)


def test_partial_split_train_only_roboflow_coco_keeps_format_detection(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "coco_train_only_roboflow"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True)
    create_image(16, train_dir)
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "img_16.jpg",
                        "width": 512,
                        "height": 512,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [128, 128, 256, 256],
                        "area": 65536,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 0, "name": "budgie"}],
            }
        )
    )

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="coco",
        delete_local=True,
        save_dir=tempdir,
        parser_kwargs={"use_keypoint_ann": True},
    )

    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 0
    dataset.delete_dataset(delete_local=True)


def test_coco_single_split_rejects_keypoint_options(
    dataset_name: str,
    tempdir: Path,
):
    """Keypoint options must not be silently dropped for a single split.

    Regression: for a source without split directories, ``COCOParser.parse``
    delegated through ``**kwargs``, but ``use_keypoint_ann`` and
    ``keypoint_ann_paths`` are bound to named parameters and never reach it.
    They vanished without a warning, so a pose model could be trained on a
    dataset holding no keypoints at all. Previously this raised ``TypeError``.
    """
    dataset_dir = tempdir / "coco_single_split"
    image_dir = dataset_dir / "data"
    image_dir.mkdir(parents=True)
    create_image(0, image_dir)
    (dataset_dir / "labels.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "img_0.jpg",
                        "width": 512,
                        "height": 512,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 0,
                        "bbox": [128, 128, 256, 256],
                        "area": 65536,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 0, "name": "budgie"}],
            }
        )
    )

    with pytest.raises(ValueError, match="use_keypoint_ann"):
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="coco",
            parser_kwargs={"use_keypoint_ann": True},
            delete_local=True,
            save_dir=tempdir,
        )

    # Without the keypoint options the same single-split source still imports.
    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        dataset_type="coco",
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 1
    finally:
        dataset.delete_dataset(delete_local=True)


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


def test_coco_split_source_rejects_unknown_parser_kwargs(
    dataset_name: str,
    tempdir: Path,
):
    """An unknown parser argument must be as loud for split sources.

    Regression: ``COCOParser.parse`` forwarded ``**kwargs`` to the
    inherited parse for a single-split source but dropped them entirely
    for one with split directories - the common layout. A typo such as
    ``use_keypoints_ann`` therefore raised ``TypeError`` against one COCO
    source and imported a keypoint-free dataset without a word against
    the other.
    """
    dataset_dir = tempdir / "coco_split_kwargs"
    _write_coco_split(
        dataset_dir / "train",
        roboflow=True,
        data=_coco_data(
            [{"id": 1, "file_name": "img_0.jpg", "width": 512, "height": 512}],
            categories=[{"id": 0, "name": "budgie"}],
        ),
    )
    create_image(0, dataset_dir / "train")

    with pytest.raises(TypeError, match="use_keypoints_ann"):
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            dataset_type="coco",
            parser_kwargs={"use_keypoints_ann": True},
            delete_local=True,
            save_dir=tempdir,
        )

    assert not LuxonisDataset.exists(dataset_name)


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
    assert COCOParser.detect(empty) is None

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
    # Recognizing a split decodes its annotations, so they are handed
    # back with the arguments instead of being read again to parse them.
    assert parser.validate_split(train_only / "train") == {
        "image_dir": image_dir,
        "annotation_path": annotation_path,
        "annotation_data": _coco_data([], categories=[]),
    }
    layout = COCOParser.detect(train_only)
    assert layout is not None
    assert layout.splits == {
        "train": parser.validate_split(train_only / "train")
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
        "annotation_data": _coco_data([], categories=[]),
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
    assert (COCOParser._load_coco_json(path) is not None) is expected
    if content == "{":
        path.unlink()
        assert COCOParser._load_coco_json(path) is None


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
    assert (
        parser._val_files_moved_to_test(
            images, has_test_files=False, split_val_to_test=True
        )
        == images[2:]
    )
    assert (
        parser._val_files_moved_to_test(
            images, has_test_files=False, split_val_to_test=False
        )
        == []
    )
    assert (
        parser._val_files_moved_to_test(
            images, has_test_files=True, split_val_to_test=True
        )
        == []
    )


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

    records = _records(parser._split_records(image_dir, annotation_path))
    assert parser._skeletons == {
        "bird": {"labels": ["head", "tail"], "edges": [(0, 1)]}
    }
    assert len(records) == 2
    assert records[0]["annotation"]["instance_segmentation"]
    assert records[1]["annotation"]["instance_id"] == 2
    assert records[1]["annotation"]["keypoints"]["keypoints"] == [
        (0.0, 1.0, 2),
        (0.5, 0.5, 1),
    ]
    files = list(dict.fromkeys(record["file"] for record in records))
    # The split declares a skeleton, so an image without annotations is
    # not part of it.
    assert unlabeled not in files
    assert parser._split_files(image_dir, annotation_path) == files
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
    assert _records(parser._split_records(image_dir, plain_annotations)) == [
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

    splits = _files_by_split(_parse(parser, root))
    # No test split at all, so the validation images stand in for it.
    # Regression: the halving used `round(len * 0.5)`, which rounds half
    # to even, so a lone validation image got a split point of zero and
    # moved to `test` whole, leaving `val` empty. It must stay in `val`.
    assert len(splits.get("val", [])) == 1
    assert len(splits.get("test", [])) == 0

    (root / "test").mkdir()
    with pytest.raises(ValueError, match="Test split"):
        _parse(parser, root)
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
    tagged = _parse(parser, single_split)
    assert tagged
    # A source that is not organized into splits carries no split name.
    assert all(split_name is None for split_name, _ in tagged)

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

    tagged = _parse(
        parser,
        keypoint_root,
        use_keypoint_ann=True,
        keypoint_ann_paths=paths,
        split_val_to_test=False,
    )
    assert len(tagged) == 2

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
    splits = _files_by_split(
        _parse(
            parser,
            keypoint_root,
            use_keypoint_ann=True,
            keypoint_ann_paths=paths,
            split_val_to_test=False,
        )
    )
    # The keypoint annotations of the test split are missing, so the
    # split contributes nothing and is not halved out of the val split.
    assert splits.get("test", []) == []

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
    splits = _files_by_split(
        _parse(
            parser,
            keypoint_root,
            use_keypoint_ann=True,
            keypoint_ann_paths=paths,
        )
    )
    assert len(splits["test"]) == 1


def test_coco_partial_keypoint_paths_fail_with_a_clear_message(
    tmp_path: Path,
):
    """A partial `keypoint_ann_paths` must name the option, not crash.

    Regression: the `test` split handled a missing mapping entry and a
    missing file, but `train` and `val` indexed the caller-supplied
    mapping directly - a mapping without their key raised a bare
    `KeyError`, and one naming a nonexistent file failed later inside
    the annotation loading with a `FileNotFoundError` that named
    neither the option nor the split.
    """
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

    parser = _plugin(COCOParser)
    with pytest.raises(ValueError, match="keypoint_ann_paths"):
        _parse(
            parser,
            root,
            use_keypoint_ann=True,
            keypoint_ann_paths={"val": "raw/person_keypoints_val2017.json"},
        )

    with pytest.raises(ValueError, match="not found"):
        _parse(
            parser,
            root,
            use_keypoint_ann=True,
            keypoint_ann_paths={
                "train": "raw/missing_train.json",
                "val": "raw/missing_val.json",
            },
        )


def test_coco_val_to_test_halves_only_the_labelled_images(tmp_path: Path):
    """Halving the validation split must count what it imports.

    The keypoint annotations of a COCO release list every image but
    annotate only some of them, and a split declaring a skeleton drops
    the unannotated ones. Halving has to count the images that really
    become records, or the halves handed to `val` and `test` are not
    halves of what is imported at all.
    """
    categories = [
        {"id": 1, "name": "bird", "keypoints": ["head"], "skeleton": []}
    ]
    root = tmp_path / "dataset"
    for split_name, stems, labelled in (
        ("train", ["train_0"], 1),
        ("validation", ["val_0", "val_1", "val_2", "val_3"], 2),
    ):
        image_dir, _ = _write_coco_split(
            root / split_name,
            roboflow=False,
            data=_coco_data(
                [
                    {
                        "id": index,
                        "file_name": f"{stem}.jpg",
                        "width": 20,
                        "height": 10,
                    }
                    for index, stem in enumerate(stems)
                ],
                [
                    {
                        "id": index,
                        "image_id": index,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 5],
                        "keypoints": [1, 1, 2],
                    }
                    for index in range(labelled)
                ],
                categories=categories,
            ),
        )
        for stem in stems:
            _image(image_dir / f"{stem}.jpg")

    parser = _plugin(COCOParser)
    splits = _files_by_split(_parse(parser, root))
    assert [file.name for file in splits["val"]] == ["val_0.jpg"]
    assert [file.name for file in splits["test"]] == ["val_1.jpg"]


def _count_json_loads(
    monkeypatch: pytest.MonkeyPatch, decoded: list[str]
) -> None:
    """Record the name of every file decoded by the COCO parser."""
    real_load = json.load

    def counting_load(file: TextIO, **kwargs: Any) -> Any:
        decoded.append(Path(file.name).name)
        return real_load(file, **kwargs)

    monkeypatch.setattr(coco_parser_module.json, "load", counting_load)


def _count_polygon_decodes(
    monkeypatch: pytest.MonkeyPatch, decoded: list[Any]
) -> None:
    """Record every polygon the COCO parser converts to RLE."""
    real_frpyobjects = mask_util.frPyObjects

    def counting_frpyobjects(*args: Any, **kwargs: Any) -> Any:
        decoded.append(args[0])
        return real_frpyobjects(*args, **kwargs)

    monkeypatch.setattr(mask_util, "frPyObjects", counting_frpyobjects)


def test_coco_decodes_each_annotation_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard against decoding the same annotation file over and over.

    Regression: a parse decoded each split's annotations five times --
    once to recognize the split, once to validate it again inside the
    split loop, once in `clean_annotations` for the train split and once
    more while parsing it, on top of the pass detection makes. The
    decoded annotations now travel from `detect` to `parse` inside the
    layout, so an annotation file is read exactly once per import.
    """
    root = tmp_path / "dataset"
    # The train split holds a file name `clean_annotations` strips, so
    # the cleaning pass really rewrites the annotations instead of
    # returning the untouched path.
    for split_name, image_names in (
        ("train", ["train.jpg", "000000341448.jpg"]),
        ("validation", ["validation.jpg"]),
    ):
        image_dir, _ = _write_coco_split(
            root / split_name,
            roboflow=False,
            data=_coco_data(
                [
                    {
                        "id": index,
                        "file_name": name,
                        "width": 20,
                        "height": 10,
                    }
                    for index, name in enumerate(image_names)
                ]
            ),
        )
        for name in image_names:
            _image(image_dir / name)

    decoded: list[str] = []
    _count_json_loads(monkeypatch, decoded)

    parser = _plugin(COCOParser)
    assert len(_parse(parser, root)) == 2

    assert decoded == ["labels.json", "labels.json"]
    assert (root / "train" / "labels_fixed.json").exists()


def test_coco_records_stream_one_segmentation_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Guard the single expensive pass over the annotations.

    Regression: the file list was collected by running the whole
    generator a first time, so every polygon was converted to RLE twice
    and the records were not lazy in any useful sense -- all of the work
    had already happened by the time `parse` returned. The records now
    stream: nothing is decoded before one is pulled, each pulled record
    decodes its own segmentation and no other, and listing the files of
    the split decodes none at all -- while reporting the very same files
    in the very same order as the records.
    """
    parser = _plugin(COCOParser)
    image_dir = tmp_path / "images"
    for name in ("one.jpg", "two.jpg", "skipped.jpg"):
        _image(image_dir / name, size=(20, 10))
    polygon = [[2.0, 1.0, 12.0, 1.0, 12.0, 6.0, 2.0, 6.0]]
    annotation_path = tmp_path / "labels.json"
    annotation_path.write_text(
        json.dumps(
            _coco_data(
                [
                    {
                        "id": index,
                        "file_name": name,
                        "width": 20,
                        "height": 10,
                    }
                    for index, name in enumerate(
                        ("one.jpg", "two.jpg", "skipped.jpg", "missing.jpg")
                    )
                ],
                [
                    {
                        "id": 10,
                        "image_id": 0,
                        "category_id": 1,
                        "bbox": [2, 1, 10, 5],
                        "segmentation": polygon,
                    },
                    {
                        "id": 11,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [2, 1, 10, 5],
                        "segmentation": polygon,
                    },
                    # The only annotation of `skipped.jpg`, so the image
                    # contributes no record and no file either.
                    {
                        "id": 12,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [2, 1, 10, 5],
                        "iscrowd": 1,
                    },
                ],
                [{"id": 1, "name": "bird"}],
            )
        )
    )

    decoded_polygons: list[Any] = []
    _count_polygon_decodes(monkeypatch, decoded_polygons)

    records = parser._split_records(image_dir, annotation_path)
    assert len(decoded_polygons) == 0

    first = next(iter(records))
    assert len(decoded_polygons) == 1

    collected = cast(list[dict[str, Any]], [first, *records])
    assert len(decoded_polygons) == 2
    files = list(dict.fromkeys(record["file"] for record in collected))
    assert [file.name for file in files] == ["one.jpg", "two.jpg"]

    assert parser._split_files(image_dir, annotation_path) == files
    assert len(decoded_polygons) == 2


def test_coco_enumerated_files_match_the_tagged_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Enumerating the splits must not parse them, and must agree.

    Only count-based `split_ratios` need the files before a record is
    added, and the annotation index the parse reads anyway answers that:
    listing the splits decodes no segmentation at all. What it reports
    has to be what the records are tagged with, down to the validation
    images that stand in for a test split carrying no labels.
    """
    root = tmp_path / "dataset"
    polygon = [[2.0, 1.0, 12.0, 1.0, 12.0, 6.0, 2.0, 6.0]]
    for split_name, stems in (
        ("train", ["train_0", "train_1"]),
        ("validation", ["val_0", "val_1", "val_2"]),
    ):
        image_dir, _ = _write_coco_split(
            root / split_name,
            roboflow=False,
            data=_coco_data(
                [
                    {
                        "id": index,
                        "file_name": f"{stem}.jpg",
                        "width": 20,
                        "height": 10,
                    }
                    for index, stem in enumerate(stems)
                ],
                [
                    {
                        "id": index,
                        "image_id": index,
                        "category_id": 1,
                        "bbox": [2, 1, 10, 5],
                        "segmentation": polygon,
                    }
                    for index in range(len(stems))
                ],
                [{"id": 1, "name": "bird"}],
            ),
        )
        for stem in stems:
            _image(image_dir / f"{stem}.jpg", size=(20, 10))

    decoded_polygons: list[Any] = []
    _count_polygon_decodes(monkeypatch, decoded_polygons)

    layout = COCOParser.detect(root)
    assert layout is not None
    parser = _plugin(COCOParser)
    enumerated = parser.enumerate_files(root, layout)
    assert enumerated is not None
    assert len(decoded_polygons) == 0
    assert {name: len(files) for name, files in enumerated.items()} == {
        "train": 2,
        "val": 2,
        "test": 1,
    }

    assert enumerated == _files_by_split(_parse(parser, root))
    assert len(decoded_polygons) == 5


def test_coco_fallback_instance_ids_avoid_the_taken_ones(tmp_path: Path):
    """Guard the id set built only once an annotation turns up without one.

    Regression: the set of instance ids already in use is now collected
    lazily, and collecting it too late -- or after a fallback id was
    handed out -- would start handing out ids that another annotation
    already owns.
    """
    parser = _plugin(COCOParser)
    image_dir = tmp_path / "images"
    _image(image_dir / "image.jpg", size=(20, 10))
    annotation_path = tmp_path / "labels.json"
    annotation_path.write_text(
        json.dumps(
            _coco_data(
                [
                    {
                        "id": 1,
                        "file_name": "image.jpg",
                        "width": 20,
                        "height": 10,
                    }
                ],
                [
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]},
                    {
                        "id": 0,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 2, 2],
                    },
                    {"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]},
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 2, 2],
                    },
                    {
                        "id": 3,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 2, 2],
                    },
                ],
                [{"id": 1, "name": "bird"}],
            )
        )
    )

    records = _records(parser._split_records(image_dir, annotation_path))
    assert [record["annotation"]["instance_id"] for record in records] == [
        2,
        0,
        4,
        1,
        3,
    ]


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 2],
        [0, 0, 2, 2, 2],
        ["nope", 0, 2, 2],
        [None, 0, 2, 2],
        5,
        float("nan"),
        [float("nan"), 0, 2, 2],
        [float("inf"), 0, 2, 2],
    ],
)
def test_coco_unusable_bboxes_are_reported_not_raised(
    tmp_path: Path, bbox: Any
):
    """Guard the unrolled bounding box conversion.

    Regression: the four coordinates used to be converted by a generator
    expression whose unpacking raised on a box that was not four values.
    Converting them one by one has to keep leaving through the very same
    `TypeError` / `ValueError`, or a malformed box would abort the whole
    parse instead of being reported and skipped.
    """
    parser = _plugin(COCOParser)
    image_dir = tmp_path / "images"
    _image(image_dir / "image.jpg", size=(20, 10))
    annotation_path = tmp_path / "labels.json"
    annotation_path.write_text(
        json.dumps(
            _coco_data(
                [
                    {
                        "id": 1,
                        "file_name": "image.jpg",
                        "width": 20,
                        "height": 10,
                    }
                ],
                [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": bbox,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 2, 2],
                    },
                ],
                [{"id": 1, "name": "bird"}],
            )
        )
    )

    records = _records(parser._split_records(image_dir, annotation_path))
    assert [record["annotation"]["instance_id"] for record in records] == [2]
    assert len(parser._issues.messages) == 1


def test_coco_write_json_matches_json_dump(tmp_path: Path):
    """Guard the chunked JSON writer against drifting from `json.dump`.

    Regression: `clean_annotations` writes the cleaned annotations one
    child at a time to reach the C encoder without holding the whole
    encoded file in memory. The bytes it writes have to stay exactly the
    ones `json.dump` writes, down to the separators and to how keys that
    are not strings are coerced.
    """
    limit = coco_parser_module._JSON_CHUNK_LIMIT
    large_list = list(range(limit + 5))
    large_dict = {f"key_{index}": index for index in range(limit + 5)}
    payload: dict[Any, Any] = {
        "images": [
            {"id": index, "file_name": f"ünï_{index}.jpg", "ratio": index / 7}
            for index in range(limit + 3)
        ],
        "nested": [large_list, [1, 2, 3], large_dict, {}, []],
        "deep": {"level": {"values": large_list}},
        "scalars": [
            None,
            True,
            False,
            -0.0,
            1e300,
            float("inf"),
            float("nan"),
            'quote " backslash \\ newline \n',
        ],
        7: "int key",
        2.5: "float key",
        None: "null key",
        False: "bool key",
        # Keys that are not strings in front of a child large enough to
        # be written on its own, which is where the key is emitted apart
        # from its value.
        9: large_list,
        3.5: large_dict,
        True: large_list,
        "empty_dict": {},
        "empty_list": [],
    }

    chunked = tmp_path / "chunked.json"
    with open(chunked, "w") as file:
        coco_parser_module._write_json(file, payload)
    dumped = tmp_path / "dumped.json"
    with open(dumped, "w") as file:
        json.dump(payload, file)
    assert chunked.read_text() == dumped.read_text()

    for value in ([], {}, "text", 3, None, [[]], {"a": []}):
        single = tmp_path / "single.json"
        with open(single, "w") as file:
            coco_parser_module._write_json(file, value)
        assert single.read_text() == json.dumps(value)


def test_coco_write_json_does_not_buffer_the_whole_file(tmp_path: Path):
    """Guard the memory bound of the chunked JSON writer.

    Regression: encoding the annotations with a single `json.dumps` is
    just as fast but holds the whole encoded file, several hundred
    megabytes for real COCO annotations, in memory at once. Writing one
    child at a time has to keep the text held at any moment proportional
    to a single child.
    """
    payload = {
        "annotations": [
            {
                "id": index,
                "image_id": index,
                "bbox": [1.5, 2.5, 3.5, 4.5],
                "segmentation": [[float(point) for point in range(40)]],
            }
            for index in range(3000)
        ]
    }
    written = tmp_path / "chunked.json"
    tracemalloc.start()
    try:
        with open(written, "w") as file:
            coco_parser_module._write_json(file, payload)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    encoded_size = written.stat().st_size
    assert encoded_size > 500_000
    assert peak < encoded_size // 4


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
    assert cleaned.read_text() == json.dumps(data)

    # Regression: `annotations` is optional in a COCO file (test sets
    # ship without it), but cleaning one that also held a blacklisted
    # file name indexed the key directly and raised `KeyError` in the
    # middle of an import.
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
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
                "categories": [],
            }
        )
    )
    cleaned = clean_annotations(annotation_path)
    data = json.loads(cleaned.read_text())
    assert [image["id"] for image in data["images"]] == [2]
    assert data["annotations"] == []


def test_clean_coco_annotations_filters_in_place(tmp_path: Path):
    """Guard the decoded annotations `parse` passes in and reuses.

    Regression: `parse` hands the annotations it already decoded to
    `clean_annotations` and then straight on to `_parse_split` instead
    of reading the cleaned file back. That only holds while the cleaning
    filters the very dictionary it was given, so the caller ends up with
    the contents of the path it gets back.
    """
    annotation_path = tmp_path / "labels.json"
    payload = _coco_data(
        [
            {
                "id": 1,
                "file_name": "000000341448.jpg",
                "width": 1,
                "height": 1,
            },
            {"id": 2, "file_name": "safe.jpg", "width": 1, "height": 1},
        ],
        [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]},
        ],
    )
    annotation_path.write_text(json.dumps(payload))

    given = json.loads(annotation_path.read_text())
    cleaned = clean_annotations(annotation_path, given)
    assert cleaned == annotation_path.with_name("labels_fixed.json")
    assert given == json.loads(cleaned.read_text())
    assert [image["id"] for image in given["images"]] == [2]
