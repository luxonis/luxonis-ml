"""SOLO parser."""

import json
from pathlib import Path
from typing import (
    Any,
)

import cv2
import numpy as np
import pytest
from PIL import Image

from luxonis_ml.data.parsers import (
    SOLOParser,
)
from tests.test_data.parsers.helpers import (
    _collect_raw_records,
    _image,
    _plugin,
)


def _record_paths(records: list[dict[str, Any]]) -> list[Path]:
    """Collect the files of a record stream the way the importer does.

    Every record names the file it belongs to, and the importer keeps the
    first occurrence of each, so an image carrying several records is
    reported once.
    """
    return list(dict.fromkeys(Path(record["file"]) for record in records))


def _write_solo_split(split_path: Path) -> None:
    """Write a one-sequence SOLO split with box, instance and semantic masks."""
    sequence_path = split_path / "sequence.0"
    sequence_path.mkdir(parents=True)

    cv2.imwrite(
        str(sequence_path / "step0.camera.jpg"),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )
    for mask_name, colour in (
        ("step0.camera.instance.png", (0, 0, 255)),
        ("step0.camera.semantic.png", (0, 255, 0)),
    ):
        mask = np.zeros((8, 8, 3), dtype=np.uint8)
        mask[:4, :4] = colour
        cv2.imwrite(str(sequence_path / mask_name), mask)

    prefix = "type.unity.com/unity.solo."
    box_type = f"{prefix}BoundingBox2DAnnotation"
    instance_type = f"{prefix}InstanceSegmentationAnnotation"
    semantic_type = f"{prefix}SemanticSegmentationAnnotation"

    (split_path / "annotation_definitions.json").write_text(
        json.dumps(
            {
                "annotationDefinitions": [
                    {
                        "@type": box_type,
                        "spec": [{"label_name": "budgie", "label_id": 1}],
                    }
                ]
            }
        )
    )
    (split_path / "metadata.json").write_text(
        json.dumps({"totalSequences": 1})
    )
    (split_path / "metric_definitions.json").write_text("{}")
    (split_path / "sensor_definitions.json").write_text("{}")
    (sequence_path / "step0.frame_data.json").write_text(
        json.dumps(
            {
                "step": 0,
                "captures": [
                    {
                        "filename": "step0.camera.jpg",
                        "dimension": [8, 8],
                        "annotations": [
                            {
                                "@type": box_type,
                                "values": [
                                    {
                                        "labelName": "budgie",
                                        "origin": [0, 0],
                                        "dimension": [4, 4],
                                        "instanceId": 1,
                                    }
                                ],
                            },
                            {
                                "@type": instance_type,
                                "filename": "step0.camera.instance.png",
                                "instances": [
                                    {
                                        "color": [255, 0, 0, 255],
                                        "instanceId": 1,
                                    }
                                ],
                            },
                            {
                                "@type": semantic_type,
                                "filename": "step0.camera.semantic.png",
                                "instances": [
                                    {
                                        "labelName": "budgie",
                                        "pixelValue": [0, 255, 0, 255],
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        )
    )


def test_solo_decodes_every_mask_exactly_once(
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Parsing SOLO must not decode a mask twice.

    Regression: ``files`` was built with ``_get_added_images(generator())``
    and the record stream then ran the same generator again, so every
    semantic and instance mask PNG was read and every per-instance mask
    rebuilt twice — roughly doubling import time for the heaviest supported
    format. The parse now walks the split once, so nothing is decoded until
    a record is asked for, each mask is decoded exactly once, and the masks
    reaching the records still have full image dimensions.
    """
    split_path = tempdir / "solo" / "train"
    _write_solo_split(split_path)

    decoded: list[str] = []
    original_imread = cv2.imread

    def counting_imread(
        path,  # noqa: ANN001
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        decoded.append(str(path))
        return original_imread(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counting_imread)

    records = _plugin(SOLOParser)._split_records(split_path)
    assert decoded == [], "no mask may be decoded before a record is asked for"

    collected = _collect_raw_records(records)
    assert len(decoded) == 2, "each mask is decoded exactly once"
    assert [path.name for path in _record_paths(collected)] == [
        "step0.camera.jpg"
    ]
    masks = [
        value["mask"]
        for record in collected
        for value in record["annotation"].values()
        if isinstance(value, dict) and "mask" in value
    ]
    assert [mask.shape for mask in masks] == [(8, 8), (8, 8)]


def _solo_definitions(*, include_bbox: bool = True) -> dict[str, Any]:
    definitions: list[dict[str, Any]] = [
        {
            "@type": "type.unity.com/unity.solo.KeypointAnnotation",
            "template": {
                "keypoints": [
                    {"label": "tail", "index": 1},
                    {"label": "head", "index": 0},
                ]
            },
        },
        {
            "@type": (
                "type.unity.com/unity.solo.SemanticSegmentationAnnotation"
            )
        },
    ]
    if include_bbox:
        definitions.append(
            {
                "@type": ("type.unity.com/unity.solo.BoundingBox2DAnnotation"),
                "spec": [
                    {"label_name": "cat", "label_id": 2},
                    {"label_name": "bird", "label_id": 1},
                ],
            }
        )
    return {"annotationDefinitions": definitions}


def _write_solo_frame(
    split: Path,
    annotations: list[dict[str, Any]],
    *,
    image_name: str = "step0.camera.jpg",
    create_image: bool = True,
) -> Path:
    sequence = split / "sequence.0"
    sequence.mkdir(parents=True, exist_ok=True)
    if create_image:
        _image(sequence / image_name, size=(20, 10))
    frame = {
        "step": "0",
        "captures": [
            {
                "filename": image_name,
                "dimension": [20, 10],
                "annotations": annotations,
            },
            {
                "filename": image_name,
                "dimension": [20, 10],
                "annotations": [],
            },
        ],
    }
    frame_path = sequence / "step0.frame_data.json"
    frame_path.write_text(json.dumps(frame))
    return frame_path


def _write_solo_metadata(split: Path, *, total_sequences: int = 1) -> None:
    split.mkdir(parents=True, exist_ok=True)
    (split / "annotation_definitions.json").write_text(
        json.dumps(_solo_definitions())
    )
    (split / "metadata.json").write_text(
        json.dumps({"totalSequences": total_sequences})
    )
    (split / "metric_definitions.json").write_text("{}")
    (split / "sensor_definitions.json").write_text("{}")


def test_solo_parser_all_annotation_types(tmp_path: Path):
    parser = _plugin(SOLOParser)
    split = tmp_path / "train"
    assert parser.validate_split(split) is None
    split.mkdir()
    assert parser.validate_split(split) is None
    _write_solo_metadata(split, total_sequences=2)

    sequence = split / "sequence.0"
    sequence.mkdir()
    semantic_mask = sequence / "semantic.png"
    instance_mask = sequence / "instance.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(semantic_mask)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(instance_mask)
    annotations = [
        {
            "@type": (
                "type.unity.com/unity.solo.SemanticSegmentationAnnotation"
            ),
            "filename": semantic_mask.name,
            "instances": [
                {
                    "labelName": "bird",
                    "pixelValue": [255, 0, 0, 255],
                }
            ],
        },
        {
            "@type": "type.unity.com/unity.solo.BoundingBox2DAnnotation",
            "values": [
                {
                    "labelName": "bird",
                    "origin": [2, 1],
                    "dimension": [10, 5],
                    "instanceId": 1,
                }
            ],
        },
        {
            "@type": (
                "type.unity.com/unity.solo.InstanceSegmentationAnnotation"
            ),
            "filename": instance_mask.name,
            "instances": [
                {
                    "color": [255, 0, 0, 255],
                    "instanceId": 1,
                }
            ],
        },
        {
            "@type": "type.unity.com/unity.solo.KeypointAnnotation",
            "values": [
                {
                    "instanceId": 1,
                    "keypoints": [
                        {"location": [2, 1], "state": 2},
                        {"location": [10, 5], "state": 1},
                    ],
                }
            ],
        },
    ]
    _write_solo_frame(split, annotations)

    assert parser.validate_split(split) == {"split_path": split}
    records = _collect_raw_records(parser._split_records(split))
    assert parser._skeletons == {
        "bird": {"labels": ["head", "tail"]},
        "cat": {"labels": ["head", "tail"]},
    }
    assert len(records) == 2
    assert records[0]["annotation"]["class"] == "bird"
    combined = records[1]["annotation"]
    assert combined["boundingbox"] == {
        "x": 0.1,
        "y": 0.1,
        "w": 0.5,
        "h": 0.5,
    }
    assert combined["keypoints"]["keypoints"] == [
        (0.1, 0.1, 2),
        (0.5, 0.5, 1),
    ]
    assert "instance_segmentation" in combined

    definitions = _solo_definitions()
    assert parser._get_solo_annotation_types(definitions) == [
        "KeypointAnnotation",
        "SemanticSegmentationAnnotation",
        "BoundingBox2DAnnotation",
    ]
    assert parser._get_solo_bbox_class_names(definitions) == ["bird", "cat"]
    assert parser._get_solo_keypoint_names(definitions) == ["head", "tail"]


def test_solo_parser_structure_errors(tmp_path: Path):
    """A split the parser cannot read at all must fail before it streams.

    Each call below raises without a record being pulled: the definitions
    are read and validated when the split is opened, not while it is
    walked, so an import fails before a dataset is created.
    """
    parser = _plugin(SOLOParser)
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="path non-existent"):
        parser._split_records(missing)

    no_definitions = tmp_path / "no-definitions"
    no_definitions.mkdir()
    with pytest.raises(FileNotFoundError, match="annotation_definitions"):
        parser._split_records(no_definitions)

    no_bbox = tmp_path / "no-bbox"
    no_bbox.mkdir()
    (no_bbox / "annotation_definitions.json").write_text(
        json.dumps(_solo_definitions(include_bbox=False))
    )
    with pytest.raises(ValueError, match="No class_names"):
        parser._split_records(no_bbox)


@pytest.mark.parametrize(
    ("annotation_type", "mask_name"),
    [
        ("SemanticSegmentationAnnotation", "semantic.png"),
        ("InstanceSegmentationAnnotation", "instance.png"),
    ],
)
def test_solo_parser_missing_masks(
    tmp_path: Path,
    annotation_type: str,
    mask_name: str,
):
    """A missing or undecodable mask must fail the parse.

    Both used to be found by the file-enumeration pass, which ran eagerly.
    The walk that produces the records is the only walk left, so they now
    surface while the split streams — with the same exception and the same
    message, raised by the parser rather than by anything downstream of it.
    """
    parser = _plugin(SOLOParser)
    split = tmp_path / annotation_type
    _write_solo_metadata(split)
    _write_solo_frame(
        split,
        [
            {
                "@type": f"type.unity.com/unity.solo.{annotation_type}",
                "filename": mask_name,
                "instances": [],
            }
        ],
    )
    with pytest.raises(FileNotFoundError, match="not existent"):
        _collect_raw_records(parser._split_records(split))

    mask = split / "sequence.0" / mask_name
    mask.write_text("broken")
    with pytest.raises(ValueError, match="Failed to read mask image"):
        _collect_raw_records(parser._split_records(split))


def test_solo_parser_missing_image(tmp_path: Path):
    parser = _plugin(SOLOParser)
    split = tmp_path / "missing-image"
    _write_solo_metadata(split)
    _write_solo_frame(split, [], create_image=False)
    with pytest.raises(FileNotFoundError, match="not existent"):
        _collect_raw_records(parser._split_records(split))


def _write_solo_sequences(
    split: Path, *, sequences: int = 2, instances: int = 3
) -> None:
    """Write a SOLO split with more records than images.

    Each sequence holds one image, one semantic instance and ``instances``
    instances shared by the boxes, the masks and the keypoints, so a walk
    that reports one item per record is easy to tell apart from one that
    reports one item per image.
    """
    _write_solo_metadata(split, total_sequences=sequences)
    prefix = "type.unity.com/unity.solo."
    for index in range(sequences):
        sequence = split / f"sequence.{index}"
        sequence.mkdir(parents=True)
        _image(sequence / "step0.camera.jpg", size=(20, 10))
        for mask_name, colour in (
            ("instance.png", (255, 0, 0)),
            ("semantic.png", (0, 255, 0)),
        ):
            Image.new("RGB", (4, 4), color=colour).save(sequence / mask_name)

        instance_ids = list(range(instances))
        frame = {
            "step": index,
            "captures": [
                {
                    "filename": "step0.camera.jpg",
                    "dimension": [20, 10],
                    "annotations": [
                        {
                            "@type": (
                                f"{prefix}SemanticSegmentationAnnotation"
                            ),
                            "filename": "semantic.png",
                            "instances": [
                                {
                                    "labelName": "bird",
                                    "pixelValue": [0, 255, 0, 255],
                                }
                            ],
                        },
                        {
                            "@type": f"{prefix}BoundingBox2DAnnotation",
                            "values": [
                                {
                                    "labelName": "bird",
                                    "origin": [2, 1],
                                    "dimension": [10, 5],
                                    "instanceId": instance_id,
                                }
                                for instance_id in instance_ids
                            ],
                        },
                        {
                            "@type": (
                                f"{prefix}InstanceSegmentationAnnotation"
                            ),
                            "filename": "instance.png",
                            "instances": [
                                {
                                    "color": [255, 0, 0, 255],
                                    "instanceId": instance_id,
                                }
                                for instance_id in instance_ids
                            ],
                        },
                        {
                            "@type": f"{prefix}KeypointAnnotation",
                            "values": [
                                {
                                    "instanceId": instance_id,
                                    "keypoints": [
                                        {"location": [2, 1], "state": 2},
                                        {"location": [10, 5], "state": 1},
                                    ],
                                }
                                for instance_id in instance_ids
                            ],
                        },
                    ],
                }
            ],
        }
        (sequence / "step0.frame_data.json").write_text(json.dumps(frame))


def test_solo_walks_the_split_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """One walk must feed both the records and the files they name.

    Regression: ``files`` was collected by running the record generator
    again, so every frame JSON was read twice and every box, keypoint list
    and per-instance mask was rebuilt only to be discarded. The importer
    now collects the files from the records as they stream past, which is
    what these counters pin down: one read per frame file, one instance
    mask per record that carries one, and one file per image even though
    each image carries four records.
    """
    split = tmp_path / "train"
    _write_solo_sequences(split, sequences=2, instances=3)

    frame_reads: list[str] = []
    read_text = Path.read_text

    def counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name.endswith("frame_data.json"):
            frame_reads.append(str(self))
        return read_text(self, *args, **kwargs)

    masked: list[int] = []
    build_mask = SOLOParser._instance_mask

    def counting_instance_mask(
        mask_int: np.ndarray, target_int: int
    ) -> np.ndarray:
        masked.append(target_int)
        return build_mask(mask_int, target_int)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    monkeypatch.setattr(
        SOLOParser, "_instance_mask", staticmethod(counting_instance_mask)
    )

    records = _plugin(SOLOParser)._split_records(split)
    assert frame_reads == [], "no frame is read before a record is asked for"
    assert masked == [], "the walk must not build a mask it was not asked for"

    collected = _collect_raw_records(records)
    assert len(frame_reads) == 2, "each frame file is read exactly once"
    assert len(collected) == 8  # 1 semantic + 3 merged, per sequence
    assert len(masked) == 8  # 1 semantic + 3 instance masks, per sequence
    assert [path.name for path in _record_paths(collected)] == [
        "step0.camera.jpg"
    ] * 2


def test_solo_records_are_produced_one_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The record stream must stay lazy.

    Regression: the records must never be materialized — a SOLO dataset can
    be far larger than memory. Pulling a single record may therefore decode
    only the masks that record needs, and nothing at all may be decoded
    before the first one is asked for.
    """
    split = tmp_path / "train"
    _write_solo_sequences(split, sequences=2, instances=3)

    decoded: list[str] = []
    original_imread = cv2.imread

    def counting_imread(
        path,  # noqa: ANN001
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        decoded.append(str(path))
        return original_imread(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imread", counting_imread)

    records = _plugin(SOLOParser)._split_records(split)
    assert iter(records) is records, "records must stay a lazy iterator"
    assert decoded == [], "no mask may be decoded before a record is asked for"

    next(records)
    assert len(decoded) == 1, "only the mask of the first record is decoded"

    _collect_raw_records(records)
    assert len(decoded) == 4  # one semantic and one instance mask per sequence


def test_solo_parse_tags_records_with_their_split(tmp_path: Path):
    """`detect` finds the splits and `parse` tags every record with one.

    The layout detection returns is handed straight to `parse`, so the
    source is inspected once per import, and the ``valid`` directory SOLO
    writes is reported under the canonical ``val`` name. The skeletons are
    read after the records are exhausted, so a parse may learn them while
    it streams.
    """
    for name in ("train", "valid"):
        _write_solo_split(tmp_path / name)

    layout = SOLOParser.detect(tmp_path)
    assert layout is not None
    assert layout.split_names == ["train", "val"]

    parser = _plugin(SOLOParser)
    result = parser.parse(tmp_path, layout)
    assert iter(result.records) is result.records
    assert result.skeletons == {}, "nothing is parsed before it is consumed"

    tagged = []
    for split_name, record in result.records:
        assert isinstance(record, dict)
        tagged.append((split_name, Path(record["file"]).parent.parent.name))

    # One semantic and one merged record per split.
    assert tagged == [
        ("train", "train"),
        ("train", "train"),
        ("val", "valid"),
        ("val", "valid"),
    ]
    assert result.skeletons == {"budgie": {"labels": []}}
    assert result.skeletons is parser._skeletons
