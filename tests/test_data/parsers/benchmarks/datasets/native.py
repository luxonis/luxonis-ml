"""Benchmark datasets for the native parser.

The native format keeps one ``annotations.json`` per split holding the
record dictionaries `LuxonisDataset.add` itself accepts. Benchmarking it
therefore has to cover both the parser's own work -- discovering splits
and rewriting every media and mask path it finds -- and the breadth of
the record contract the format can express.

Every image plays one of `ROLE_COUNT` roles, cycling with the image
index, so a single dataset exercises all record shapes at once:

- absolute media paths on images that carry no annotation,
- multi-source records using ``files`` instead of ``file``,
- Windows-style relative paths,
- semantic and instance segmentation masks stored as PNG files,
- inline run-length encoded semantic segmentation, whose ``mask`` is not
  a path and must be left untouched,
- bounding boxes, keypoints, and detections carrying several task types,
- classification-only records,
- annotation metadata together with nested sub-detections,
- record-level ``sample_metadata``, per-role task names, and records
  leaving ``task_name`` out entirely.

Two things are deliberately left out. The single-split layout, where
``annotations.json`` sits directly in the dataset root, is mutually
exclusive with the split layout generated here, and the split layout is
the representative one. Polyline segmentation is not expressible in this
format at all: `SegmentationAnnotation` requires the points to be
tuples, and JSON only round-trips them as lists.
"""

import json
from pathlib import Path
from typing import Any

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    keypoints,
    write_images,
    write_masks,
)

SPLITS = ("train", "val", "test")

ROLE_COUNT = 10
"""Number of record shapes cycled over the images of a split."""

MASK_KEYS = {
    3: "segmentation",
    4: "instance_segmentation",
    6: "instance_segmentation",
}
"""Annotation key holding a mask file path, per role."""

TASK_NAMES: dict[int, str | None] = {
    0: None,
    1: "detection",
    2: "detection",
    3: "segmentation",
    4: "instance",
    5: "keypoints",
    6: "multitask",
    7: "segmentation",
    8: None,
    9: "attributes",
}
"""Task name each role assigns, or `None` to leave the key out."""

MASK_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))


def _box(index: int) -> dict[str, float]:
    """Return a deterministic bounding box payload."""
    x, y, w, h = bbox(index)
    return {"x": x, "y": y, "w": w, "h": h}


def _keypoints(index: int, count: int) -> dict[str, Any]:
    """Return a deterministic keypoint payload."""
    return {"keypoints": keypoints(index, count)}


def _rle(size: int) -> dict[str, Any]:
    """Return an uncompressed run-length encoded square mask.

    The counts are the only segmentation form whose ``mask`` is inline
    data rather than a file, so the parser has to leave it alone.
    """
    total = size * size
    head = total // 4
    return {
        "height": size,
        "width": size,
        "counts": [head, head, total - 2 * head],
    }


def _annotations(
    scale: Scale, classes: list[str]
) -> dict[int, list[dict[str, Any]]]:
    """Return the image-independent annotation payloads per role.

    Role ``0`` is absent because its images carry no annotation. Mask
    paths are the only per-image part of a payload and are filled in by
    `_split_entries`.

    Args:
        scale: Size of the dataset being generated.
        classes: Class names to cycle through.

    Returns:
        Annotation payloads keyed by role, each holding one payload per
        annotated instance.

    """
    instances = range(scale.annotations_per_image)

    def name_of(index: int) -> str:
        return classes[index % len(classes)]

    return {
        1: [
            {
                "class": name_of(index),
                "instance_id": index,
                "boundingbox": _box(index),
            }
            for index in instances
        ],
        2: [
            {"class": name_of(index), "boundingbox": _box(index)}
            for index in instances
        ],
        3: [{"class": name_of(index)} for index in instances],
        4: [
            {
                "class": name_of(index),
                "instance_id": index,
                "boundingbox": _box(index),
            }
            for index in instances
        ],
        5: [
            {
                "class": name_of(index),
                "instance_id": index,
                "boundingbox": _box(index),
                "keypoints": _keypoints(index, scale.keypoints),
            }
            for index in instances
        ],
        6: [
            {
                "class": name_of(index),
                "instance_id": index,
                "boundingbox": _box(index),
                "keypoints": _keypoints(index, scale.keypoints),
            }
            for index in instances
        ],
        7: [
            {
                "class": name_of(index),
                "segmentation": _rle(scale.image_size),
            }
            for index in instances
        ],
        8: [{"class": name_of(index)} for index in instances],
        9: [
            {
                "class": name_of(index),
                "metadata": {
                    "text": f"note_{index}",
                    "count": index,
                    "score": round(0.05 * index, 4),
                },
                "sub_detections": {
                    "head": {
                        "class": name_of(index + 1),
                        "boundingbox": _box(index + 1),
                    }
                },
            }
            for index in instances
        ],
    }


def _split_entries(
    split_dir: Path,
    split: str,
    scale: Scale,
    classes: list[str],
) -> list[dict[str, Any]]:
    """Write one split's media and return its annotation entries.

    Args:
        split_dir: Directory the split is written to.
        split: Name of the split, recorded in ``sample_metadata``.
        scale: Size of the dataset being generated.
        classes: Class names to cycle through.

    Returns:
        The entries belonging in the split's ``annotations.json``.

    """
    names = [f"img_{index}" for index in range(scale.images)]
    image_dir = split_dir / "images"
    write_images(image_dir, names, scale.image_size)
    absolute_dir = image_dir.resolve()

    write_images(
        split_dir / "depth",
        names[1::ROLE_COUNT],
        scale.image_size,
        suffix=".png",
    )
    write_masks(
        split_dir / "masks",
        [
            name
            for index, name in enumerate(names)
            if index % ROLE_COUNT in MASK_KEYS
        ],
        scale.image_size,
        MASK_COLORS,
    )

    payloads = _annotations(scale, classes)
    entries: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        role = index % ROLE_COUNT
        metadata = {
            "record_id": index,
            "split": split,
            "camera": "left" if index % 2 else "right",
            "tags": ["night", "warehouse"],
        }

        if role == 0:
            entries.append(
                {
                    "file": (absolute_dir / f"{name}.jpg").as_posix(),
                    "sample_metadata": metadata,
                    "annotation": None,
                }
            )
            continue

        if role == 1:
            source: dict[str, Any] = {
                "files": {
                    "image": f"images/{name}.jpg",
                    "depth": f"depth/{name}.png",
                }
            }
        elif role == 2:
            source = {"file": f"images\\{name}.jpg"}
        else:
            source = {"file": f"images/{name}.jpg"}

        task_name = TASK_NAMES[role]
        if task_name is not None:
            source["task_name"] = task_name
        # Role 8 is the one shape without record-level metadata.
        if role != 8:
            source["sample_metadata"] = metadata

        mask_key = MASK_KEYS.get(role)
        for payload in payloads[role]:
            if mask_key is not None:
                payload = {**payload, mask_key: {"mask": f"masks/{name}.png"}}
            entries.append({**source, "annotation": payload})

    return entries


def _referenced(entries: list[dict[str, Any]]) -> set[str]:
    """Return every distinct media path ``entries`` point at.

    The parser reports one file per distinct path, so many annotations
    on one image collapse into a single reported file.
    """
    paths: set[str] = set()
    for entry in entries:
        if "file" in entry:
            paths.add(entry["file"])
        else:
            paths.update(entry["files"].values())
    return paths


@benchmark_dataset("native")
def build_native(root: Path, scale: Scale) -> BenchmarkCase:
    """Native LDF splits covering every record shape of the format.

    The default scale is used unchanged: the annotations are one JSON
    document per split, and the media are written from a single encoded
    image and a single encoded mask, so generation stays cheap even
    though the parse itself has to walk tens of thousands of records.
    """
    classes = class_names(scale.classes)
    expected_records = 0
    expected_files = 0

    for split in SPLITS:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        entries = _split_entries(split_dir, split, scale, classes)
        (split_dir / "annotations.json").write_text(json.dumps(entries))
        expected_records += len(entries)
        expected_files += len(_referenced(entries))

    return BenchmarkCase(
        dataset_type="native",
        source=root,
        features=(
            "detection",
            "keypoints",
            "semantic segmentation",
            "instance segmentation",
            "classification",
            "annotation metadata",
            "sub-detections",
            "sample metadata",
            "multi-source records",
            "path resolution",
            "unannotated images",
            "multi-split",
        ),
        images=scale.images * len(SPLITS),
        expected_records=expected_records,
        expected_files=expected_files,
    )
