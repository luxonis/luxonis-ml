"""Benchmark datasets for the YOLOv6 parser.

The YOLOv6 layout keeps images and labels in separate top-level trees,
``images/<split>`` and ``labels/<split>``, with a single ``data.yaml``
holding the class names. The parser emits bounding boxes only, so the
generated dataset varies what actually changes its work: the number of
splits, the image suffixes its filter has to recognize, and the two ways
an image can end up without annotations.

Not covered: the parser also accepts being pointed straight at one
``images/<split>`` directory, resolving ``labels/`` and ``data.yaml``
relative to that directory's grandparent. That is a strict subset of the
work done for the multi-split root, and only one builder may be
registered per dataset type, so the three-split root is used here.
"""

from pathlib import Path

import yaml

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    write_images,
)

SPLITS = ("train", "valid", "test")

# Every UNLABELLED_PERIOD-th image is written without a label file and
# the image right after it with an empty one, so both routes to a record
# with `annotation=None` are exercised.
UNLABELLED_PERIOD = 25

# One image in PNG_PERIOD is a PNG, so the parser's suffix filter sees
# more than a single supported extension.
PNG_PERIOD = 10


def _label_payload(scale: Scale) -> str:
    """Return the label file shared by every annotated image."""
    lines = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        class_id = index % scale.classes
        lines.append(f"{class_id} {x + w / 2} {y + h / 2} {w} {h}")
    return "\n".join(lines) + "\n"


def _write_split(
    root: Path,
    split: str,
    scale: Scale,
    payload: str,
) -> int:
    """Write one split and return the number of records it yields."""
    names = [f"img_{index}" for index in range(scale.images)]
    image_dir = root / "images" / split
    write_images(
        image_dir,
        (name for index, name in enumerate(names) if index % PNG_PERIOD),
        scale.image_size,
    )
    write_images(
        image_dir,
        (name for index, name in enumerate(names) if not index % PNG_PERIOD),
        scale.image_size,
        suffix=".png",
    )

    label_dir = root / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    records = 0
    for index, name in enumerate(names):
        position = index % UNLABELLED_PERIOD
        if position == 0:
            records += 1
        elif position == 1:
            (label_dir / f"{name}.txt").write_text("")
            records += 1
        else:
            (label_dir / f"{name}.txt").write_text(payload)
            records += scale.annotations_per_image
    return records


@benchmark_dataset("yolov6")
def build_detection(root: Path, scale: Scale) -> BenchmarkCase:
    """Bounding boxes across the three splits of a YOLOv6 root."""
    names = class_names(scale.classes)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "names": names,
                "nc": len(names),
                "train": "../images/train",
                "val": "../images/valid",
                "test": "../images/test",
            }
        )
    )

    payload = _label_payload(scale)
    records = sum(
        _write_split(root, split, scale, payload) for split in SPLITS
    )
    images = scale.images * len(SPLITS)

    return BenchmarkCase(
        dataset_type="yolov6",
        source=root,
        features=(
            "detection",
            "multi-split",
            "separate images/labels trees",
            "mixed image suffixes",
            "images without annotations",
        ),
        images=images,
        expected_records=records,
        expected_files=images,
    )
