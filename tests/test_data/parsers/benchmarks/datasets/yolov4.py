"""Benchmark datasets for the YOLOv4 parser.

The parser reads one ``_annotations.txt`` and one ``_classes.txt`` per
split, so the generated dataset uses the multi-split Roboflow layout
(``train``/``valid``/``test``). The parser also accepts a bare split
directory as the source, but that variant is only the multi-split one
with a single split, so it is not benchmarked separately.
"""

from pathlib import Path

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    write_images,
)

SPLITS = ("train", "valid", "test")

MISSING_IMAGE = "missing.jpg"
"""Image referenced by the annotations but never written.

Exercises the parser's missing-image branch; one such line per split
makes the expected issue count exactly ``len(SPLITS)``.
"""


def _boxes(scale: Scale) -> str:
    """Return the shared annotation payload of one image line.

    YOLOv4 boxes are absolute ``x1,y1,x2,y2,class_id`` pixel corners.
    Every image has the same size, so a single payload is built once and
    reused for every annotated line.
    """
    size = scale.image_size
    boxes = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        boxes.append(
            f"{round(x * size)},{round(y * size)},"
            f"{round((x + w) * size)},{round((y + h) * size)},"
            f"{index % scale.classes}"
        )
    return " ".join(boxes)


def _counts(scale: Scale) -> tuple[int, int, int, int]:
    """Return the per-split image count of every kind of image.

    Returns:
        ``(annotated, nested, unannotated, unlisted)``: images listed
        with boxes, images listed with boxes through a nested relative
        path, images listed with no boxes at all, and images present in
        the split directory but absent from the annotation file.

    """
    portion = scale.images // 10
    return (scale.images - 3 * portion, portion, portion, portion)


def _write_split(root: Path, split: str, scale: Scale, boxes: str) -> int:
    """Write one YOLOv4 split and return its image count."""
    annotated, nested, unannotated, unlisted = _counts(scale)
    split_dir = root / split

    annotated_names = [f"img_{index}" for index in range(annotated)]
    unannotated_names = [f"empty_{index}" for index in range(unannotated)]
    unlisted_names = [f"extra_{index}" for index in range(unlisted)]
    # Reusing the root stems makes the nested images differ from images
    # in the split root by their directory alone.
    nested_names = [f"img_{index}" for index in range(nested)]

    write_images(
        split_dir,
        annotated_names + unannotated_names + unlisted_names,
        scale.image_size,
    )
    write_images(split_dir / "nested", nested_names, scale.image_size)

    lines = [f"{name}.jpg {boxes}" for name in annotated_names]
    lines += [f"nested/{name}.jpg {boxes}" for name in nested_names]
    lines += [f"{name}.jpg" for name in unannotated_names]
    lines.append(f"{MISSING_IMAGE} {boxes}")

    (split_dir / "_annotations.txt").write_text("\n".join(lines) + "\n")
    (split_dir / "_classes.txt").write_text(
        "\n".join(class_names(scale.classes)) + "\n"
    )
    return scale.images


@benchmark_dataset("yolov4")
def build_detection(root: Path, scale: Scale) -> BenchmarkCase:
    """Absolute-pixel detection labels across three splits.

    One annotation file per split holds every kind of line the parser
    understands: images with boxes, images reached through a nested
    relative path, images listed without a single box, and an image that
    does not exist. Images the annotation file never mentions are also
    written, since the parser emits them as annotation-less records.
    """
    boxes = _boxes(scale)
    images = sum(_write_split(root, split, scale, boxes) for split in SPLITS)

    annotated, nested, unannotated, unlisted = _counts(scale)
    records_per_split = (
        (annotated + nested) * scale.annotations_per_image
        + unannotated
        + unlisted
    )

    return BenchmarkCase(
        dataset_type="yolov4",
        source=root,
        features=(
            "detection",
            "classification",
            "multi-split",
            "roboflow-layout",
            "nested image paths",
            "unannotated images",
            "unlisted images",
            "missing images",
            "image decode",
        ),
        images=images,
        expected_records=records_per_split * len(SPLITS),
        expected_files=images,
        expected_issues=len(SPLITS),
    )
