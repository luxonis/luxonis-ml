"""Benchmark datasets for the YOLOv8 parser."""

from pathlib import Path

import yaml

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    keypoint_names,
    keypoints,
    polygon,
    write_images,
)

SPLITS = ("train", "valid", "test")


def _write_split(
    root: Path,
    split: str,
    scale: Scale,
    lines: list[str],
) -> int:
    """Write one Roboflow-layout split and return its image count."""
    names = [f"img_{index}" for index in range(scale.images)]
    write_images(root / split / "images", names, scale.image_size)
    label_dir = root / split / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(lines) + "\n"
    for name in names:
        (label_dir / f"{name}.txt").write_text(payload)
    return len(names)


@benchmark_dataset("yolov8")
def build_detection(root: Path, scale: Scale) -> BenchmarkCase:
    """Build detection labels across three splits."""
    lines = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        class_id = index % scale.classes
        lines.append(f"{class_id} {x + w / 2} {y + h / 2} {w} {h}")

    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": class_names(scale.classes)})
    )
    images = sum(_write_split(root, split, scale, lines) for split in SPLITS)

    return BenchmarkCase(
        dataset_type="yolov8",
        source=root,
        features=("detection", "multi-split", "roboflow-layout"),
        images=images,
        expected_records=images * scale.annotations_per_image,
        expected_files=images,
    )


@benchmark_dataset("yolov8instancesegmentation")
def build_segmentation(root: Path, scale: Scale) -> BenchmarkCase:
    """Build polygon labels, making the parser read every image."""
    lines = []
    for index in range(scale.annotations_per_image):
        points = polygon(index, scale.polygon_points)
        flat = " ".join(f"{x} {y}" for x, y in points)
        lines.append(f"{index % scale.classes} {flat}")

    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": class_names(scale.classes)})
    )
    images = sum(_write_split(root, split, scale, lines) for split in SPLITS)

    return BenchmarkCase(
        dataset_type="yolov8instancesegmentation",
        source=root,
        features=("instance segmentation", "multi-split", "image decode"),
        images=images,
        expected_records=images * scale.annotations_per_image,
        expected_files=images,
    )


@benchmark_dataset("yolov8keypoints")
def build_keypoints(root: Path, scale: Scale) -> BenchmarkCase:
    """Build keypoint labels with an explicit ``kpt_shape``."""
    lines = []
    for index in range(scale.annotations_per_image):
        x, y, w, h = bbox(index)
        flat = " ".join(
            f"{kx} {ky} {visibility}"
            for kx, ky, visibility in keypoints(index, scale.keypoints)
        )
        lines.append(
            f"{index % scale.classes} {x + w / 2} {y + h / 2} {w} {h} {flat}"
        )

    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "names": class_names(scale.classes),
                "kpt_shape": [scale.keypoints, 3],
                "keypoints": keypoint_names(scale.keypoints),
            }
        )
    )
    images = sum(_write_split(root, split, scale, lines) for split in SPLITS)

    return BenchmarkCase(
        dataset_type="yolov8keypoints",
        source=root,
        features=("keypoints", "multi-split", "visibility flags"),
        images=images,
        expected_records=images * scale.annotations_per_image,
        expected_files=images,
    )
