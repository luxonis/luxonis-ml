"""Dataset builders and the timing harness shared by the benchmarks.

A benchmark dataset is produced by a builder registered for one dataset
type. Builders receive a `Scale` describing how much data to write and
return a `BenchmarkCase` telling the harness what to parse and what the
parser is expected to produce.
"""

import time
import tracemalloc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from luxonis_ml.data.datasets.base_dataset import _record_files
from luxonis_ml.data.parsers.parser_plugin import (
    ParseIssueCollector,
    get_parser_plugin,
)


@dataclass(frozen=True)
class Scale:
    """How much data a benchmark dataset should contain.

    Attributes:
        images: Number of images per split.
        annotations_per_image: Number of annotated instances per image.
        classes: Number of distinct class names.
        keypoints: Number of keypoints per skeleton.
        polygon_points: Number of points per segmentation polygon.
        image_size: Width and height of the generated images.

    """

    images: int = 2000
    annotations_per_image: int = 10
    classes: int = 20
    keypoints: int = 17
    polygon_points: int = 12
    image_size: int = 64

    def shrunk(self, factor: int) -> "Scale":
        """Return a scale with ``factor`` times fewer images.

        Used by builders whose format is heavy enough per image that the
        common image count would dominate the suite's runtime.
        """
        return Scale(
            images=max(1, self.images // factor),
            annotations_per_image=self.annotations_per_image,
            classes=self.classes,
            keypoints=self.keypoints,
            polygon_points=self.polygon_points,
            image_size=self.image_size,
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """One parser fed one generated dataset.

    Attributes:
        dataset_type: Registered parser type used to parse `source`.
        source: Root of the generated dataset.
        features: Parser features the dataset exercises, for reporting.
        images: Number of images written.
        expected_records: Number of records the parser must yield.
        expected_files: Number of files the parser must report.
        parse_kwargs: Extra keyword arguments passed to `parse`.
        expected_issues: Number of skipped annotations the parser is
            expected to report.

    """

    dataset_type: str
    source: Path
    features: tuple[str, ...]
    images: int
    expected_records: int
    expected_files: int
    parse_kwargs: dict[str, Any] = field(default_factory=dict)
    expected_issues: int = 0


DatasetBuilder = Callable[[Path, Scale], BenchmarkCase]

BENCHMARK_DATASETS: dict[str, DatasetBuilder] = {}


def benchmark_dataset(
    dataset_type: str,
) -> Callable[[DatasetBuilder], DatasetBuilder]:
    """Register a synthetic dataset builder for ``dataset_type``."""

    def decorator(builder: DatasetBuilder) -> DatasetBuilder:
        if dataset_type in BENCHMARK_DATASETS:
            raise KeyError(
                f"A benchmark dataset for '{dataset_type}' is already "
                "registered."
            )
        BENCHMARK_DATASETS[dataset_type] = builder
        return builder

    return decorator


def class_names(count: int) -> list[str]:
    """Return deterministic class names."""
    return [f"class_{index}" for index in range(count)]


def keypoint_names(count: int) -> list[str]:
    """Return deterministic keypoint names."""
    return [f"kpt_{index}" for index in range(count)]


def write_images(
    directory: Path,
    names: Iterable[str],
    size: int,
    *,
    suffix: str = ".jpg",
) -> list[Path]:
    """Write identical images under ``directory``, encoding only once.

    Benchmarks care about how fast a parser walks and reads a dataset,
    not about image content, so the pixel data is encoded a single time
    and the same bytes are written to every path. Encoding per image
    would otherwise dominate dataset generation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), 127, dtype=np.uint8)
    image[: size // 2, : size // 2] = 255
    success, buffer = cv2.imencode(suffix, image)
    if not success:  # pragma: no cover - depends on the OpenCV build
        raise RuntimeError(f"OpenCV cannot encode '{suffix}' images.")
    payload = buffer.tobytes()

    paths = []
    for name in names:
        path = directory / f"{name}{suffix}"
        path.write_bytes(payload)
        paths.append(path)
    return paths


def write_masks(
    directory: Path,
    names: Iterable[str],
    size: int,
    colors: Sequence[tuple[int, int, int]],
    *,
    suffix: str = ".png",
) -> list[Path]:
    """Write identical colour masks, encoding only once.

    Each colour in ``colors`` occupies one horizontal band so that a
    parser extracting per-instance masks finds every colour present.
    """
    directory.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size, size, 3), dtype=np.uint8)
    band = max(1, size // max(1, len(colors)))
    for index, color in enumerate(colors):
        start = index * band
        mask[start : start + band] = color
    success, buffer = cv2.imencode(suffix, mask)
    if not success:  # pragma: no cover - depends on the OpenCV build
        raise RuntimeError(f"OpenCV cannot encode '{suffix}' images.")
    payload = buffer.tobytes()

    paths = []
    for name in names:
        path = directory / f"{name}{suffix}"
        path.write_bytes(payload)
        paths.append(path)
    return paths


def polygon(index: int, points: int) -> list[tuple[float, float]]:
    """Return a deterministic normalized polygon."""
    offset = (index % 10) / 100
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    xs = 0.5 + 0.2 * np.cos(angles) + offset
    ys = 0.5 + 0.2 * np.sin(angles) + offset
    return [
        (round(float(x), 6), round(float(y), 6))
        for x, y in zip(xs, ys, strict=True)
    ]


def bbox(index: int) -> tuple[float, float, float, float]:
    """Return a deterministic normalized ``(x, y, w, h)`` box."""
    offset = (index % 10) / 100
    return (0.1 + offset, 0.1 + offset, 0.2, 0.2)


def keypoints(index: int, count: int) -> list[tuple[float, float, int]]:
    """Return deterministic normalized keypoints with visibility."""
    offset = (index % 10) / 100
    return [
        (
            round(0.1 + offset + 0.01 * kpt, 6),
            round(0.1 + offset + 0.01 * kpt, 6),
            2 if kpt % 3 else 1,
        )
        for kpt in range(count)
    ]


@dataclass
class Measurement:
    """Result of parsing one benchmark case.

    Attributes:
        dataset_type: The parsed dataset type.
        features: Features the case exercises.
        images: Number of images in the generated dataset.
        records: Number of records the parser yielded.
        seconds: Fastest wall-clock parse over all repeats.
        peak_mib: Peak memory allocated during a separate traced parse.

    """

    dataset_type: str
    features: tuple[str, ...]
    images: int
    records: int
    seconds: float
    peak_mib: float

    @property
    def records_per_second(self) -> float:
        """Records yielded per second."""
        return self.records / self.seconds if self.seconds else 0.0

    @property
    def images_per_second(self) -> float:
        """Images processed per second."""
        return self.images / self.seconds if self.seconds else 0.0


def drain(case: BenchmarkCase) -> int:
    """Parse ``case`` fully and return how many records it yielded.

    What `measure` times. Deriving the file list from the records is
    deliberately left out: the importer reads the files off a record it
    has already validated, so collecting them here from raw dictionaries
    - a generator call and a `Path` per record rather than per file -
    would charge the parser for the harness's own accounting. It is over
    half the measured time for a format with many annotations per image.
    """
    plugin_type, _, layout = get_parser_plugin(case.source, case.dataset_type)
    result = plugin_type(ParseIssueCollector()).parse(
        case.source, layout, **case.parse_kwargs
    )
    return sum(1 for _ in result.records)


def parse_once(case: BenchmarkCase) -> tuple[int, int, int]:
    """Parse ``case`` fully and return record, file and issue counts.

    The files are collected from the records themselves, the way the
    dataset importer collects them, so that a benchmark run also checks
    the parser reports exactly the files its records name.
    """
    plugin_type, _, layout = get_parser_plugin(case.source, case.dataset_type)
    issues = ParseIssueCollector()
    plugin = plugin_type(issues)
    result = plugin.parse(case.source, layout, **case.parse_kwargs)

    files: dict[Path, None] = {}
    records = 0
    for _split_name, record in result.records:
        records += 1
        for file in _record_files(record):
            files[Path(file)] = None
    return records, len(files), len(issues.messages)


def measure(case: BenchmarkCase, *, repeat: int = 1) -> Measurement:
    """Time ``case`` and measure its peak memory.

    Timing and memory are measured in separate passes because
    `tracemalloc` roughly doubles the time a parse takes; mixing them
    would make the reported throughput useless.
    """
    best = float("inf")
    records = 0
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        records = drain(case)
        best = min(best, time.perf_counter() - start)

    tracemalloc.start()
    try:
        drain(case)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return Measurement(
        dataset_type=case.dataset_type,
        features=case.features,
        images=case.images,
        records=records,
        seconds=best,
        peak_mib=peak / 1024**2,
    )
