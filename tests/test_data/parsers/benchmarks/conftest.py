"""Fixtures and reporting for the parser benchmarks."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from _pytest.terminal import TerminalReporter

from tests.test_data.parsers.benchmarks.harness import (
    BENCHMARK_DATASETS,
    BenchmarkCase,
    Measurement,
    Scale,
)

MEASUREMENTS: list[Measurement] = []


@pytest.fixture(scope="session")
def benchmark_scale(request: pytest.FixtureRequest) -> Scale:
    """Return the dataset size requested on the command line."""
    factor: float = request.config.getoption("--benchmark-scale") or 1.0
    base = Scale()
    return Scale(
        images=max(1, round(base.images * factor)),
        annotations_per_image=base.annotations_per_image,
        classes=base.classes,
        keypoints=base.keypoints,
        polygon_points=base.polygon_points,
        image_size=base.image_size,
    )


@pytest.fixture(scope="session")
def benchmark_repeat(request: pytest.FixtureRequest) -> int:
    """Return how many times each parse is timed."""
    repeat: int = request.config.getoption("--benchmark-repeat") or 1
    return repeat


@pytest.fixture(scope="session")
def benchmark_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the directory holding all generated benchmark datasets."""
    return tmp_path_factory.mktemp("parser_benchmarks")


@pytest.fixture(scope="session")
def build_benchmark_case(
    benchmark_root: Path, benchmark_scale: Scale
) -> Callable[[str], BenchmarkCase]:
    """Return a builder that generates and caches one dataset per type."""
    cache: dict[str, BenchmarkCase] = {}

    def build(dataset_type: str) -> BenchmarkCase:
        if dataset_type not in cache:
            root = benchmark_root / dataset_type.replace("/", "_")
            root.mkdir(parents=True, exist_ok=True)
            builder = BENCHMARK_DATASETS[dataset_type]
            cache[dataset_type] = builder(root, benchmark_scale)
        return cache[dataset_type]

    return build


def record_measurement(measurement: Measurement) -> None:
    """Store a measurement for the end-of-session report."""
    MEASUREMENTS.append(measurement)


def _load_baseline(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {entry["dataset_type"]: entry for entry in payload["benchmarks"]}


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, config: pytest.Config
) -> None:
    """Print the benchmark table and write the JSON report."""
    if not MEASUREMENTS:
        return

    baseline_option = config.getoption("--benchmark-compare")
    baseline = _load_baseline(Path(baseline_option)) if baseline_option else {}

    terminalreporter.write_sep("=", "parser benchmarks")
    header = (
        f"{'dataset type':<38}{'images':>8}{'records':>9}"
        f"{'sec':>9}{'rec/s':>11}{'peak MiB':>10}"
    )
    if baseline:
        header += f"{'vs base':>10}"
    terminalreporter.write_line(header)

    for measurement in sorted(
        MEASUREMENTS, key=lambda item: item.dataset_type
    ):
        line = (
            f"{measurement.dataset_type:<38}"
            f"{measurement.images:>8}"
            f"{measurement.records:>9}"
            f"{measurement.seconds:>9.3f}"
            f"{measurement.records_per_second:>11,.0f}"
            f"{measurement.peak_mib:>10.1f}"
        )
        previous = baseline.get(measurement.dataset_type)
        if previous:
            change = (
                measurement.seconds / previous["seconds"] - 1
                if previous["seconds"]
                else 0.0
            )
            line += f"{change * 100:>9.1f}%"
        terminalreporter.write_line(line)

    destination = config.getoption("--benchmark-json")
    if destination:
        payload = {
            "benchmarks": [
                {
                    "dataset_type": measurement.dataset_type,
                    "features": list(measurement.features),
                    "images": measurement.images,
                    "records": measurement.records,
                    "seconds": measurement.seconds,
                    "records_per_second": measurement.records_per_second,
                    "peak_mib": measurement.peak_mib,
                }
                for measurement in sorted(
                    MEASUREMENTS, key=lambda item: item.dataset_type
                )
            ]
        }
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        terminalreporter.write_line(f"benchmark report written to {path}")
