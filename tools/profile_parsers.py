"""Profile parsers and prove that optimizing them changed no output.

The benchmark datasets under ``tests/test_data/parsers/benchmarks`` are
deterministic, so the same dataset can be regenerated before and after a
change and the parser output compared exactly::

    python tools/profile_parsers.py digests before.json
    # ... optimize a parser ...
    python tools/profile_parsers.py compare before.json

    python tools/profile_parsers.py profile --dataset-type yolov8
"""

import cProfile
import io
import json
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich import print as rich_print

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_data.parsers.benchmarks.datasets import (
    BENCHMARKED_TYPES,
)
from tests.test_data.parsers.benchmarks.equivalence import (
    compare as compare_digests,
)
from tests.test_data.parsers.benchmarks.equivalence import (
    digest,
)
from tests.test_data.parsers.benchmarks.harness import (
    BENCHMARK_DATASETS,
    BenchmarkCase,
    Scale,
    measure,
    parse_once,
)

app = App(name="profile_parsers")

DatasetTypes = Annotated[list[str] | None, Parameter(alias="-t")]
Images = Annotated[int, Parameter(alias="-i")]

# Digests are generated at a fixed absolute location rather than in a
# temporary directory. Fixed, because a parser that downloads a referenced
# image names it after a hash of its URL, so a moving dataset root would
# make the output differ between two runs of identical code. Absolute,
# because the digest identifies a file by its path relative to the dataset
# root - against a relative root that never matches the absolute paths
# parsers emit, every file would collapse to its bare name and files of
# different splits would become indistinguishable.
DIGEST_ROOT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "parser_digests"
)


def _selected(dataset_types: list[str] | None) -> list[str]:
    if not dataset_types:
        return list(BENCHMARKED_TYPES)
    unknown = sorted(set(dataset_types) - set(BENCHMARKED_TYPES))
    if unknown:
        raise SystemExit(
            f"Unknown dataset types: {unknown}. "
            f"Available: {sorted(BENCHMARKED_TYPES)}"
        )
    return dataset_types


def _build(root: Path, dataset_type: str, scale: Scale) -> BenchmarkCase:
    directory = root / dataset_type.replace("/", "_")
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    return BENCHMARK_DATASETS[dataset_type](directory, scale)


@app.command
def digests(
    output: Path,
    dataset_type: DatasetTypes = None,
    images: Images = 200,
) -> None:
    """Write an exact-output digest for each benchmark dataset.

    Args:
        output: Where to write the digests.
        dataset_type: Dataset types to include; defaults to all.
        images: Images per split in the generated datasets.

    """
    scale = Scale(images=images)
    digests_by_type = {}
    for name in _selected(dataset_type):
        case = _build(DIGEST_ROOT, name, scale)
        digests_by_type[name] = digest(case)
        rich_print(
            f"[green]digested[/green] {name} "
            f"({len(digests_by_type[name]['records'])} records)"
        )
    # The scale travels with the digests: comparing runs of different
    # sizes would report every parser as changed, which is a false
    # positive on the very gate this tool provides.
    payload = {"images": images, "digests": digests_by_type}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rich_print(f"wrote {output}")


@app.command
def compare(
    baseline: Path,
    dataset_type: DatasetTypes = None,
    images: Images = 200,
) -> None:
    """Compare current parser output against a saved digest file.

    Args:
        baseline: Digest file written by the ``digests`` command.
        dataset_type: Dataset types to include; defaults to all.
        images: Images per split; must match the baseline run.

    """
    payload = json.loads(baseline.read_text())
    if "digests" in payload:
        saved = payload["digests"]
        baseline_images = payload.get("images")
        if baseline_images != images:
            raise SystemExit(
                f"The baseline was taken with --images {baseline_images}, "
                f"but this run uses --images {images}. The comparison "
                "would report every parser as changed."
            )
    else:
        # A digest file from before the scale was recorded next to the
        # digests; the caller has to match the sizes on their own.
        saved = payload
    scale = Scale(images=images)
    failures = 0
    for name in _selected(dataset_type):
        if name not in saved:
            rich_print(f"[yellow]skipped[/yellow] {name}: not in the baseline")
            continue
        case = _build(DIGEST_ROOT, name, scale)
        differences = compare_digests(saved[name], digest(case))
        if differences:
            failures += 1
            rich_print(f"[red]CHANGED[/red] {name}: {', '.join(differences)}")
        else:
            rich_print(f"[green]identical[/green] {name}")
    if failures:
        raise SystemExit(f"{failures} parser(s) changed their output.")


@app.command
def profile(
    dataset_type: DatasetTypes = None,
    images: Images = 400,
    rows: int = 12,
) -> None:
    """Profile parsing and print the hottest functions.

    Args:
        dataset_type: Dataset types to profile; defaults to all.
        images: Images per split in the generated datasets.
        rows: How many profile rows to print per parser.

    """
    scale = Scale(images=images)
    with tempfile.TemporaryDirectory() as tmp:
        for name in _selected(dataset_type):
            case = _build(Path(tmp), name, scale)
            profiler = cProfile.Profile()
            profiler.enable()
            parse_once(case)
            profiler.disable()

            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats(
                "tottime"
            ).print_stats(rows)
            rich_print(f"[bold]{name}[/bold]")
            rich_print("\n".join(stream.getvalue().splitlines()[4:]))


@app.command
def benchmark(
    dataset_type: DatasetTypes = None,
    images: Images = 2000,
    repeat: int = 1,
) -> None:
    """Time each parser and print a throughput table.

    Args:
        dataset_type: Dataset types to time; defaults to all.
        images: Images per split in the generated datasets.
        repeat: How many times each parse is timed; the fastest wins.

    """
    scale = Scale(images=images)
    with tempfile.TemporaryDirectory() as tmp:
        rich_print(
            f"{'dataset type':<38}{'images':>8}{'records':>9}"
            f"{'gen s':>8}{'sec':>9}{'rec/s':>11}{'peak MiB':>10}"
        )
        for name in _selected(dataset_type):
            start = time.perf_counter()
            case = _build(Path(tmp), name, scale)
            generated = time.perf_counter() - start
            result = measure(case, repeat=repeat)
            rich_print(
                f"{name:<38}{result.images:>8}{result.records:>9}"
                f"{generated:>8.1f}{result.seconds:>9.3f}"
                f"{result.records_per_second:>11,.0f}"
                f"{result.peak_mib:>10.1f}"
            )


if __name__ == "__main__":
    app()
