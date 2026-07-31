r"""Compare two parser benchmark reports and fail on a regression.

Both reports are produced by ``pytest -m benchmark ... --benchmark-json``,
each on its own ref. CI runs the two refs as a matrix so they are measured
the same way, then calls this::

    python tools/compare_benchmarks.py base.json head.json \
        --threshold 25 --output comparison.md

Exits non-zero when a parser is more than ``--threshold`` percent slower
than the baseline, which is what fails the release check.
"""

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from rich import print

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_data.parsers.benchmarks.comparison import (
    compare_reports,
    load_report,
    render_markdown,
)

app = App(name="compare_benchmarks")


@app.default
def compare(
    baseline: Path,
    current: Path,
    *,
    threshold: Annotated[float, Parameter(alias="-t")] = 25.0,
    output: Annotated[Path | None, Parameter(alias="-o")] = None,
    baseline_label: str = "baseline",
    current_label: str = "current",
) -> None:
    """Compare two benchmark reports.

    Args:
        baseline: Report to compare against.
        current: Report under test.
        threshold: How much slower a parser may get, in percent, before
            the comparison fails. Two runs of identical code differ by up
            to ~12% on the parsers quick enough for scheduling noise to
            show, so the default leaves room for that.
        output: Where to write the Markdown report. Printed either way.
        baseline_label: Name of the baseline ref, for the table header.
        current_label: Name of the ref under test.

    Raises:
        SystemExit: If a parser regressed by more than ``threshold``, or
            if the two reports have no dataset type in common.

    """
    report = compare_reports(load_report(baseline), load_report(current))
    markdown = render_markdown(
        report,
        threshold_percent=threshold,
        baseline_label=baseline_label,
        current_label=current_label,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown)
    print(markdown)

    if not report.compared:
        raise SystemExit(
            "The two reports share no dataset type, so there is nothing to "
            "compare. Does the baseline ref have the benchmark suite?"
        )

    regressions = report.regressions(threshold)
    if regressions:
        raise SystemExit(
            f"{len(regressions)} parser(s) more than {threshold:g}% slower "
            f"than {baseline_label}."
        )


if __name__ == "__main__":
    app()
