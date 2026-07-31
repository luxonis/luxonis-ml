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
    render_single_markdown,
)

app = App(name="compare_benchmarks")


def _write(path: Path | None, markdown: str) -> None:
    """Write ``markdown`` to ``path``, if there is one."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown)


@app.default
def compare(
    baseline: Path,
    current: Path,
    *,
    threshold: Annotated[float, Parameter(alias="-t")] = 25.0,
    output: Annotated[Path | None, Parameter(alias="-o")] = None,
    plain_output: Path | None = None,
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
        output: Where to write the Markdown report, with the numbers set
            as math and coloured. Printed either way.
        plain_output: Where to write the same report without math or
            colour, for a job summary, which renders neither and would
            show the markup itself.
        baseline_label: Name of the baseline ref, for the table header.
        current_label: Name of the ref under test.

    Raises:
        SystemExit: If a parser regressed by more than ``threshold``, or
            if the two reports have no dataset type in common.

    """
    report = compare_reports(load_report(baseline), load_report(current))

    def render(*, rich: bool) -> str:
        return render_markdown(
            report,
            threshold_percent=threshold,
            baseline_label=baseline_label,
            current_label=current_label,
            rich=rich,
        )

    markdown = render(rich=True)
    _write(output, markdown)
    _write(plain_output, render(rich=False))
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


@app.command
def render(
    report: Path,
    *,
    output: Annotated[Path | None, Parameter(alias="-o")] = None,
    plain_output: Path | None = None,
    label: str = "current",
    baseline_label: str = "",
) -> None:
    """Render a single benchmark report, with nothing to compare it to.

    What CI publishes when only one ref measured anything - a baseline
    older than the benchmark suite, or a run asked for one ref alone.

    Args:
        report: Report to render.
        output: Where to write the Markdown, with the numbers set as
            math. Printed either way.
        plain_output: Where to write the same report without math, for a
            job summary.
        label: Name of the ref that was measured.
        baseline_label: Name of the ref that measured nothing, when a
            comparison was meant to happen.

    """

    def rendered(*, rich: bool) -> str:
        return render_single_markdown(
            load_report(report),
            label=label,
            baseline_label=baseline_label,
            rich=rich,
        )

    markdown = rendered(rich=True)
    _write(output, markdown)
    _write(plain_output, rendered(rich=False))
    print(markdown)


if __name__ == "__main__":
    app()
