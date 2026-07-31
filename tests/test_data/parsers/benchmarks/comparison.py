"""Compare two benchmark reports and decide whether one regressed.

A report is the JSON that ``--benchmark-json`` writes. A release is
checked by running the suite on the release ref and on the ref before it
and comparing the two, so the verdict has to be a threshold rather than
someone reading a table.

Only parse time decides the verdict. Peak allocation is reported next to
it because a large move is worth seeing, but it is not compared: the peak
is stable within one run of the suite and not across differently measured
ones, so gating on it would fail releases for the measurement rather than
for the code.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Key holding the per-parser measurements inside a report.
BENCHMARKS_KEY = "benchmarks"


@dataclass(frozen=True)
class Comparison:
    """One dataset type as measured in both reports.

    Attributes:
        dataset_type: Parser the measurements belong to.
        records: Records the current run yielded.
        baseline_seconds: Parse time in the baseline report.
        current_seconds: Parse time in the current report.
        baseline_peak_mib: Peak allocation in the baseline report.
        current_peak_mib: Peak allocation in the current report.

    """

    dataset_type: str
    records: int
    baseline_seconds: float
    current_seconds: float
    baseline_peak_mib: float
    current_peak_mib: float
    baseline_relative_stdev: float = 0.0
    current_relative_stdev: float = 0.0

    @property
    def noise(self) -> float:
        """Scatter the two runs bring between them, as a fraction.

        The parsers that finish in hundredths of a second scatter by more
        than a slow one ever will, whatever the threshold is set to, so a
        change is believed only once it clears their own noise.
        """
        return (
            self.baseline_relative_stdev**2 + self.current_relative_stdev**2
        ) ** 0.5

    @property
    def seconds_change(self) -> float:
        """Fraction slower than the baseline; negative is faster."""
        if not self.baseline_seconds:
            return 0.0
        return self.current_seconds / self.baseline_seconds - 1.0

    @property
    def peak_change(self) -> float:
        """Fraction more memory than the baseline."""
        if not self.baseline_peak_mib:
            return 0.0
        return self.current_peak_mib / self.baseline_peak_mib - 1.0


@dataclass(frozen=True)
class ComparisonReport:
    """What two reports say about the same suite.

    Attributes:
        compared: Dataset types measured in both reports.
        only_current: Types the current report added.
        only_baseline: Types the current report no longer measures.

    """

    compared: list[Comparison]
    only_current: list[str]
    only_baseline: list[str]

    @property
    def baseline_seconds(self) -> float:
        """Total baseline parse time over the compared types."""
        return sum(item.baseline_seconds for item in self.compared)

    @property
    def current_seconds(self) -> float:
        """Total current parse time over the compared types."""
        return sum(item.current_seconds for item in self.compared)

    @property
    def seconds_change(self) -> float:
        """Fraction slower over the compared types as a whole."""
        if not self.baseline_seconds:
            return 0.0
        return self.current_seconds / self.baseline_seconds - 1.0

    def regressions(
        self, threshold_percent: float, *, noise_factor: float = 2.0
    ) -> list["Comparison"]:
        """Return the types that got slower for a believable reason.

        A type has to clear two bars: the threshold, and its own measured
        scatter. The second one keeps the parsers that finish in
        hundredths of a second from failing a release whenever the runner
        is briefly busy with something else.

        Args:
            threshold_percent: How much slower a parser may get before it
                counts as a regression, in percent.
            noise_factor: Multiples of the two runs' combined scatter the
                change has to exceed as well.

        Returns:
            The offending comparisons, slowest first.

        """
        over = [
            item
            for item in self.compared
            if item.seconds_change * 100.0 > threshold_percent
            and item.seconds_change > noise_factor * item.noise
        ]
        return sorted(over, key=lambda item: item.seconds_change, reverse=True)


def load_report(path: Path | str) -> dict[str, dict[str, Any]]:
    """Read a benchmark report, keyed by dataset type.

    Args:
        path: JSON file written by ``--benchmark-json``.

    Returns:
        Each measurement, keyed by its dataset type.

    """
    payload = json.loads(Path(path).read_text())
    return {entry["dataset_type"]: entry for entry in payload[BENCHMARKS_KEY]}


def compare_reports(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> ComparisonReport:
    """Pair up two reports by dataset type.

    Args:
        baseline: Report to compare against.
        current: Report under test.

    Returns:
        The pairing, plus the types only one of them measured.

    """
    compared = [
        Comparison(
            dataset_type=name,
            records=current[name]["records"],
            baseline_seconds=baseline[name]["seconds"],
            current_seconds=current[name]["seconds"],
            baseline_peak_mib=baseline[name]["peak_mib"],
            current_peak_mib=current[name]["peak_mib"],
            baseline_relative_stdev=baseline[name].get("relative_stdev", 0.0),
            current_relative_stdev=current[name].get("relative_stdev", 0.0),
        )
        for name in sorted(current)
        if name in baseline
    ]
    return ComparisonReport(
        compared=compared,
        only_current=sorted(set(current) - set(baseline)),
        only_baseline=sorted(set(baseline) - set(current)),
    )


def _percent(change: float) -> str:
    """Format a fractional change with an explicit sign."""
    return f"{change * 100.0:+.1f}%"


def render_markdown(
    report: ComparisonReport,
    *,
    threshold_percent: float,
    baseline_label: str,
    current_label: str,
) -> str:
    """Render a comparison as a Markdown report.

    Args:
        report: Comparison to render.
        threshold_percent: Threshold the verdict line quotes.
        baseline_label: Name of the baseline ref, for the table header.
        current_label: Name of the ref under test.

    Returns:
        Markdown, ready for a job summary or a pull request comment.

    """
    regressions = report.regressions(threshold_percent)
    lines = [
        "### Parser benchmarks",
        "",
        f"`{current_label}` against `{baseline_label}`, "
        f"regression threshold {threshold_percent:g}%.",
        "",
        f"| dataset type | records | {baseline_label} | {current_label} "
        "| change | noise | peak MiB |",
        "| --- | --: | --: | --: | --: | --: | --: |",
    ]
    slowest_first = sorted(
        report.compared, key=lambda item: item.seconds_change, reverse=True
    )
    for item in slowest_first:
        flag = " ⚠️" if item in regressions else ""
        lines.append(
            f"| `{item.dataset_type}` | {item.records:,} "
            f"| {item.baseline_seconds:.3f}s | {item.current_seconds:.3f}s "
            f"| {_percent(item.seconds_change)}{flag} "
            f"| ±{item.noise * 100.0:.1f}% "
            f"| {item.baseline_peak_mib:.1f} → {item.current_peak_mib:.1f} |"
        )

    lines += [
        "",
        f"**Total: {report.baseline_seconds:.2f}s → "
        f"{report.current_seconds:.2f}s "
        f"({_percent(report.seconds_change)})**",
        "",
    ]

    if regressions:
        listed = ", ".join(
            f"`{item.dataset_type}` {_percent(item.seconds_change)}"
            for item in regressions
        )
        lines.append(
            f"⚠️ {len(regressions)} parser(s) more than "
            f"{threshold_percent:g}% slower: {listed}."
        )
    else:
        lines.append(
            f"✅ No parser is more than {threshold_percent:g}% slower than "
            f"`{baseline_label}`."
        )

    if report.only_current:
        lines.append(
            f"\nOnly in `{current_label}`: {', '.join(report.only_current)}."
        )
    if report.only_baseline:
        lines.append(
            f"\nOnly in `{baseline_label}`: {', '.join(report.only_baseline)}."
        )

    lines += [
        "",
        "<sub>A parser counts as regressed only when it is past the "
        "threshold *and* more than twice the two runs' combined scatter "
        "(`noise`), so the quickest parsers cannot fail a release on "
        "timing jitter alone. Only parse time decides the verdict: "
        "`peak MiB` is shown because a large move is worth seeing, but "
        "it is stable only within one run of the suite, so it is not "
        "compared.</sub>",
    ]
    return "\n".join(lines) + "\n"
