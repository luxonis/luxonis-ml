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


#: Marks a row carries in either format. A change past the threshold is
#: `❗`, one past half of it `⚠️`, anything else `✅`.
STATUS_GOOD = "✅"
STATUS_NEAR = "⚠️"
STATUS_OVER = "❗"

#: Colour each mark carries in a rich table, so the number and the mark
#: say the same thing. Named colours rather than hex: GitHub's MathJax
#: renders `\textcolor{green}` and leaves `\textcolor{#1a7f37}` as markup.
_STATUS_COLOUR = {
    STATUS_GOOD: "green",
    STATUS_NEAR: "orange",
    STATUS_OVER: "red",
}


def _status(
    change_percent: float, *, threshold_percent: float, regressed: bool
) -> str:
    """Return the mark a change of this size earns.

    Args:
        change_percent: How much slower the parser got, in percent.
        threshold_percent: Threshold a regression has to clear.
        regressed: Whether the change counted as a regression, which
            takes the measured noise into account as well.

    Returns:
        One of `STATUS_OVER`, `STATUS_NEAR` or `STATUS_GOOD`.

    """
    if regressed:
        return STATUS_OVER
    if change_percent > threshold_percent / 2.0:
        return STATUS_NEAR
    return STATUS_GOOD


def _math(body: str, colour: str | None = None) -> str:
    r"""Wrap a rendered value as GitHub inline math.

    A closing `$` may be followed by punctuation but never by a letter:
    `$0.4$s` is left as literal text, dollar signs and all. So no unit
    ever trails the delimiter here and the column header carries it
    instead. A `%` has the mirrored problem and has to stay *outside* the
    math - Markdown strips the backslash from `\\%` before MathJax sees
    it, and the bare `%` left behind opens a comment that swallows the
    rest of the expression.
    """
    if colour is not None:
        body = f"\\textcolor{{{colour}}}{{{body}}}"
    return f"${body}$"


def _grouped(value: int) -> str:
    """Digit-group an integer for math, where a bare comma spaces wrong."""
    return f"{value:,}".replace(",", "{,}")


def render_markdown(
    report: ComparisonReport,
    *,
    threshold_percent: float,
    baseline_label: str,
    current_label: str,
    rich: bool = True,
) -> str:
    """Render a comparison as a Markdown report.

    Args:
        report: Comparison to render.
        threshold_percent: Threshold the verdict line quotes.
        baseline_label: Name of the baseline ref, for the table header.
        current_label: Name of the ref under test.
        rich: Whether to set the numbers as math and colour them. A pull
            request comment renders both; a job summary renders neither
            and would show the markup itself, so it asks for the plain
            form. The marks and the monospaced names render either way.

    Returns:
        Markdown, ready for a job summary or a pull request comment.

    """
    regressions = report.regressions(threshold_percent)
    regressed = {item.dataset_type for item in regressions}
    lines = [
        "### Parser benchmarks",
        "",
        f"`{current_label}` against `{baseline_label}`, "
        f"regression threshold {threshold_percent:g}%.",
        "",
        f"|  | dataset type | records | {baseline_label} (s) "
        f"| {current_label} (s) | change | noise | peak MiB |",
        "| :-: | --- | --: | --: | --: | --: | --: | --: |",
    ]
    slowest_first = sorted(
        report.compared, key=lambda item: item.seconds_change, reverse=True
    )
    for item in slowest_first:
        change = item.seconds_change * 100.0
        mark = _status(
            change,
            threshold_percent=threshold_percent,
            regressed=item.dataset_type in regressed,
        )
        noise = item.noise * 100.0
        if rich:
            peak = (
                f"{item.baseline_peak_mib:.1f} \\to "
                f"{item.current_peak_mib:.1f}"
            )
            cells = (
                mark,
                f"`{item.dataset_type}`",
                _math(_grouped(item.records)),
                _math(f"{item.baseline_seconds:.3f}"),
                _math(f"{item.current_seconds:.3f}"),
                _math(f"{change:+.1f}", _STATUS_COLOUR[mark]) + "%",
                _math(f"\\pm{noise:.1f}") + "%",
                _math(peak),
            )
        else:
            cells = (
                mark,
                f"`{item.dataset_type}`",
                f"{item.records:,}",
                f"{item.baseline_seconds:.3f}",
                f"{item.current_seconds:.3f}",
                f"{change:+.1f}%",
                f"±{noise:.1f}%",
                f"{item.baseline_peak_mib:.1f} → {item.current_peak_mib:.1f}",
            )
        lines.append(f"| {' | '.join(cells)} |")

    total_change = report.seconds_change * 100.0
    total_mark = _status(
        total_change,
        threshold_percent=threshold_percent,
        regressed=bool(regressions),
    )
    if rich:
        total_seconds = _math(
            f"{report.baseline_seconds:.2f} \\to {report.current_seconds:.2f}"
        )
        total_percent = _math(
            f"{total_change:+.1f}", _STATUS_COLOUR[total_mark]
        )
        total = f"**Total: {total_seconds} seconds ({total_percent}%)**"
    else:
        total = (
            f"**Total: {report.baseline_seconds:.2f}s → "
            f"{report.current_seconds:.2f}s ({total_change:+.1f}%)**"
        )

    lines += ["", total, ""]

    if regressions:
        listed = ", ".join(
            f"`{item.dataset_type}` {_percent(item.seconds_change)}"
            for item in regressions
        )
        lines.append(
            f"{STATUS_OVER} {len(regressions)} parser(s) more than "
            f"{threshold_percent:g}% slower: {listed}."
        )
    else:
        lines.append(
            f"{STATUS_GOOD} No parser is more than {threshold_percent:g}% "
            f"slower than `{baseline_label}`."
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
        f"<sub>{STATUS_GOOD} under half the threshold · {STATUS_NEAR} past "
        f"half of it · {STATUS_OVER} regressed. A parser counts as "
        "regressed only when it is past the threshold *and* more than "
        "twice the two runs' combined scatter (`noise`), so the quickest "
        "parsers cannot fail a release on timing jitter alone. Only parse "
        "time decides the verdict: `peak MiB` is shown because a large "
        "move is worth seeing, but it is stable only within one run of "
        "the suite, so it is neither compared nor marked.</sub>",
    ]
    return "\n".join(lines) + "\n"


def render_single_markdown(
    measurements: dict[str, dict[str, Any]],
    *,
    label: str,
    baseline_label: str = "",
    rich: bool = True,
) -> str:
    """Render one report, with nothing to compare it against.

    The shape a run takes before the baseline ref carries the benchmark
    suite. No row can be marked, because there is no threshold to mark it
    against; the numbers are all the report has to say.

    Args:
        measurements: Report keyed by dataset type, from `load_report`.
        label: Name of the ref that was measured.
        baseline_label: Name of the ref that measured nothing, when there
            was one. Named in the report so the reader knows a comparison
            was meant to happen.
        rich: Whether to set the numbers as math, as in `render_markdown`.

    Returns:
        Markdown, ready for a job summary or a pull request comment.

    """
    if baseline_label:
        preamble = (
            f"`{label}`. No results for `{baseline_label}` - it has no "
            "benchmark suite to run, so there is nothing to compare "
            "against."
        )
    else:
        preamble = f"`{label}`, with nothing to compare against."

    lines = [
        "### Parser benchmarks",
        "",
        preamble,
        "",
        "| dataset type | images | records | seconds | records/s | peak MiB |",
        "| --- | --: | --: | --: | --: | --: |",
    ]
    for name in sorted(measurements):
        entry = measurements[name]
        if rich:
            cells = (
                f"`{name}`",
                _math(_grouped(entry["images"])),
                _math(_grouped(entry["records"])),
                _math(f"{entry['seconds']:.3f}"),
                _math(_grouped(round(entry["records_per_second"]))),
                _math(f"{entry['peak_mib']:.1f}"),
            )
        else:
            cells = (
                f"`{name}`",
                f"{entry['images']:,}",
                f"{entry['records']:,}",
                f"{entry['seconds']:.3f}",
                f"{round(entry['records_per_second']):,}",
                f"{entry['peak_mib']:.1f}",
            )
        lines.append(f"| {' | '.join(cells)} |")

    return "\n".join(lines) + "\n"
