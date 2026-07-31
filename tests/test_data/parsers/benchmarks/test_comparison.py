"""Comparing two benchmark reports.

Not marked as a benchmark: it decides whether a release is allowed
through, so it runs with the normal suite rather than only when someone
asks for `-m benchmark`.
"""

import json
from pathlib import Path

import pytest

from tests.test_data.parsers.benchmarks.comparison import (
    compare_reports,
    load_report,
    render_markdown,
)


def _report(**measurements: tuple[float, float]) -> dict:
    """Build a report from ``{dataset_type: (seconds, peak_mib)}``."""
    return {
        "benchmarks": [
            {
                "dataset_type": name,
                "features": [],
                "images": 10,
                "records": 100,
                "seconds": seconds,
                "records_per_second": 100 / seconds if seconds else 0,
                "peak_mib": peak,
            }
            for name, (seconds, peak) in measurements.items()
        ]
    }


def _report_keyed(**measurements: tuple[float, float]) -> dict:
    """Return an in-memory report keyed the way `load_report` returns it."""
    return {
        entry["dataset_type"]: entry
        for entry in _report(**measurements)["benchmarks"]
    }


def _write(path: Path, **measurements: tuple[float, float]) -> Path:
    path.write_text(json.dumps(_report(**measurements)))
    return path


def test_load_report_keys_by_dataset_type(tmp_path: Path):
    path = _write(tmp_path / "r.json", coco=(1.0, 5.0), voc=(2.0, 6.0))
    report = load_report(path)
    assert set(report) == {"coco", "voc"}
    assert report["coco"]["seconds"] == 1.0


def test_regression_is_reported_over_the_threshold():
    """A parser past the threshold fails; one under it does not."""
    baseline = _report_keyed(coco=(1.0, 5.0), voc=(1.0, 5.0))
    current = _report_keyed(coco=(1.2, 5.0), voc=(1.05, 5.0))

    report = compare_reports(baseline, current)
    assert [item.dataset_type for item in report.regressions(10.0)] == ["coco"]
    assert report.regressions(25.0) == []

    coco = next(i for i in report.compared if i.dataset_type == "coco")
    assert coco.seconds_change == pytest.approx(0.2)


def test_improvements_are_never_regressions():
    baseline = _report_keyed(coco=(2.0, 5.0))
    current = _report_keyed(coco=(1.0, 5.0))

    report = compare_reports(baseline, current)
    assert report.regressions(0.0) == []
    assert report.seconds_change == pytest.approx(-0.5)


def test_types_missing_from_either_side_are_listed_not_compared():
    """A parser added or removed cannot be compared, only reported.

    The baseline ref can predate a parser entirely, which must not read as
    a regression and must not silently disappear from the report either.
    """
    baseline = _report_keyed(coco=(1.0, 5.0), gone=(1.0, 5.0))
    current = _report_keyed(coco=(1.0, 5.0), fresh=(1.0, 5.0))

    report = compare_reports(baseline, current)
    assert [item.dataset_type for item in report.compared] == ["coco"]
    assert report.only_current == ["fresh"]
    assert report.only_baseline == ["gone"]


def test_zero_baseline_time_does_not_divide_by_zero():
    """A baseline that measured 0s must not blow up the comparison."""
    baseline = _report_keyed(coco=(0.0, 0.0))
    current = _report_keyed(coco=(1.0, 5.0))

    report = compare_reports(baseline, current)
    assert report.compared[0].seconds_change == 0.0
    assert report.compared[0].peak_change == 0.0
    assert report.seconds_change == 0.0


def test_markdown_states_the_verdict_and_flags_the_offender():
    baseline = _report_keyed(coco=(1.0, 5.0), voc=(1.0, 5.0))
    current = _report_keyed(coco=(1.5, 9.0), voc=(1.0, 5.0))

    markdown = render_markdown(
        compare_reports(baseline, current),
        threshold_percent=10.0,
        baseline_label="main",
        current_label="release/1.2.3",
    )

    assert "`coco`" in markdown
    assert "+50.0%" in markdown
    assert "1 parser(s) more than 10% slower" in markdown
    # The slowest row comes first, so the regression is the one read.
    assert markdown.index("`coco`") < markdown.index("`voc`")
    # Memory is shown but never part of the verdict.
    assert "5.0 → 9.0" in markdown

    passing = render_markdown(
        compare_reports(baseline, current),
        threshold_percent=100.0,
        baseline_label="main",
        current_label="release/1.2.3",
    )
    assert "No parser is more than 100% slower" in passing


def _noisy(seconds: float, relative_stdev: float) -> dict:
    """One measurement carrying a known scatter."""
    return {
        "dataset_type": "clsdir",
        "records": 100,
        "seconds": seconds,
        "peak_mib": 1.0,
        "relative_stdev": relative_stdev,
    }


def test_a_change_within_the_measurement_scatter_is_not_a_regression():
    """The quickest parsers must not fail a release on jitter.

    `clsdir` parses in hundredths of a second, so its timings scatter by
    more than a slow parser ever will. A 30% change on a parser that
    scatters by 25% says nothing, and the threshold alone cannot tell the
    two apart.
    """
    noisy_baseline = {"clsdir": _noisy(1.0, 0.25)}
    noisy = {"clsdir": _noisy(1.3, 0.25)}
    assert compare_reports(noisy_baseline, noisy).regressions(10.0) == []

    # A scattered baseline leaves the comparison just as uncertain, even
    # when the run under test is steady.
    steady = {"clsdir": _noisy(1.3, 0.01)}
    assert compare_reports(noisy_baseline, steady).regressions(10.0) == []

    # Both runs steady: the same 30% is now worth reporting.
    steady_baseline = {"clsdir": _noisy(1.0, 0.01)}
    assert [
        item.dataset_type
        for item in compare_reports(steady_baseline, steady).regressions(10.0)
    ] == ["clsdir"]


def test_the_noise_gate_can_be_turned_off():
    """`noise_factor=0` restores a plain threshold comparison."""
    baseline = {"clsdir": _noisy(1.0, 0.25)}
    current = {"clsdir": _noisy(1.3, 0.25)}

    report = compare_reports(baseline, current)
    assert len(report.regressions(10.0, noise_factor=0.0)) == 1
