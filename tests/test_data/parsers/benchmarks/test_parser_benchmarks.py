"""Throughput benchmarks for every registered parser.

Each parser is fed a large synthetic dataset exercising every feature it
supports. The parse is timed and its output verified, so a benchmark run
also proves the parser still produces the expected number of records at
a scale the functional tests never reach.
"""

from collections.abc import Callable

import pytest

from tests.test_data.parsers.benchmarks.conftest import record_measurement
from tests.test_data.parsers.benchmarks.datasets import BENCHMARKED_TYPES
from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    measure,
    parse_once,
)

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("dataset_type", BENCHMARKED_TYPES)
def test_parser_throughput(
    dataset_type: str,
    build_benchmark_case: Callable[[str], BenchmarkCase],
    benchmark_repeat: int,
    benchmark_aggregate: str,
):
    case = build_benchmark_case(dataset_type)
    records, files, issues = parse_once(case)

    assert records == case.expected_records
    assert files == case.expected_files
    assert issues == case.expected_issues

    record_measurement(
        measure(case, repeat=benchmark_repeat, aggregate=benchmark_aggregate)
    )
