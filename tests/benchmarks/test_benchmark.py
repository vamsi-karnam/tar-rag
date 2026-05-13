"""Pytest entry-point for the comparison benchmark.

Run only when ``-m benchmark`` is passed::

    pytest tests/benchmarks/ -v -m benchmark

This test asserts the per-query expected confidence/attempts from the
canonical fixture set, validating that the orchestrator's progressive
fallback behaves as documented in Section 3 of the implementation plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .bench_harness import in_memory_adapter_from_fixture, run_comparison
from .bench_report import render_text_report


@pytest.mark.benchmark
def test_offline_benchmark_meets_expectations(
    sample_artifacts: Path,
    mock_scores: dict,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = in_memory_adapter_from_fixture(mock_scores)
    report = run_comparison(artifacts_dir=sample_artifacts, adapter=adapter)
    text = render_text_report(report)
    # Print the full report so `-s` flag shows it; capsys keeps it
    # accessible for debugging without flooding the default test output.
    print("\n" + text)

    failures: list[str] = []
    for row in report.rows:
        query = row.query
        expected_conf: set[str] = query.expected_confidence
        actual_conf = row.tar_rag.confidence
        if expected_conf and actual_conf not in expected_conf:
            failures.append(
                f"{query.id}: expected confidence in {sorted(expected_conf)} "
                f"got {actual_conf!r}"
            )
        if query.max_attempts is not None and row.tar_rag.attempts_made > query.max_attempts:
            failures.append(
                f"{query.id}: attempts_made={row.tar_rag.attempts_made} "
                f"exceeds max_attempts={query.max_attempts}"
            )
    assert not failures, "Benchmark expectations failed:\n" + "\n".join(failures)


@pytest.mark.benchmark
def test_offline_benchmark_filters_low_confidence_chunks(
    sample_artifacts: Path, mock_scores: dict
) -> None:
    """For OOC queries tar-rag must forward zero chunks (confidence gate),
    while the unfiltered baseline still spends tokens on weak matches.

    For specific queries tar-rag must achieve a higher avg top score than
    baseline — that's the structural-filter advantage in numerical form.
    """
    adapter = in_memory_adapter_from_fixture(mock_scores)
    report = run_comparison(artifacts_dir=sample_artifacts, adapter=adapter)

    ooc_rows = [row for row in report.rows if row.query.category == "ooc"]
    assert ooc_rows, "expected OOC queries in the canonical fixture set"
    tar_ooc_forwarded = sum(row.tar_rag.chunks_forwarded for row in ooc_rows)
    base_ooc_forwarded = sum(row.baseline.chunks_forwarded for row in ooc_rows)
    assert tar_ooc_forwarded == 0, (
        f"tar-rag forwarded {tar_ooc_forwarded} OOC chunks — should gate them out"
    )
    assert base_ooc_forwarded > 0, (
        "baseline should still forward weak OOC chunks (no confidence gate)"
    )

    specific_rows = [row for row in report.rows if row.query.category == "specific"]
    assert specific_rows, "expected specific queries in the canonical fixture set"
    tar_specific_top = sum(row.tar_rag.top_score for row in specific_rows) / len(specific_rows)
    base_specific_top = sum(row.baseline.top_score for row in specific_rows) / len(specific_rows)
    assert tar_specific_top > base_specific_top, (
        f"tar-rag avg top score on specific queries ({tar_specific_top:.2f}) should "
        f"exceed baseline ({base_specific_top:.2f})"
    )
