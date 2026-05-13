"""Render a ``BenchmarkReport`` as a human-readable text table."""

from __future__ import annotations

import statistics
from collections.abc import Iterable

from .bench_harness import BenchmarkReport, ComparisonRow, QueryMetrics


def _row(label: str, baseline: str, tar_rag: str, width_label: int = 30) -> str:
    return f"{label.ljust(width_label)} {baseline.ljust(20)} {tar_rag}"


def _format_metrics(metrics: QueryMetrics) -> dict[str, str]:
    return {
        "attempts": str(metrics.attempts_made),
        "top_score": f"{metrics.top_score:.2f}",
        "confidence": metrics.confidence,
        "results": str(metrics.result_count),
        "forwarded": str(metrics.chunks_forwarded),
        "chars": f"{metrics.snippet_chars:,}",
        "wall_ms": f"{metrics.wall_time_ms:.1f}",
        "cache": "yes" if metrics.cache_hit else "no",
    }


def _render_row_block(row: ComparisonRow) -> str:
    base = _format_metrics(row.baseline)
    tar = _format_metrics(row.tar_rag)
    lines = [
        f"Query: \"{row.query.text}\"",
        "",
        _row("", "Baseline (unfiltered)", "tar-rag (filtered)"),
        "-" * 80,
        _row("Attempts made", base["attempts"], tar["attempts"]),
        _row("Top score", base["top_score"], tar["top_score"]),
        _row("Confidence tier", base["confidence"], tar["confidence"]),
        _row("Total chunks retrieved", base["results"], tar["results"]),
        _row("Chunks forwarded to LLM", base["forwarded"], tar["forwarded"]),
        _row("Total snippet chars", base["chars"], tar["chars"]),
        _row("Wall time (ms)", base["wall_ms"], tar["wall_ms"]),
        _row("Cache hit", base["cache"], tar["cache"]),
        "-" * 80,
        "",
    ]
    return "\n".join(lines)


def _aggregate(rows: Iterable[ComparisonRow]) -> str:
    rows = list(rows)
    if not rows:
        return ""

    def avg(values: Iterable[float]) -> float:
        values_list = list(values)
        return statistics.mean(values_list) if values_list else 0.0

    base_avg_top = avg(row.baseline.top_score for row in rows)
    tar_avg_top = avg(row.tar_rag.top_score for row in rows)
    base_avg_chars = avg(row.baseline.snippet_chars for row in rows)
    tar_avg_chars = avg(row.tar_rag.snippet_chars for row in rows)
    base_avg_forwarded = avg(row.baseline.chunks_forwarded for row in rows)
    tar_avg_forwarded = avg(row.tar_rag.chunks_forwarded for row in rows)
    tar_high = sum(1 for row in rows if row.tar_rag.confidence == "high")
    tar_med = sum(1 for row in rows if row.tar_rag.confidence == "medium")
    tar_low = sum(1 for row in rows if row.tar_rag.confidence in {"low", "none"})
    attempts_1 = sum(1 for row in rows if row.tar_rag.attempts_made == 1)

    chars_delta_pct = (
        ((tar_avg_chars - base_avg_chars) / base_avg_chars) * 100
        if base_avg_chars
        else 0.0
    )
    forwarded_delta_pct = (
        ((tar_avg_forwarded - base_avg_forwarded) / base_avg_forwarded) * 100
        if base_avg_forwarded
        else 0.0
    )

    total = len(rows)
    lines = [
        f"Benchmark Summary ({total} queries)",
        "-" * 60,
        f"tar-rag confidence:        high={tar_high}/{total}  medium={tar_med}/{total}  low_or_none={tar_low}/{total}",
        f"Avg top score              baseline={base_avg_top:.2f}   tar-rag={tar_avg_top:.2f}",
        f"Avg chunks forwarded       baseline={base_avg_forwarded:.1f}   tar-rag={tar_avg_forwarded:.1f}  ({forwarded_delta_pct:+.1f}%)",
        f"Avg snippet chars          baseline={base_avg_chars:.0f}   tar-rag={tar_avg_chars:.0f}  ({chars_delta_pct:+.1f}%)",
        f"Queries resolved on        attempt 1: {attempts_1}/{total}",
        "-" * 60,
    ]
    return "\n".join(lines)


def render_text_report(report: BenchmarkReport) -> str:
    """Return the human-readable comparison report."""
    blocks = [_render_row_block(row) for row in report.rows]
    blocks.append(_aggregate(report.rows))
    return "\n".join(blocks)
