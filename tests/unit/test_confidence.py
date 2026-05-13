"""Unit tests for ConfidenceScorer + ConfidenceConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from tar_rag import (
    ConfidenceConfig,
    ConfidenceScorer,
    ConfidenceThresholds,
    SearchResult,
)


def _r(score: float) -> SearchResult:
    return SearchResult(score=score, snippet="", metadata={})


def test_empty_results_score_none() -> None:
    assert ConfidenceScorer().score([]) == "none"


def test_high_via_single_threshold() -> None:
    assert ConfidenceScorer().score([_r(0.80)]) == "high"


def test_high_via_combo_threshold() -> None:
    # top=0.70 (>=0.68), second=0.60 (>=0.55) -> high
    assert ConfidenceScorer().score([_r(0.70), _r(0.60)]) == "high"


def test_medium_when_top_above_medium_min_but_combo_fails() -> None:
    # top=0.65 (<0.78, >=0.58, combo second=0.40 < 0.55)
    assert ConfidenceScorer().score([_r(0.65), _r(0.40)]) == "medium"


def test_low_when_top_below_medium_min() -> None:
    assert ConfidenceScorer().score([_r(0.50)]) == "low"


def test_thresholds_are_configurable() -> None:
    strict = ConfidenceScorer(ConfidenceThresholds(high_single=0.95, medium_min=0.80))
    assert strict.score([_r(0.85)]) == "medium"
    assert strict.score([_r(0.96)]) == "high"


def test_config_round_trip(tmp_path: Path) -> None:
    config = ConfidenceConfig(
        thresholds=ConfidenceThresholds(
            high_single=0.9, high_combo=0.8, high_combo_second=0.7, medium_min=0.6
        ),
    )
    path = tmp_path / "confidence_config.json"
    config.save(path)
    loaded = ConfidenceConfig.load(path)
    assert loaded.thresholds.high_single == 0.9
    assert loaded.thresholds.medium_min == 0.6


def test_default_config_carries_universal_thresholds_and_tuning_note() -> None:
    config = ConfidenceConfig()
    # Default tuning values — generic starting points calibrated against
    # text-embedding-3-large. Not optimal for any particular corpus.
    assert config.thresholds.high_single == pytest.approx(0.78)
    assert config.thresholds.high_combo == pytest.approx(0.68)
    assert config.thresholds.high_combo_second == pytest.approx(0.55)
    assert config.thresholds.medium_min == pytest.approx(0.58)
    # The caveat note must be prominent — tuning is required per embedding model.
    note_lower = config.notes.lower()
    assert "tunable" in note_lower
    assert "embedding model" in note_lower
