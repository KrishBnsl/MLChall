from __future__ import annotations

import pandas as pd
import pytest

from mlchallenge.metrics import (
    entity_fbeta,
    macro_entity_fbeta,
    predictions_at_threshold,
    tune_threshold_from_oof,
)


def test_entity_f0_5_matches_problem_statement_example() -> None:
    predicted = {"S2-00047", "S2-00193", "S3-00812"}
    expected = {"S2-00047", "S3-00812"}
    assert entity_fbeta(predicted, expected, beta=0.5) == pytest.approx(5 / 7)


def test_macro_metric_handles_singletons_exactly() -> None:
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    assert macro_entity_fbeta({"S1-1": set(), "S1-2": {"S2-1"}}, truth) == 1.0
    assert macro_entity_fbeta({"S1-1": {"S2-9"}, "S1-2": {"S2-1"}}, truth) == 0.5
    assert macro_entity_fbeta({"S1-2": set()}, truth) == 0.5


def test_threshold_selection_prefers_higher_threshold_on_tie() -> None:
    scored = pd.DataFrame(
        [
            ("S1-1", "S2-1", 0.90),
            ("S1-1", "S2-9", 0.40),
        ],
        columns=("source1_entity_id", "candidate_entity_id", "score"),
    )
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    predictions = predictions_at_threshold(scored, truth, 0.5)
    assert predictions == {"S1-1": frozenset({"S2-1"}), "S1-2": frozenset()}
    threshold, score, table = tune_threshold_from_oof(scored, truth, [0.5, 0.8])
    assert threshold == 0.8
    assert score == 1.0
    assert len(table) == 2
