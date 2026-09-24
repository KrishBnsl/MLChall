"""Organizer-aligned entity-level evaluation functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def entity_fbeta(
    predicted: Iterable[str],
    expected: Iterable[str],
    *,
    beta: float = 0.5,
) -> float:
    """Score one Source 1 entity, including organizer-defined singleton behavior."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    predicted_set = frozenset(predicted)
    expected_set = frozenset(expected)
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set or not expected_set:
        return 0.0
    true_positive = len(predicted_set & expected_set)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(expected_set)
    beta_sq = beta**2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def macro_entity_fbeta(
    predictions: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    *,
    beta: float = 0.5,
) -> float:
    """Macro-average F-beta over exactly the Source 1 IDs in ground truth."""

    unknown = set(predictions) - set(truth)
    if unknown:
        raise ValueError(f"predictions contain unknown Source 1 IDs: {sorted(unknown)[:10]}")
    if not truth:
        raise ValueError("truth mapping is empty")
    scores = [
        entity_fbeta(predictions.get(key, ()), value, beta=beta) for key, value in truth.items()
    ]
    return float(np.mean(scores))


def predictions_at_threshold(
    scored_pairs: pd.DataFrame,
    source1_ids: Iterable[str],
    threshold: float,
) -> dict[str, frozenset[str]]:
    required = {"source1_entity_id", "candidate_entity_id", "score"}
    missing = required - set(scored_pairs.columns)
    if missing:
        raise ValueError(f"scored pairs missing columns: {sorted(missing)}")
    predictions = {str(entity_id): set() for entity_id in source1_ids}
    selected = scored_pairs.loc[scored_pairs["score"] >= threshold]
    for row in selected.itertuples(index=False):
        predictions[str(row.source1_entity_id)].add(str(row.candidate_entity_id))
    return {key: frozenset(value) for key, value in predictions.items()}


def tune_threshold_from_oof(
    scored_pairs: pd.DataFrame,
    truth: Mapping[str, Iterable[str]],
    thresholds: Iterable[float],
) -> tuple[float, float, pd.DataFrame]:
    """Choose a threshold from inner out-of-fold scores only.

    The caller is responsible for proving that `scored_pairs` are genuinely out-of-fold. Ties are
    resolved toward the higher threshold, reflecting the challenge's precision-heavy objective.
    """

    rows: list[dict[str, float]] = []
    for threshold in sorted({float(item) for item in thresholds}):
        predictions = predictions_at_threshold(scored_pairs, truth.keys(), threshold)
        score = macro_entity_fbeta(predictions, truth, beta=0.5)
        rows.append({"threshold": threshold, "macro_f0_5": score})
    if not rows:
        raise ValueError("at least one threshold is required")
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["macro_f0_5", "threshold"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    return float(best["threshold"]), float(best["macro_f0_5"]), table
