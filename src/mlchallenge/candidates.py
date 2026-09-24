"""Transparent candidate generation and blocking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from mlchallenge.contracts import truth_mapping
from mlchallenge.normalization import candidate_text

CANDIDATE_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocking_score",
)


@dataclass(frozen=True)
class CandidateConfig:
    top_k_per_source: int = 50
    analyzer: str = "char_wb"
    ngram_min: int = 2
    ngram_max: int = 5
    min_df: int = 1
    max_features: int | None = 500_000

    def __post_init__(self) -> None:
        if self.top_k_per_source < 1:
            raise ValueError("top_k_per_source must be positive")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("invalid n-gram range")


def _text_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [candidate_text(row.business_name, row.business_address) for row in frame.itertuples()],
        index=frame.index,
        dtype="string",
    )


def _candidates_for_one_source(
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    target_source: str,
    config: CandidateConfig,
) -> pd.DataFrame:
    if source1.empty or targets.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    vectorizer = TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_features=config.max_features,
        dtype=np.float32,
        sublinear_tf=True,
    )
    target_matrix = vectorizer.fit_transform(_text_series(targets))
    query_matrix = vectorizer.transform(_text_series(source1))
    neighbor_count = min(config.top_k_per_source, len(targets))
    search = NearestNeighbors(
        n_neighbors=neighbor_count,
        algorithm="brute",
        metric="cosine",
        n_jobs=-1,
    ).fit(target_matrix)
    distances, indices = search.kneighbors(query_matrix, return_distance=True)

    source1_ids = source1["entity_id"].astype(str).to_numpy()
    target_ids = targets["entity_id"].astype(str).to_numpy()
    rows = []
    for query_index, source1_id in enumerate(source1_ids):
        for distance, target_index in zip(
            distances[query_index], indices[query_index], strict=True
        ):
            rows.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": target_ids[target_index],
                    "candidate_source": target_source,
                    "blocking_score": float(np.clip(1.0 - distance, 0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: CandidateConfig,
) -> pd.DataFrame:
    """Retrieve top-k candidates independently from Sources 2 and 3.

    Country is deliberately not a hard filter. It becomes a downstream comparison feature, which
    avoids assuming that labels are complete/consistent and supports unseen country strings.
    """

    result = pd.concat(
        [
            _candidates_for_one_source(source1, source2, target_source="source2", config=config),
            _candidates_for_one_source(source1, source3, target_source="source3", config=config),
        ],
        ignore_index=True,
    )
    if result.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return (
        result.sort_values(
            ["source1_entity_id", "blocking_score", "candidate_entity_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["source1_entity_id", "candidate_entity_id"])
        .reset_index(drop=True)
    )


def blocking_diagnostics(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    possible_target_count: int,
) -> dict[str, float | int]:
    truth = truth_mapping(ground_truth)
    candidate_map = {
        source1_id: frozenset(group["candidate_entity_id"].astype(str))
        for source1_id, group in candidates.groupby("source1_entity_id", sort=False)
    }
    positive_total = sum(len(matches) for matches in truth.values())
    positive_found = sum(
        len(matches & candidate_map.get(source1_id, frozenset()))
        for source1_id, matches in truth.items()
    )
    complete_entities = sum(
        matches <= candidate_map.get(source1_id, frozenset())
        for source1_id, matches in truth.items()
    )
    all_pairs = len(truth) * possible_target_count
    return {
        "source1_entities": len(truth),
        "candidate_pairs": len(candidates),
        "positive_pairs": positive_total,
        "positive_pairs_retrieved": positive_found,
        "positive_pair_recall": positive_found / positive_total if positive_total else 1.0,
        "entities_with_all_matches_retrieved": complete_entities,
        "complete_entity_recall": complete_entities / len(truth) if truth else 0.0,
        "reduction_ratio": 1.0 - (len(candidates) / all_pairs) if all_pairs else 1.0,
    }
