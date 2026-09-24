from __future__ import annotations

import numpy as np

from mlchallenge.candidates import CandidateConfig, blocking_diagnostics, generate_candidates
from mlchallenge.features import FEATURE_COLUMNS, build_pair_features
from mlchallenge.modeling import BaselineMatcher, label_candidate_pairs


def test_candidates_do_not_hard_filter_unseen_country(synthetic_frames) -> None:
    frames = synthetic_frames
    candidates = generate_candidates(
        frames["source1"],
        frames["source2"],
        frames["source3"],
        CandidateConfig(top_k_per_source=2, max_features=None),
    )
    france_candidates = candidates.loc[candidates["source1_entity_id"] == "S1-003"]
    assert set(france_candidates["candidate_source"]) == {"source2", "source3"}
    assert "S2-104" in set(france_candidates["candidate_entity_id"])
    assert "S3-204" in set(france_candidates["candidate_entity_id"])


def test_features_labels_model_and_blocking_diagnostics(synthetic_frames) -> None:
    frames = synthetic_frames
    candidates = generate_candidates(
        frames["source1"],
        frames["source2"],
        frames["source3"],
        CandidateConfig(top_k_per_source=3, max_features=None),
    )
    features = build_pair_features(
        candidates, frames["source1"], frames["source2"], frames["source3"]
    )
    assert tuple(features.columns[2:]) == FEATURE_COLUMNS
    assert np.isfinite(features.loc[:, FEATURE_COLUMNS].to_numpy()).all()
    labels = label_candidate_pairs(features, frames["truth"])
    assert set(labels) == {0, 1}
    model = BaselineMatcher().fit(features, labels)
    scores = model.predict_scores(features)
    assert scores["score"].between(0, 1).all()
    diagnostics = blocking_diagnostics(
        candidates,
        frames["truth"],
        possible_target_count=len(frames["source2"]) + len(frames["source3"]),
    )
    assert diagnostics["positive_pair_recall"] == 1.0
    assert 0.0 <= diagnostics["reduction_ratio"] <= 1.0
