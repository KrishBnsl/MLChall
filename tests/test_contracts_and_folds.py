from __future__ import annotations

import pandas as pd
import pytest

from mlchallenge.contracts import ContractError, parse_id_list, validate_ground_truth
from mlchallenge.folds import build_fold_manifest


def test_parse_id_list_rejects_duplicates_and_empty_items() -> None:
    assert parse_id_list("") == ()
    assert parse_id_list("S2-1,S3-2") == ("S2-1", "S3-2")
    with pytest.raises(ContractError):
        parse_id_list("S2-1,S2-1")
    with pytest.raises(ContractError):
        parse_id_list("S2-1,")


def test_ground_truth_rejects_target_owned_by_multiple_entities(synthetic_frames) -> None:
    bad_truth = synthetic_frames["truth"].copy()
    bad_truth.loc[1, "matched_entity_ids"] = "S2-101"
    with pytest.raises(ContractError, match="assigned to both"):
        validate_ground_truth(
            bad_truth,
            synthetic_frames["source1"],
            synthetic_frames["source2"],
            synthetic_frames["source3"],
        )


def test_fold_manifest_is_complete_deterministic_and_entity_disjoint(synthetic_frames) -> None:
    frames = synthetic_frames
    first = build_fold_manifest(
        frames["source1"],
        frames["source2"],
        frames["source3"],
        frames["truth"],
        n_splits=2,
        seed=42,
    )
    second = build_fold_manifest(
        frames["source1"],
        frames["source2"],
        frames["source3"],
        frames["truth"],
        n_splits=2,
        seed=42,
    )
    pd.testing.assert_frame_equal(first, second)
    expected_ids = (
        set(frames["source1"]["entity_id"])
        | set(frames["source2"]["entity_id"])
        | set(frames["source3"]["entity_id"])
    )
    assert set(first["entity_id"]) == expected_ids
    assert not first["entity_id"].duplicated().any()
    by_id = first.set_index("entity_id")["fold"].to_dict()
    assert by_id["S1-001"] == by_id["S2-101"] == by_id["S3-201"]
    assert by_id["S1-002"] == by_id["S2-102"] == by_id["S3-202"]
    assert by_id["S1-003"] == by_id["S2-104"] == by_id["S3-204"]
