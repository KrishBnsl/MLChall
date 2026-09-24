"""Submission writing and local preflight checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from mlchallenge.contracts import ChallengeTestData, parse_id_list


def _serialize_mapping(
    source1_ids: Iterable[str],
    values: Mapping[str, Iterable[str]],
    *,
    value_column: str,
) -> pd.DataFrame:
    rows = []
    for source1_id in source1_ids:
        unique = sorted(set(values.get(str(source1_id), ())))
        rows.append({"source1_entity_id": str(source1_id), value_column: ",".join(unique)})
    return pd.DataFrame(rows, columns=("source1_entity_id", value_column))


def write_submission_files(
    test_data: ChallengeTestData,
    predictions: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source1_ids = test_data.source1["entity_id"].astype(str).tolist()
    matching = _serialize_mapping(source1_ids, predictions, value_column="matched_entity_ids")
    candidate = _serialize_mapping(source1_ids, candidates, value_column="candidate_entity_ids")
    issues = validate_submission_frames(matching, candidate, test_data)
    if issues:
        raise ValueError("submission preflight failed:\n- " + "\n- ".join(issues))
    matching_path = output / "matching_results.tsv"
    candidate_path = output / "candidate_pairs.tsv"
    matching.to_csv(matching_path, sep="\t", index=False)
    candidate.to_csv(candidate_path, sep="\t", index=False)
    return matching_path, candidate_path


def validate_submission_frames(
    matching: pd.DataFrame,
    candidates: pd.DataFrame,
    test_data: ChallengeTestData,
) -> list[str]:
    issues: list[str] = []
    expected_s1 = set(test_data.source1["entity_id"].astype(str))
    allowed_targets = set(test_data.source2["entity_id"].astype(str)) | set(
        test_data.source3["entity_id"].astype(str)
    )
    expected_columns = {
        "matching": ("source1_entity_id", "matched_entity_ids"),
        "candidates": ("source1_entity_id", "candidate_entity_ids"),
    }
    for name, frame in (("matching", matching), ("candidates", candidates)):
        if tuple(frame.columns) != expected_columns[name]:
            issues.append(
                f"{name} columns are {tuple(frame.columns)}, expected {expected_columns[name]}"
            )
            continue
        ids = frame["source1_entity_id"].astype(str)
        if ids.duplicated().any():
            issues.append(f"{name} contains duplicate Source 1 rows")
        if set(ids) != expected_s1:
            issues.append(f"{name} does not contain exactly all test Source 1 IDs")

    if issues:
        return issues

    candidate_by_s1: dict[str, set[str]] = {}
    for row in candidates.itertuples(index=False):
        try:
            values = set(parse_id_list(row.candidate_entity_ids))
        except ValueError as exc:
            issues.append(f"candidates row {row.source1_entity_id}: {exc}")
            continue
        unknown = values - allowed_targets
        if unknown:
            issues.append(
                f"candidates row {row.source1_entity_id} references unknown IDs: "
                f"{sorted(unknown)[:10]}"
            )
        candidate_by_s1[str(row.source1_entity_id)] = values

    for row in matching.itertuples(index=False):
        try:
            values = set(parse_id_list(row.matched_entity_ids))
        except ValueError as exc:
            issues.append(f"matching row {row.source1_entity_id}: {exc}")
            continue
        unknown = values - allowed_targets
        if unknown:
            issues.append(
                f"matching row {row.source1_entity_id} references unknown IDs: "
                f"{sorted(unknown)[:10]}"
            )
        outside_candidates = values - candidate_by_s1.get(str(row.source1_entity_id), set())
        if outside_candidates:
            issues.append(
                f"matching row {row.source1_entity_id} contains IDs absent from candidates: "
                f"{sorted(outside_candidates)[:10]}"
            )
    return issues
