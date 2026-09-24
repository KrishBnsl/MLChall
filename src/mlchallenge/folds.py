"""Deterministic entity-group fold assignment and leakage assertions."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd

from mlchallenge.contracts import parse_id_list, truth_mapping
from mlchallenge.normalization import normalize_text

MANIFEST_COLUMNS = ("entity_id", "source", "fold", "entity_group")


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _assign_balanced(
    items: list[tuple[str, str, int]],
    *,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """Greedily balance stratum counts and record weights with stable randomized order."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    fold_weight = [0] * n_splits
    stratum_counts: dict[str, list[int]] = defaultdict(lambda: [0] * n_splits)
    result: dict[str, int] = {}
    ordered = sorted(items, key=lambda item: (item[1], _stable_rank(item[0], seed)))
    for item_id, stratum, weight in ordered:
        selected = min(
            range(n_splits),
            key=lambda fold: (stratum_counts[stratum][fold], fold_weight[fold], fold),
        )
        result[item_id] = selected
        stratum_counts[stratum][selected] += 1
        fold_weight[selected] += weight
    return result


def build_fold_manifest(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """Assign every training record to one fold without separating labeled entities."""

    if len(source1) < n_splits:
        raise ValueError("n_splits cannot exceed the number of Source 1 records")
    truth = truth_mapping(ground_truth)
    country_by_s1 = source1.set_index("entity_id")["country"].astype(str).to_dict()
    group_items = []
    for source1_id, matches in truth.items():
        country = normalize_text(country_by_s1[source1_id]) or "<missing-country>"
        cardinality = "singleton" if not matches else "matched"
        group_items.append((source1_id, f"{country}|{cardinality}", 1 + len(matches)))
    fold_by_group = _assign_balanced(group_items, n_splits=n_splits, seed=seed)

    rows: list[dict[str, object]] = []
    assigned_targets: set[str] = set()
    for source1_id, matches in truth.items():
        fold = fold_by_group[source1_id]
        rows.append(
            {
                "entity_id": source1_id,
                "source": "source1",
                "fold": fold,
                "entity_group": source1_id,
            }
        )
        for target_id in sorted(matches):
            assigned_targets.add(target_id)
            rows.append(
                {
                    "entity_id": target_id,
                    "source": "source2" if target_id.startswith("S2-") else "source3",
                    "fold": fold,
                    "entity_group": source1_id,
                }
            )

    unmatched_items: list[tuple[str, str, int]] = []
    source_by_target: dict[str, str] = {}
    for source_name, frame in (("source2", source2), ("source3", source3)):
        for row in frame.itertuples(index=False):
            entity_id = str(row.entity_id)
            source_by_target[entity_id] = source_name
            if entity_id not in assigned_targets:
                country = normalize_text(row.country) or "<missing-country>"
                unmatched_items.append((entity_id, f"{source_name}|{country}|unmatched", 1))
    unmatched_folds = _assign_balanced(unmatched_items, n_splits=n_splits, seed=seed + 1)
    for entity_id, fold in unmatched_folds.items():
        rows.append(
            {
                "entity_id": entity_id,
                "source": source_by_target[entity_id],
                "fold": fold,
                "entity_group": f"unmatched:{entity_id}",
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
        ["fold", "source", "entity_id"], kind="mergesort"
    )
    manifest = manifest.reset_index(drop=True)
    validate_fold_manifest(manifest, source1, source2, source3, ground_truth, n_splits=n_splits)
    return manifest


def validate_fold_manifest(
    manifest: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    n_splits: int,
) -> None:
    if tuple(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(f"manifest has unexpected columns: {tuple(manifest.columns)}")
    if manifest["entity_id"].duplicated().any():
        raise ValueError("fold manifest contains duplicate entity IDs")
    expected = set(source1["entity_id"]) | set(source2["entity_id"]) | set(source3["entity_id"])
    actual = set(manifest["entity_id"])
    if actual != expected:
        raise ValueError(
            f"fold manifest coverage mismatch: missing={sorted(expected - actual)[:10]}, "
            f"unknown={sorted(actual - expected)[:10]}"
        )
    folds = set(manifest["fold"].astype(int))
    if not folds <= set(range(n_splits)) or len(folds) != n_splits:
        raise ValueError(f"manifest fold values are invalid or incomplete: {sorted(folds)}")

    fold_by_id = manifest.set_index("entity_id")["fold"].astype(int).to_dict()
    for row in ground_truth.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        for target_id in parse_id_list(row.matched_entity_ids):
            if fold_by_id[target_id] != fold_by_id[source1_id]:
                raise ValueError(
                    f"entity leakage: {source1_id} and matched {target_id} occupy different folds"
                )


def select_fold_partition(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    source: str,
    held_out_fold: int,
    validation: bool,
) -> pd.DataFrame:
    """Select records for one train/validation side from a verified manifest."""

    source_manifest = manifest.loc[manifest["source"] == source, ["entity_id", "fold"]]
    wanted = source_manifest.loc[
        source_manifest["fold"].eq(held_out_fold)
        if validation
        else source_manifest["fold"].ne(held_out_fold),
        "entity_id",
    ]
    return frame.loc[frame["entity_id"].isin(set(wanted))].reset_index(drop=True)
