"""Dataset fingerprinting and descriptive audit with no model fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mlchallenge.contracts import (
    TrainingData,
    load_test_data,
    load_training_data,
    parse_id_list,
)
from mlchallenge.normalization import normalize_text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_summary(frame: pd.DataFrame) -> dict[str, Any]:
    normalized_name = frame["business_name"].map(normalize_text)
    normalized_address = frame["business_address"].map(normalize_text)
    return {
        "rows": int(len(frame)),
        "unique_entity_ids": int(frame["entity_id"].nunique()),
        "empty_business_name": int(normalized_name.eq("").sum()),
        "empty_business_address": int(normalized_address.eq("").sum()),
        "empty_country": int(frame["country"].map(normalize_text).eq("").sum()),
        "countries": {
            str(key): int(value)
            for key, value in frame["country"].astype(str).value_counts(dropna=False).items()
        },
        "normalized_exact_duplicate_names": int(normalized_name.duplicated(keep=False).sum()),
        "normalized_exact_duplicate_addresses": int(
            normalized_address.duplicated(keep=False).sum()
        ),
        "name_length": _numeric_summary(normalized_name.str.len()),
        "address_length": _numeric_summary(normalized_address.str.len()),
    }


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {key: 0.0 for key in ("min", "median", "p95", "max")}
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def _truth_summary(training: TrainingData) -> dict[str, Any]:
    counts = training.ground_truth["matched_entity_ids"].map(
        lambda value: len(parse_id_list(value))
    )
    target_owners: dict[str, int] = {}
    for value in training.ground_truth["matched_entity_ids"]:
        for target_id in parse_id_list(value):
            target_owners[target_id] = target_owners.get(target_id, 0) + 1
    return {
        "source1_entities": int(len(counts)),
        "singleton_entities": int(counts.eq(0).sum()),
        "matched_entities": int(counts.gt(0).sum()),
        "positive_pairs": int(counts.sum()),
        "matches_per_source1": _numeric_summary(counts),
        "targets_with_multiple_source1_owners": int(
            sum(owner_count > 1 for owner_count in target_owners.values())
        ),
    }


def audit_data(data_root: str | Path) -> dict[str, Any]:
    root = Path(data_root)
    training = load_training_data(root)
    test = load_test_data(root)
    files = sorted((*root.joinpath("train").glob("*.tsv"), *root.joinpath("test").glob("*.tsv")))
    return {
        "audit_version": 1,
        "files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        "train": {
            "source1": _record_summary(training.source1),
            "source2": _record_summary(training.source2),
            "source3": _record_summary(training.source3),
            "ground_truth": _truth_summary(training),
        },
        "test": {
            "source1": _record_summary(test.source1),
            "source2": _record_summary(test.source2),
            "source3": _record_summary(test.source3),
        },
    }


def write_audit(report: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
