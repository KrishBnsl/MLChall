"""Strict readers and data-contract validation for organizer TSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
SOURCE_PREFIX = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}


class ContractError(ValueError):
    """Raised when supplied challenge data violates a documented invariant."""


@dataclass(frozen=True)
class TrainingData:
    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame
    ground_truth: pd.DataFrame


@dataclass(frozen=True)
class ChallengeTestData:
    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame


def _read_tsv(path: Path, expected_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required challenge file is missing: {path}")
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    actual = tuple(frame.columns)
    if actual != expected_columns:
        raise ContractError(
            f"{path} columns are {actual}; expected exactly {expected_columns}. "
            "Confirm that the file is tab-separated and unmodified."
        )
    return frame


def parse_id_list(value: object) -> tuple[str, ...]:
    """Parse a comma-separated organizer ID list while rejecting ambiguous empty tokens."""

    text = str(value or "").strip()
    if not text:
        return ()
    values = tuple(part.strip() for part in text.split(","))
    if any(not part for part in values):
        raise ContractError(f"Malformed ID list with an empty item: {text!r}")
    if len(values) != len(set(values)):
        raise ContractError(f"ID list contains duplicates: {text!r}")
    return values


def validate_source_frame(frame: pd.DataFrame, source: str) -> None:
    if source not in SOURCE_PREFIX:
        raise ValueError(f"Unknown source name: {source}")
    if tuple(frame.columns) != RECORD_COLUMNS:
        raise ContractError(f"{source} has unexpected columns: {tuple(frame.columns)}")

    ids = frame["entity_id"].astype(str)
    if ids.eq("").any():
        raise ContractError(f"{source} contains empty entity_id values")
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ContractError(f"{source} contains duplicate entity IDs: {duplicates[:10]}")
    bad_prefix = ids[~ids.str.startswith(SOURCE_PREFIX[source])].tolist()
    if bad_prefix:
        raise ContractError(f"{source} contains IDs with the wrong prefix: {bad_prefix[:10]}")


def validate_ground_truth(
    ground_truth: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> None:
    if tuple(ground_truth.columns) != GROUND_TRUTH_COLUMNS:
        raise ContractError(f"ground truth has unexpected columns: {tuple(ground_truth.columns)}")

    s1_ids = set(source1["entity_id"].astype(str))
    target_ids = set(source2["entity_id"].astype(str)) | set(source3["entity_id"].astype(str))
    gt_s1 = ground_truth["source1_entity_id"].astype(str)
    if gt_s1.duplicated().any():
        duplicates = gt_s1[gt_s1.duplicated(keep=False)].unique().tolist()
        raise ContractError(f"ground truth repeats Source 1 IDs: {duplicates[:10]}")
    if set(gt_s1) != s1_ids:
        missing = sorted(s1_ids - set(gt_s1))[:10]
        unknown = sorted(set(gt_s1) - s1_ids)[:10]
        raise ContractError(
            f"ground truth Source 1 coverage mismatch; missing={missing}, unknown={unknown}"
        )

    owner_by_target: dict[str, str] = {}
    for row in ground_truth.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        for target_id in parse_id_list(row.matched_entity_ids):
            if target_id not in target_ids:
                raise ContractError(f"ground truth references missing Source 2/3 ID {target_id!r}")
            if not target_id.startswith(("S2-", "S3-")):
                raise ContractError(f"invalid ground-truth target prefix: {target_id!r}")
            previous_owner = owner_by_target.setdefault(target_id, source1_id)
            if previous_owner != source1_id:
                raise ContractError(
                    f"target {target_id!r} is assigned to both {previous_owner!r} "
                    f"and {source1_id!r}; review this ambiguity before modeling"
                )


def load_training_data(data_root: str | Path) -> TrainingData:
    root = Path(data_root) / "train"
    source1 = _read_tsv(root / "train_source1.tsv", RECORD_COLUMNS)
    source2 = _read_tsv(root / "train_source2.tsv", RECORD_COLUMNS)
    source3 = _read_tsv(root / "train_source3.tsv", RECORD_COLUMNS)
    ground_truth = _read_tsv(root / "train_ground_truth.tsv", GROUND_TRUTH_COLUMNS)
    validate_source_frame(source1, "source1")
    validate_source_frame(source2, "source2")
    validate_source_frame(source3, "source3")
    validate_ground_truth(ground_truth, source1, source2, source3)
    return TrainingData(source1, source2, source3, ground_truth)


def load_test_data(data_root: str | Path) -> ChallengeTestData:
    root = Path(data_root) / "test"
    source1 = _read_tsv(root / "test_source1.tsv", RECORD_COLUMNS)
    source2 = _read_tsv(root / "test_source2.tsv", RECORD_COLUMNS)
    source3 = _read_tsv(root / "test_source3.tsv", RECORD_COLUMNS)
    validate_source_frame(source1, "source1")
    validate_source_frame(source2, "source2")
    validate_source_frame(source3, "source3")
    return ChallengeTestData(source1, source2, source3)


def truth_mapping(ground_truth: pd.DataFrame) -> dict[str, frozenset[str]]:
    return {
        str(row.source1_entity_id): frozenset(parse_id_list(row.matched_entity_ids))
        for row in ground_truth.itertuples(index=False)
    }
