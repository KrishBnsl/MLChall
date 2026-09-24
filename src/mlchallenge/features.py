"""Auditable pairwise features for a non-neural baseline."""

from __future__ import annotations

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

from mlchallenge.normalization import compact_text, digit_set, normalize_text, token_set

FEATURE_COLUMNS = (
    "blocking_score",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_jaccard",
    "name_exact",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_exact",
    "address_digit_jaccard",
    "country_exact",
    "country_missing_either",
    "candidate_is_source3",
    "name_length_ratio",
    "address_length_ratio",
)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _length_ratio(left: str, right: str) -> float:
    maximum = max(len(left), len(right))
    return min(len(left), len(right)) / maximum if maximum else 1.0


def build_pair_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """Join candidate records and compute numeric, inspectable similarities."""

    target = pd.concat([source2, source3], ignore_index=True)
    left = source1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "s1_name",
            "business_address": "s1_address",
            "country": "s1_country",
        }
    )
    right = target.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "candidate_name",
            "business_address": "candidate_address",
            "country": "candidate_country",
        }
    )
    merged = candidates.merge(left, on="source1_entity_id", how="left", validate="many_to_one")
    merged = merged.merge(right, on="candidate_entity_id", how="left", validate="many_to_one")
    if merged[["s1_name", "candidate_name"]].isna().any().any():
        raise ValueError("candidate pairs reference IDs absent from source records")

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        left_name = normalize_text(row.s1_name)
        right_name = normalize_text(row.candidate_name)
        left_address = normalize_text(row.s1_address)
        right_address = normalize_text(row.candidate_address)
        left_country = normalize_text(row.s1_country)
        right_country = normalize_text(row.candidate_country)
        rows.append(
            {
                "source1_entity_id": str(row.source1_entity_id),
                "candidate_entity_id": str(row.candidate_entity_id),
                "blocking_score": float(row.blocking_score),
                "name_ratio": ratio(left_name, right_name) / 100.0,
                "name_token_set_ratio": token_set_ratio(left_name, right_name) / 100.0,
                "name_token_jaccard": _jaccard(token_set(left_name), token_set(right_name)),
                "name_exact": float(
                    bool(left_name) and compact_text(left_name) == compact_text(right_name)
                ),
                "address_ratio": ratio(left_address, right_address) / 100.0,
                "address_token_set_ratio": token_set_ratio(left_address, right_address) / 100.0,
                "address_token_jaccard": _jaccard(
                    token_set(left_address), token_set(right_address)
                ),
                "address_exact": float(
                    bool(left_address) and compact_text(left_address) == compact_text(right_address)
                ),
                "address_digit_jaccard": _jaccard(
                    digit_set(row.s1_address), digit_set(row.candidate_address)
                ),
                "country_exact": float(bool(left_country) and left_country == right_country),
                "country_missing_either": float(not left_country or not right_country),
                "candidate_is_source3": float(str(row.candidate_entity_id).startswith("S3-")),
                "name_length_ratio": _length_ratio(left_name, right_name),
                "address_length_ratio": _length_ratio(left_address, right_address),
            }
        )
    return pd.DataFrame(
        rows,
        columns=("source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS),
    )
